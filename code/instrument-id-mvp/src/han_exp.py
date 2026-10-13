"""
Reproduction of Han, Kim, Lee 2017 (arXiv 1605.09507).

Training : random 43-frame (1 s) crops from 3-second training spectrograms
Inference: sliding 43-frame windows (50% overlap) over full test-clip audio,
           predictions aggregated by normalised sum (strategy S2), threshold 0.5

Usage:
    python src/han_exp.py
    python src/han_exp.py --epochs 30 --seeds 5 --threshold 0.5
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as T
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import CLASSES, CLASS_TO_IDX, _scan_testing_dir, train_val_split
from features import (
    SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH,
    compute_mel_spectrogram,
)
from model import Han2017CNN, HarmonicHan2017CNN, DeltaHarmonicCNN, DeepHarmonicCNN

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "data" / "IRMAS" / "training"
DEFAULT_TEST_ROOT  = PROJECT_ROOT / "data" / "IRMAS" / "testing"
CKPT_DIR           = PROJECT_ROOT / "checkpoints"

WIN_FRAMES = 43   # 1 second at 22050 Hz / 512 hop


# ── Datasets ──────────────────────────────────────────────────────────────────

class HanTrainDataset(Dataset):
    """Returns random 43-frame crops from cached 130-frame training spectrograms."""
    def __init__(self, wav_paths, labels):
        self.wav_paths = wav_paths
        self.labels    = labels

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        mel = compute_mel_spectrogram(self.wav_paths[idx])  # [128, 130]
        t   = mel.shape[1]
        if t >= WIN_FRAMES:
            start = random.randint(0, t - WIN_FRAMES)
            mel   = mel[:, start:start + WIN_FRAMES]
        else:
            mel = np.pad(mel, ((0, 0), (0, WIN_FRAMES - t)), mode="constant",
                         constant_values=np.log(1e-9))
        x = torch.from_numpy(mel).unsqueeze(0).float()       # [1, 128, 43]
        y = torch.from_numpy(self.labels[idx]).float()
        return x, y


class HanValDataset(Dataset):
    """Returns centre-crop for validation (deterministic)."""
    def __init__(self, wav_paths, labels):
        self.wav_paths = wav_paths
        self.labels    = labels

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        mel = compute_mel_spectrogram(self.wav_paths[idx])  # [128, 130]
        t   = mel.shape[1]
        start = max(0, (t - WIN_FRAMES) // 2)
        mel   = mel[:, start:start + WIN_FRAMES]
        if mel.shape[1] < WIN_FRAMES:
            mel = np.pad(mel, ((0, 0), (0, WIN_FRAMES - mel.shape[1])),
                         mode="constant", constant_values=np.log(1e-9))
        x = torch.from_numpy(mel).unsqueeze(0).float()
        y = torch.from_numpy(self.labels[idx]).float()
        return x, y


# ── Sliding-window test inference ─────────────────────────────────────────────

_mel_tf = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
)
_resample_cache: dict[int, T.Resample] = {}


def _full_mel(wav_path: Path) -> np.ndarray:
    """Compute full log-mel spectrogram without time truncation."""
    data, sr = sf.read(wav_path, always_2d=True)
    waveform = torch.from_numpy(data.mean(axis=1)).float().unsqueeze(0)
    if sr != SAMPLE_RATE:
        if sr not in _resample_cache:
            _resample_cache[sr] = T.Resample(sr, SAMPLE_RATE)
        waveform = _resample_cache[sr](waveform)
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    mel = _mel_tf(waveform)                  # [1, 128, T]
    return torch.log(mel + 1e-9).squeeze(0).numpy()  # [128, T]


def predict_sliding(model, wav_paths, labels, device, threshold=0.5):
    """S2 aggregation: sum per-window sigmoid scores, normalise, threshold."""
    model.eval()
    preds, targets = [], []
    for wav_path, label in tqdm(zip(wav_paths, labels), total=len(wav_paths),
                                desc="sliding-window eval"):
        mel   = _full_mel(wav_path)          # [128, T_full]
        T_f   = mel.shape[1]
        hop   = max(1, WIN_FRAMES // 2)      # 50% overlap = 21 frames
        starts = list(range(0, max(1, T_f - WIN_FRAMES + 1), hop))

        scores = np.zeros(len(CLASSES), dtype=np.float32)
        with torch.no_grad():
            for start in starts:
                w = mel[:, start:start + WIN_FRAMES]
                if w.shape[1] < WIN_FRAMES:
                    w = np.pad(w, ((0, 0), (0, WIN_FRAMES - w.shape[1])),
                               mode="constant", constant_values=np.log(1e-9))
                x = torch.from_numpy(w).unsqueeze(0).unsqueeze(0).float().to(device)
                prob = torch.sigmoid(model(x)).cpu().numpy().squeeze()
                scores += prob

        # S2 normalisation
        mx = scores.max()
        if mx > 0:
            scores /= mx
        preds.append((scores >= threshold).astype(int))
        targets.append(label.astype(int))

    return np.vstack(preds), np.vstack(targets)


# ── Training ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def find_training_root(base: Path) -> Path:
    if any((base / c).is_dir() for c in CLASSES):
        return base
    for child in sorted(base.iterdir()):
        if child.is_dir() and any((child / c).is_dir() for c in CLASSES):
            return child
    raise RuntimeError(f"Cannot find instrument folders under {base}.")


def _scan_training_dir(train_root: Path):
    wav_paths, labels = [], []
    for class_dir in sorted(train_root.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in CLASS_TO_IDX:
            continue
        idx = CLASS_TO_IDX[class_dir.name]
        for wav in sorted(class_dir.glob("*.wav")):
            label = np.zeros(len(CLASSES), dtype=np.float32)
            label[idx] = 1.0
            wav_paths.append(wav)
            labels.append(label)
    return wav_paths, labels


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def val_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += criterion(model(x), y).item() * x.size(0)
    return total / len(loader.dataset)


def val_f1(model, loader, device, threshold=0.3):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
            preds.append((probs >= threshold).astype(int))
            targets.append(y.numpy().astype(int))
    p, t = np.vstack(preds), np.vstack(targets)
    return f1_score(t, p, average="macro", zero_division=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def train_one_config(name, model_cls, train_paths, train_labels, val_loader,
                     args, device):
    """Train model_cls for args.seeds seeds, return (scores, best_state, best_seed)."""
    n = sum(p.numel() for p in model_cls().parameters() if p.requires_grad)
    print(f"\n=== {name}  ({n:,} params) ===")

    best_val_f1, best_state, best_seed = -1.0, None, -1
    seed_scores = []

    for seed in tqdm(range(args.seeds), desc=f"{name[:20]} seeds", unit="seed"):
        set_seed(seed)
        train_loader = DataLoader(
            HanTrainDataset(train_paths, train_labels),
            batch_size=args.batch_size, shuffle=True, num_workers=0,
        )
        model     = model_cls().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"seed {seed}", leave=False)
        best_vl, best_epoch_state, bad_epochs, stopped_at = float("inf"), None, 0, args.epochs
        for epoch in epoch_bar:
            tl = train_epoch(model, train_loader, criterion, optimizer, device)
            vf = val_f1(model, val_loader, device)
            epoch_bar.set_postfix(train=f"{tl:.4f}", val_f1=f"{vf:.4f}")
            if args.early_stop:
                # Han et al.: "stopped when the validation loss did not decrease
                # for more than two epochs"; keep the best-val-loss weights
                vl = val_loss(model, val_loader, criterion, device)
                if vl < best_vl:
                    best_vl, bad_epochs = vl, 0
                    best_epoch_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad_epochs += 1
                    if bad_epochs > args.early_stop:
                        stopped_at = epoch
                        break
        if args.early_stop and best_epoch_state is not None:
            model.load_state_dict(best_epoch_state)
            tqdm.write(f"  [{name} seed {seed}] early stop at epoch {stopped_at}")

        final_f1 = val_f1(model, val_loader, device)
        seed_scores.append(final_f1)
        tqdm.write(f"  [{name} seed {seed}] val F1-macro: {final_f1:.4f}")
        CKPT_DIR.mkdir(exist_ok=True)
        torch.save({
            "model_cls":   model_cls.__name__,
            "config_name": name,
            "seed":        seed,
            "f1_macro":    final_f1,
            "epochs":      args.epochs,
            "split":       "recording",
            "state_dict":  {k: v.cpu() for k, v in model.state_dict().items()},
        }, CKPT_DIR / f"{name}{args.tag}_rec_seed{seed}.pt")

        if final_f1 > best_val_f1:
            best_val_f1 = final_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_seed   = seed

    print(f"Val F1-macro: mean={np.mean(seed_scores):.4f} ± {np.std(seed_scores):.4f}"
          f"  (best seed {best_seed}: {best_val_f1:.4f})")

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / f"{name}{args.tag}_rec_best.pt"   # recording split; old *_best.pt kept
    torch.save({
        "model_cls":   model_cls.__name__,
        "config_name": name,
        "seed":        best_seed,
        "f1_macro":    best_val_f1,
        "epochs":      args.epochs,
        "state_dict":  best_state,
    }, ckpt_path)
    print(f"Saved → {ckpt_path.name}")

    return seed_scores, best_state, best_seed


def eval_on_test(name, model_cls, state, test_paths, test_labels, device, threshold):
    model = model_cls().to(device)
    model.load_state_dict(state)
    preds, targets = predict_sliding(model, test_paths, test_labels, device, threshold)
    macro   = f1_score(targets, preds, average="macro", zero_division=0)
    micro   = f1_score(targets, preds, average="micro", zero_division=0)
    per_cls = f1_score(targets, preds, average=None,    zero_division=0)
    return macro, micro, per_cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--seeds",      type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--data-dir",   type=Path,  default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--test-dir",   type=Path,  default=DEFAULT_TEST_ROOT)
    parser.add_argument("--early-stop", type=int, default=0,
                        help="patience in epochs on val loss (Han et al. use 2); 0 = off")
    parser.add_argument("--tag", type=str, default="",
                        help="suffix for checkpoint names, e.g. _es")
    parser.add_argument("--test-all-seeds", action="store_true",
                        help="evaluate every saved seed on the test set → mean ± std")
    parser.add_argument("--only",       type=str,   default=None,
                        help="comma-separated name substrings to run only those configs")
    args = parser.parse_args()

    device = get_device()
    print(f"Device    : {device}")
    print(f"Threshold : {args.threshold}")

    train_root = find_training_root(args.data_dir)
    all_paths, all_labels = _scan_training_dir(train_root)
    train_paths, train_labels, val_paths, val_labels = train_val_split(
        all_paths, all_labels, seed=42
    )
    print(f"Train: {len(train_paths)}  Val: {len(val_paths)}")

    val_loader = DataLoader(
        HanValDataset(val_paths, val_labels),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    configs = [
        ("Han2017CNN",              Han2017CNN),
        ("HarmonicHan2017CNN",      HarmonicHan2017CNN),
        ("DeltaHarmonicCNN_1sec",   DeltaHarmonicCNN),
        ("DeepHarmonicCNN_1sec",    DeepHarmonicCNN),
    ]
    if args.only:
        filters = [f.strip() for f in args.only.split(",")]
        names   = {n for n, _ in configs}
        # an exact config name matches only itself; anything else is a substring
        configs = [(n, c) for n, c in configs
                   if any(f == n if f in names else f in n for f in filters)]

    all_results = {}
    for name, model_cls in configs:
        scores, best_state, best_seed = train_one_config(
            name, model_cls, train_paths, train_labels, val_loader, args, device
        )
        all_results[name] = dict(scores=scores, state=best_state, cls=model_cls)

    if args.test_all_seeds:
        from ensemble_test import SCORE_DIR, clip_scores
        test_paths, test_labels = _scan_testing_dir(args.test_dir)
        y = np.vstack(test_labels).astype(int)
        print(f"\n─── Test set, every seed ({args.seeds} seeds, 1-s sliding window, S2) ───")
        for name in all_results:
            for t in (0.3, 0.4, 0.5):
                mac, mic = [], []
                for seed in range(args.seeds):
                    stem = f"{name}{args.tag}_rec_seed{seed}"
                    if t == 0.3:
                        (SCORE_DIR / f"{stem}.npy").unlink(missing_ok=True)
                    pred = (clip_scores(stem, test_paths, device) >= t).astype(int)
                    mac.append(f1_score(y, pred, average="macro", zero_division=0))
                    mic.append(f1_score(y, pred, average="micro", zero_division=0))
                print(f"  {name}  thr {t:.1f}:  macro {np.mean(mac):.4f} ± {np.std(mac):.4f}"
                      f"  [{', '.join(f'{v:.3f}' for v in mac)}]"
                      f"   micro {np.mean(mic):.4f} ± {np.std(mic):.4f}")
        return

    # ── Test-set comparison ───────────────────────────────────────────────────
    if not args.test_dir.exists():
        print(f"\nTest dir not found: {args.test_dir}  — skipping test evaluation")
        return

    test_paths, test_labels = _scan_testing_dir(args.test_dir)
    print(f"\nEvaluating on test set (sliding window, threshold={args.threshold}) — {len(test_paths)} clips\n")

    print(f"{'Config':<25}  {'Val mean':>8}  {'Val std':>7}  {'Test macro':>10}  {'Test micro':>10}")
    print("─" * 68)
    for name, r in all_results.items():
        macro, micro, per_cls = eval_on_test(
            name, r["cls"], r["state"], test_paths, test_labels, device, args.threshold
        )
        r["test_macro"] = macro
        r["test_micro"] = micro
        r["per_cls"]    = per_cls
        vm = np.mean(r["scores"])
        vs = np.std(r["scores"])
        print(f"{name:<25}  {vm:>8.4f}  {vs:>7.4f}  {macro:>10.4f}  {micro:>10.4f}")

    print(f"\nPer-class F1 (test set, threshold={args.threshold}):")
    names = list(all_results.keys())
    header = f"  {'Class':<6}" + "".join(f"  {n[:18]:>18}" for n in names)
    print(header)
    print("  " + "─" * (8 + 20 * len(names)))
    for i, cls in enumerate(CLASSES):
        vals = [all_results[n]["per_cls"][i] for n in names]
        best = max(vals)
        row  = f"  {cls:<6}"
        for v in vals:
            marker = "*" if v == best and best > min(vals) + 0.005 else " "
            row   += f"  {v:>17.3f}{marker}"
        print(row)

    print("\nPublished SOTA:")
    print("  Han et al. 2017 (original)   macro=0.503  micro=0.602")


if __name__ == "__main__":
    main()
