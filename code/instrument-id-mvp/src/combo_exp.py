"""
Combined augmentation and scheduler experiments.

Configurations (all 5 seeds × 30 epochs, same split):
  1. HarmonicCNN                          — baseline
  2. HarmonicCNN + Mixup                  — label-mixing augmentation
  3. HarmonicCNN + pitch±2 + SpecAugment  — two complementary augmentations
  4. HarmonicCNN + cosine LR              — better optimisation schedule
  5. DeltaHarmonicCNN + cosine LR         — fix for delta instability

Usage:
    python src/combo_exp.py
    python src/combo_exp.py --epochs 30 --seeds 5
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from progress import Progress
from features import compute_mel_spectrogram
from dataset import CLASSES, IRMASDataset, genre_id, load_training_dataset, train_val_split
from model import (BasicCNN, HarmonicCNN, DeltaHarmonicCNN, DeepHarmonicCNN, SEDeepHarmonicCNN,
                   Han2017CNN, HarmonicHan2017CNN,
                   DeepExactHarmonicCNN, DeepExactOctaveCNN, DeepMelCNN,
                   DeepExactOctaveMaxCNN, DeepMelMaxCNN,
                   DeepMelFreqCNN, DeepMelPonsCNN,
                   DeepMelLogPonsCNN, DeepMelHPSPonsCNN,
                   DeepMelMultiLogPonsCNN, DeepMelLogPonsMaxCNN,
                   PonsMultiLayerCNN, LogPonsMultiLayerCNN)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "data" / "IRMAS" / "training"

EVAL_UNITS  = 3     # progress weight of one test evaluation (~10 s with the RAM mel cache ≈ a few epochs)
PROGRESS: Progress | None = None

MIXUP_ALPHA = 0.4   # Beta(α, α) parameter for mixup
POLYMIX_P       = 0.5   # fraction of training clips that get a second source mixed in
POLYMIX_GAIN_DB = 6.0   # relative level of the two sources: uniform in ±6 dB


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_training_root(base: Path) -> Path:
    if any((base / c).is_dir() for c in CLASSES):
        return base
    for child in sorted(base.iterdir()):
        if child.is_dir() and any((child / c).is_dir() for c in CLASSES):
            return child
    raise RuntimeError(f"Cannot find instrument folders under {base}.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ── GPU-resident data ──────────────────────────────────────────────────────────

_PRELOADED: dict = {}


def preload_mels(paths, device) -> torch.Tensor:
    """All cached log-mels of `paths` as one [N, 1, 128, 130] tensor on `device`.

    The whole IRMAS training set is ~400 MB, so it fits in GPU memory and batches
    no longer wait on per-file .npy reads in the main process (GPU sat at
    ~25–45% with the DataLoader). Freed when the process exits.
    """
    key = (tuple(str(p) for p in paths), str(device))
    if key not in _PRELOADED:
        mels = np.stack([compute_mel_spectrogram(p) for p in tqdm(paths, desc="preload", leave=False)])
        _PRELOADED[key] = torch.from_numpy(mels).float().unsqueeze(1).to(device)
    return _PRELOADED[key]


class GPULoader:
    """DataLoader stand-in serving batches of GPU-resident tensors.

    Same semantics as DataLoader(batch_size, shuffle, drop_last=False); the
    shuffle order comes from torch.randperm, so set_seed() makes it reproducible.
    x stays on the GPU; y (and genre g) stay on the CPU like DataLoader output.
    """

    def __init__(self, x, y, g=None, batch_size=32, shuffle=False):
        self.x, self.y, self.g = x, y, g
        self.batch_size, self.shuffle = batch_size, shuffle
        self.dataset = y            # len(loader.dataset) is used to average losses

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def __iter__(self):
        n = len(self.y)
        order = torch.randperm(n) if self.shuffle else torch.arange(n)
        for i in range(0, n, self.batch_size):
            idx = order[i:i + self.batch_size]
            x = self.x[idx.to(self.x.device)]
            if self.g is None:
                yield x, self.y[idx]
            else:
                yield x, self.y[idx], self.g[idx]


# ── Training loops ─────────────────────────────────────────────────────────────

def polymix(x: torch.Tensor, y: torch.Tensor, genre: torch.Tensor | None = None,
            p: float = POLYMIX_P, gain_db: float = POLYMIX_GAIN_DB):
    """Synthetic polyphony: add a second training clip to a random subset.

    Unlike Mixup (which interpolates log-mel values and labels), sources are
    summed in linear power — log(e^a + g·e^b) — as two instruments playing
    together would be, and the label is the union of both labels.
    x: [B, 1, F, T] log-mel, y: [B, C] multi-hot.
    genre: [B] genre ids — if given, partners come from the same genre only, so
    mixtures resemble real ensembles (vio+cel in classical, voi+gel in pop-rock).
    """
    B = x.size(0)
    if genre is None:
        partner = torch.randperm(B, device=x.device)
    else:
        partner = torch.arange(B, device=x.device)
        for g in genre.unique():
            idx = (genre == g).nonzero().squeeze(1)
            partner[idx] = idx[torch.randperm(len(idx), device=x.device)]
    mixed   = torch.rand(B, device=x.device) < p
    gain    = (torch.rand(B, device=x.device) * 2 - 1) * gain_db * (np.log(10) / 10)  # dB → ln
    x_mix = torch.logaddexp(x, x[partner] + gain.view(B, 1, 1, 1))
    y_mix = torch.maximum(y, y[partner])
    m = mixed.view(B, 1, 1, 1)
    return torch.where(m, x_mix, x), torch.where(mixed.view(B, 1), y_mix, y)


def polymix_multi(x: torch.Tensor, y: torch.Tensor, genre: torch.Tensor | None = None,
                  probs=(0.8, 0.4), gain_db: float = POLYMIX_GAIN_DB):
    """PolyMix with up to len(probs)+1 sources.

    Pass k adds one more source (drawn from the *original* batch) to a random
    fraction probs[k] of the clips: with (0.8, 0.4) about 20% stay solo,
    ~48% become duos and ~32% trios — closer to the 1–3 annotated instruments
    of IRMAS test excerpts than the single 50% duo pass of polymix().
    """
    B = x.size(0)
    x0, y0 = x, y
    for p in probs:
        if genre is None:
            partner = torch.randperm(B, device=x.device)
        else:
            partner = torch.arange(B, device=x.device)
            for g in genre.unique():
                idx = (genre == g).nonzero().squeeze(1)
                partner[idx] = idx[torch.randperm(len(idx), device=x.device)]
        mixed = torch.rand(B, device=x.device) < p
        gain  = (torch.rand(B, device=x.device) * 2 - 1) * gain_db * (np.log(10) / 10)
        x = torch.where(mixed.view(B, 1, 1, 1),
                        torch.logaddexp(x, x0[partner] + gain.view(B, 1, 1, 1)), x)
        y = torch.where(mixed.view(B, 1), torch.maximum(y, y0[partner]), y)
    return x, y


def run_epoch(model, loader, criterion, optimizer, device, use_polymix=False,
              polymix_probs=None):
    model.train()
    total = 0.0
    with torch.set_grad_enabled(True):
        for batch in tqdm(loader, leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            if use_polymix:
                genre = batch[2].to(device) if len(batch) > 2 else None
                x, y = (polymix(x, y, genre) if polymix_probs is None
                        else polymix_multi(x, y, genre, probs=polymix_probs))
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def run_epoch_mixup(model, loader, criterion, optimizer, device, alpha=MIXUP_ALPHA):
    """Training epoch with Mixup augmentation."""
    model.train()
    total = 0.0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        lam = float(np.random.beta(alpha, alpha))
        idx = torch.randperm(x.size(0), device=device)
        x_mix = lam * x + (1.0 - lam) * x[idx]
        y_mix = lam * y + (1.0 - lam) * y[idx]
        loss = criterion(model(x_mix), y_mix)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def run_val_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            total += criterion(model(x), y).item() * x.size(0)
    return total / len(loader.dataset)


def evaluate(model, loader, device, threshold=0.3):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
            preds.append((probs >= threshold).astype(int))
            targets.append(y.numpy().astype(int))
    p, t = np.vstack(preds), np.vstack(targets)
    return (
        f1_score(t, p, average="macro", zero_division=0),
        f1_score(t, p, average=None,   zero_division=0),
    )


def collect_val_probs(model, loader, device):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.sigmoid(model(x.to(device))).cpu().numpy()
            probs.append(p)
            targets.append(y.numpy().astype(int))
    return np.vstack(probs), np.vstack(targets)


def tune_threshold(probs, targets, thresholds=None):
    if thresholds is None:
        thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_thr, best_f1 = 0.30, 0.0
    rows = []
    for thr in thresholds:
        f1 = f1_score(targets, (probs >= thr).astype(int), average="macro", zero_division=0)
        rows.append((thr, f1))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1, rows


# ── Generic seed runner ────────────────────────────────────────────────────────

def train_seeds(name, model_cls, train_paths, train_labels, val_paths, val_labels,
                args, device,
                pitch_shift_bins: int = 0,
                spec_augment: bool = False,
                use_mixup: bool = False,
                use_polymix: bool = False,
                genre_mix: bool = False,
                polymix_probs=None,
                cosine_lr: bool = False):

    print(f"\n=== {name} ({args.seeds} seeds × {args.epochs} epochs) ===")
    tags = []
    if pitch_shift_bins: tags.append(f"pitch±{pitch_shift_bins}")
    if spec_augment:     tags.append("SpecAugment")
    if use_mixup:        tags.append(f"Mixup(α={MIXUP_ALPHA})")
    if polymix_probs:    tags.append(f"PolyMix-multi(p={polymix_probs})")
    if use_polymix:      tags.append(f"PolyMix(p={POLYMIX_P}, ±{POLYMIX_GAIN_DB:g} dB"
                                     + (", same genre)" if genre_mix else ")"))
    if cosine_lr:        tags.append("CosineLR")
    if tags:
        print(f"    [{', '.join(tags)}]")
    n_params = sum(p.numel() for p in model_cls().parameters() if p.requires_grad)
    print(f"    params: {n_params:,}")

    # CPU-side augmentations (pitch shift, SpecAugment) live in IRMASDataset;
    # everything else trains from GPU-resident tensors.
    on_gpu = not (pitch_shift_bins or spec_augment)
    if on_gpu:
        x_tr = preload_mels(train_paths, device)
        y_tr = torch.from_numpy(np.stack(train_labels)).float()
        g_tr = torch.tensor([genre_id(p) for p in train_paths]) if genre_mix else None
        val_loader = GPULoader(preload_mels(val_paths, device),
                               torch.from_numpy(np.stack(val_labels)).float(),
                               batch_size=args.batch_size)
    else:
        val_ds     = IRMASDataset(val_paths, val_labels)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    scores, per_class_all = [], []
    best_f1    = -1.0
    best_state = None
    best_seed  = -1

    seed_bar = tqdm(range(args.seeds), desc=f"{name[:20]} seeds", unit="seed", position=1, leave=False)
    for seed in seed_bar:
        set_seed(seed)

        if on_gpu:
            train_loader = GPULoader(x_tr, y_tr, g_tr, batch_size=args.batch_size, shuffle=True)
        else:
            train_ds = IRMASDataset(
                train_paths, train_labels,
                pitch_shift_bins=pitch_shift_bins,
                spec_augment=spec_augment,
                genres=[genre_id(p) for p in train_paths] if genre_mix else None,
            )
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      shuffle=True, num_workers=0)

        model     = model_cls().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-5)
            if cosine_lr else None
        )

        epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"seed {seed}", unit="ep",
                         position=2, leave=False)
        for epoch in epoch_bar:
            if use_mixup:
                tl = run_epoch_mixup(model, train_loader, criterion, optimizer, device)
            else:
                tl = run_epoch(model, train_loader, criterion, optimizer, device,
                               use_polymix=use_polymix, polymix_probs=polymix_probs)
            vl = run_val_epoch(model, val_loader, criterion, device)
            if scheduler:
                scheduler.step()
            lr_now = optimizer.param_groups[0]["lr"]
            epoch_bar.set_postfix(train=f"{tl:.4f}", val=f"{vl:.4f}", lr=f"{lr_now:.1e}")
            if PROGRESS:
                PROGRESS.advance(1, f"{name} s{seed} ep{epoch}")

        f1_macro, f1_per = evaluate(model, val_loader, device)
        scores.append(f1_macro)
        per_class_all.append(f1_per)
        seed_bar.set_postfix(F1=f"{f1_macro:.4f}")
        tqdm.write(f"  [{name} seed {seed}] F1-macro: {f1_macro:.4f}")

        if args.save_all_seeds:
            torch.save({
                "model_cls":   model_cls.__name__,
                "config_name": name,
                "seed":        seed,
                "f1_macro":    f1_macro,
                "epochs":      args.epochs,
                "split":       args.split,
                "state_dict":  {k: v.cpu() for k, v in model.state_dict().items()},
            }, ckpt_dir / f"{ckpt_stem(name, args)}_seed{seed}.pt")

        if f1_macro > best_f1:
            best_f1    = f1_macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_seed  = seed

    # save best seed's weights
    ckpt_path = ckpt_dir / f"{ckpt_stem(name, args)}_best.pt"
    torch.save({
        "model_cls":   model_cls.__name__,
        "config_name": name,
        "seed":        best_seed,
        "f1_macro":    best_f1,
        "epochs":      args.epochs,
        "split":       args.split,
        "state_dict":  best_state,
    }, ckpt_path)
    tqdm.write(f"  [{name}] saved best (seed {best_seed}, F1={best_f1:.4f}) -> {ckpt_path.name}")

    return scores, np.mean(per_class_all, axis=0)


def ckpt_stem(name: str, args) -> str:
    """Checkpoint file stem; recording-split runs get a _rec suffix so they
    never overwrite checkpoints trained on the old leaky per-clip split."""
    stem = name.replace("+", "_").replace(" ", "_") + args.tag
    return stem + "_rec" if args.split == "recording" else stem


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seeds",      type=int,   default=5)
    parser.add_argument("--data-dir",   type=Path,  default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--only",       type=str,   default=None,
                        help="comma-separated substrings to filter configs by name")
    parser.add_argument("--tag", type=str, default="",
                        help="suffix for checkpoint names (e.g. _ep100) so variants don't overwrite")
    parser.add_argument("--save-all-seeds", action="store_true",
                        help="also save every seed as <name>_seed<n>.pt")
    parser.add_argument("--eval-test", action="store_true",
                        help="after training, evaluate every saved seed on the test set "
                             "(implies --save-all-seeds) and report mean ± std")
    parser.add_argument("--split",      choices=["recording", "clip"], default="recording",
                        help="recording: keep all excerpts of a recording on one side "
                             "(no leakage); clip: old per-clip split")
    args = parser.parse_args()

    device = get_device()
    torch.backends.cudnn.benchmark = True   # fixed input shapes → let cuDNN pick fastest conv kernels
    print(f"Device: {device}")

    train_root = find_training_root(args.data_dir)
    full_ds    = load_training_dataset(train_root)
    train_paths, train_labels, val_paths, val_labels = train_val_split(
        full_ds.wav_paths, full_ds.labels, seed=42,
        by_recording=(args.split == "recording"),
    )
    print(f"Split: {args.split}  Train: {len(train_paths)}  Val: {len(val_paths)}")

    configs = [
        dict(name="BasicCNN",
             model_cls=BasicCNN),
        dict(name="HarmonicCNN",
             model_cls=HarmonicCNN),
        dict(name="HarmonicCNN+CosineLR",
             model_cls=HarmonicCNN,
             cosine_lr=True),
        dict(name="DeltaHarmonicCNN+CosineLR",
             model_cls=DeltaHarmonicCNN,
             cosine_lr=True),
        dict(name="DeltaHarmonicCNN+CosineLR+Aug",
             model_cls=DeltaHarmonicCNN,
             cosine_lr=True,
             pitch_shift_bins=2,
             spec_augment=True),
        dict(name="DeltaHarmonicCNN+CosineLR+PitchOnly",
             model_cls=DeltaHarmonicCNN,
             cosine_lr=True,
             pitch_shift_bins=2),
        dict(name="DeepHarmonicCNN+CosineLR",
             model_cls=DeepHarmonicCNN,
             cosine_lr=True),
        dict(name="DeepHarmonicCNN+CosineLR+PitchOnly",
             model_cls=DeepHarmonicCNN,
             cosine_lr=True,
             pitch_shift_bins=2),
        dict(name="DeepHarmonicCNN+CosineLR+60ep",
             model_cls=DeepHarmonicCNN,
             cosine_lr=True),
        dict(name="SEDeepHarmonicCNN+CosineLR",
             model_cls=SEDeepHarmonicCNN,
             cosine_lr=True),
        # Frequency-exact harmonic stacking (ExactHarmonicStack)
        dict(name="DeepMelCNN+CosineLR",
             model_cls=DeepMelCNN,
             cosine_lr=True),
        dict(name="DeepExactOctaveCNN+CosineLR",
             model_cls=DeepExactOctaveCNN,
             cosine_lr=True),
        dict(name="DeepExactHarmonicCNN+CosineLR",
             model_cls=DeepExactHarmonicCNN,
             cosine_lr=True),
        dict(name="DeepExactOctaveCNN+CosineLR+PolyMix",
             model_cls=DeepExactOctaveCNN,
             cosine_lr=True,
             use_polymix=True),
        dict(name="DeepMelCNN+CosineLR+PolyMix",
             model_cls=DeepMelCNN,
             cosine_lr=True,
             use_polymix=True),
        # Music domain knowledge on top of DeepMelCNN+PolyMix
        dict(name="DeepMelCNN+CosineLR+GenreMix",
             model_cls=DeepMelCNN,
             cosine_lr=True,
             use_polymix=True,
             genre_mix=True),
        dict(name="DeepMelFreqCNN+CosineLR+PolyMix",
             model_cls=DeepMelFreqCNN,
             cosine_lr=True,
             use_polymix=True),
        dict(name="DeepMelPonsCNN+CosineLR+PolyMix",
             model_cls=DeepMelPonsCNN,
             cosine_lr=True,
             use_polymix=True),
        # Own ideas on top of DeepMelPonsCNN+PolyMix
        dict(name="DeepMelLogPonsCNN+CosineLR+PolyMix",
             model_cls=DeepMelLogPonsCNN,
             cosine_lr=True,
             use_polymix=True),
        dict(name="DeepMelHPSPonsCNN+CosineLR+PolyMix",
             model_cls=DeepMelHPSPonsCNN,
             cosine_lr=True,
             use_polymix=True),
        dict(name="DeepMelMultiLogPonsCNN+CosineLR+PolyMix",
             model_cls=DeepMelMultiLogPonsCNN,
             cosine_lr=True,
             use_polymix=True),
        dict(name="DeepMelLogPonsCNN+CosineLR+PolyMix3",
             model_cls=DeepMelLogPonsCNN,
             cosine_lr=True,
             use_polymix=True,
             polymix_probs=(0.8, 0.4)),
        dict(name="DeepMelLogPonsMaxCNN+CosineLR+PolyMix",
             model_cls=DeepMelLogPonsMaxCNN,
             cosine_lr=True,
             use_polymix=True),
        # Pons et al. 2017 multi-layer model (re-implementation) and our variants
        dict(name="PonsMultiLayerCNN+CosineLR",
             model_cls=PonsMultiLayerCNN,
             cosine_lr=True),
        dict(name="PonsMultiLayerCNN+CosineLR+PolyMix",
             model_cls=PonsMultiLayerCNN,
             cosine_lr=True,
             use_polymix=True),
        dict(name="LogPonsMultiLayerCNN+CosineLR+PolyMix",
             model_cls=LogPonsMultiLayerCNN,
             cosine_lr=True,
             use_polymix=True),
        dict(name="DeepExactOctaveMaxCNN+CosineLR",
             model_cls=DeepExactOctaveMaxCNN,
             cosine_lr=True),
        dict(name="DeepMelMaxCNN+CosineLR",
             model_cls=DeepMelMaxCNN,
             cosine_lr=True),
        # Han architecture trained on 3-second windows (ablation)
        dict(name="Han2017CNN_3sec",
             model_cls=Han2017CNN),
        # Han architecture + harmonic stacking + 3-second windows
        dict(name="HarmonicHan2017CNN_3sec",
             model_cls=HarmonicHan2017CNN),
    ]

    if args.eval_test:
        args.save_all_seeds = True
    global PROGRESS
    PROGRESS = None
    all_results = {}
    if args.only:
        filters = [f.strip() for f in args.only.split(",")]
        names   = {c["name"] for c in configs}
        # an exact config name matches only itself; anything else is a substring
        configs = [c for c in configs
                   if any(f == c["name"] if f in names else f in c["name"] for f in filters)]
        print(f"Running {len(configs)} config(s): {[c['name'] for c in configs]}")

    total_units = len(configs) * args.seeds * (args.epochs + (EVAL_UNITS if args.eval_test else 0))
    PROGRESS = Progress(f"{len(configs)} cfg × {args.seeds} seeds", total_units)

    config_bar = tqdm(configs, desc="Configs", unit="cfg", position=0)
    for cfg in config_bar:
        name = cfg.pop("name")
        model_cls = cfg.pop("model_cls")
        config_bar.set_description(f"Config: {name[:30]}")
        scores, per_class = train_seeds(
            name, model_cls, train_paths, train_labels, val_paths, val_labels,
            args, device, **cfg,
        )
        all_results[name] = {"scores": scores, "per_class": per_class}

    print("\n─── Final comparison ───")
    rows = []
    for name, r in all_results.items():
        scores = r["scores"]
        mean = np.mean(scores)
        std  = np.std(scores)
        strs = [f"{s:.4f}" for s in scores]
        rows.append((mean, name, std, strs))
        print(f"{name:<35}  {mean:.4f} ± {std:.4f}  {strs}")

    best_name = max(rows, key=lambda r: r[0])[1]
    print(f"\n  Best: {best_name}")

    names = list(all_results.keys())
    print(f"\nPer-class F1 (mean over {args.seeds} seeds):")
    header = f"  {'Class':<6}" + "".join(f"  {n[:13]:>13}" for n in names)
    print(header)
    print(f"  {'-' * (8 + 15 * len(names))}")
    for i, cls in enumerate(CLASSES):
        vals = [all_results[n]["per_class"][i] for n in names]
        best = max(vals)
        row  = f"  {cls:<6}"
        for v in vals:
            marker = "*" if v == best and best > min(vals) + 0.005 else " "
            row += f"  {v:>12.3f}{marker}"
        print(row)

    # ── Threshold tuning on val set ───────────────────────────────────────────
    print("\n─── Threshold tuning on val set ───")
    MODEL_CLS_MAP = {m.__name__: m for m in [BasicCNN, HarmonicCNN, DeltaHarmonicCNN,
                                              DeepHarmonicCNN, SEDeepHarmonicCNN,
                                              Han2017CNN, HarmonicHan2017CNN,
                                              DeepExactHarmonicCNN, DeepExactOctaveCNN,
                                              DeepMelCNN, DeepExactOctaveMaxCNN, DeepMelMaxCNN,
                                              DeepMelFreqCNN, DeepMelPonsCNN,
                                              DeepMelLogPonsCNN, DeepMelHPSPonsCNN,
                                              DeepMelMultiLogPonsCNN, DeepMelLogPonsMaxCNN,
                                              PonsMultiLayerCNN, LogPonsMultiLayerCNN]}
    val_ds_full   = IRMASDataset(val_paths, val_labels)
    val_loader_thr = DataLoader(val_ds_full, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    for name in all_results:
        ckpt_path  = ckpt_dir / f"{ckpt_stem(name, args)}_best.pt"
        if not ckpt_path.exists():
            continue
        ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_cls = MODEL_CLS_MAP.get(ckpt.get("model_cls", ""))
        if model_cls is None:
            continue
        model = model_cls().to(device)
        model.load_state_dict(ckpt["state_dict"])
        probs, tgts = collect_val_probs(model, val_loader_thr, device)
        best_thr, best_f1, rows = tune_threshold(probs, tgts)
        print(f"\n  {name}:")
        for thr, f1 in rows:
            mark = " ←" if thr == best_thr else ""
            print(f"    thr={thr:.2f}  val F1={f1:.4f}{mark}")

    if args.eval_test:
        eval_seeds_on_test(all_results, args, device)
    PROGRESS.finish()


def eval_seeds_on_test(all_results, args, device):
    """Test macro/micro F1 of every saved seed (sliding window + S2) → mean ± std."""
    from dataset import _scan_testing_dir
    from ensemble_test import SCORE_DIR, clip_scores

    test_paths, test_labels = _scan_testing_dir(PROJECT_ROOT / "data" / "IRMAS" / "testing")
    y = np.vstack(test_labels).astype(int)
    thresholds = (0.3, 0.4, 0.5)

    print(f"\n─── Test set, every seed ({args.seeds} seeds, S2 sliding window) ───")
    for name in all_results:
        macro = {t: [] for t in thresholds}
        micro = {t: [] for t in thresholds}
        for seed in range(args.seeds):
            stem = f"{ckpt_stem(name, args)}_seed{seed}"
            PROGRESS.advance(0, f"test {name} s{seed}")
            (SCORE_DIR / f"{stem}.npy").unlink(missing_ok=True)   # weights were just retrained
            scores = clip_scores(stem, test_paths, device)
            for t in thresholds:
                pred = (scores >= t).astype(int)
                macro[t].append(f1_score(y, pred, average="macro", zero_division=0))
                micro[t].append(f1_score(y, pred, average="micro", zero_division=0))
            PROGRESS.advance(EVAL_UNITS)
        print(f"\n  {name}")
        for t in thresholds:
            print(f"    thr {t:.1f}:  macro {np.mean(macro[t]):.4f} ± {np.std(macro[t]):.4f}"
                  f"  [{', '.join(f'{v:.3f}' for v in macro[t])}]"
                  f"   micro {np.mean(micro[t]):.4f} ± {np.std(micro[t]):.4f}")


if __name__ == "__main__":
    main()
