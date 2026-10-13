"""
Evaluate HarmonicCNN / DeltaHarmonicCNN checkpoints on the IRMAS polyphonic
test set using 3-second sliding windows (50% overlap) + S2 aggregation.

This is a fair apples-to-apples comparison with Han2017CNN which also uses
sliding-window inference — the only difference is window length (3 s vs 1 s).

Usage:
    python src/evaluate_test_sliding.py
    python src/evaluate_test_sliding.py --threshold 0.3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import CLASSES, _scan_testing_dir
from features import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH
from model import (BasicCNN, HarmonicCNN, DeltaHarmonicCNN, DeepHarmonicCNN, SEDeepHarmonicCNN,
                   Han2017CNN, HarmonicHan2017CNN,
                   DeepExactHarmonicCNN, DeepExactOctaveCNN, DeepMelCNN,
                   DeepExactOctaveMaxCNN, DeepMelMaxCNN,
                   DeepMelFreqCNN, DeepMelPonsCNN,
                   DeepMelLogPonsCNN, DeepMelHPSPonsCNN,
                   DeepMelMultiLogPonsCNN, DeepMelLogPonsMaxCNN,
                   PonsMultiLayerCNN, LogPonsMultiLayerCNN)

PROJECT_ROOT = Path(__file__).parent.parent
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
TEST_DIR     = PROJECT_ROOT / "data" / "IRMAS" / "testing"

WIN_FRAMES = 130   # 3-second window — same as training
HOP_FRAMES = 65    # 50% overlap

MODEL_MAP = {
    "BasicCNN":            BasicCNN,
    "HarmonicCNN":         HarmonicCNN,
    "DeltaHarmonicCNN":    DeltaHarmonicCNN,
    "DeepHarmonicCNN":     DeepHarmonicCNN,
    "SEDeepHarmonicCNN":   SEDeepHarmonicCNN,
    "Han2017CNN":          Han2017CNN,
    "HarmonicHan2017CNN":  HarmonicHan2017CNN,
    "DeepExactHarmonicCNN": DeepExactHarmonicCNN,
    "DeepExactOctaveCNN":  DeepExactOctaveCNN,
    "DeepMelCNN":          DeepMelCNN,
    "DeepExactOctaveMaxCNN": DeepExactOctaveMaxCNN,
    "DeepMelMaxCNN":       DeepMelMaxCNN,
    "DeepMelFreqCNN":      DeepMelFreqCNN,
    "DeepMelPonsCNN":      DeepMelPonsCNN,
    "DeepMelLogPonsCNN":   DeepMelLogPonsCNN,
    "DeepMelHPSPonsCNN":   DeepMelHPSPonsCNN,
    "DeepMelMultiLogPonsCNN": DeepMelMultiLogPonsCNN,
    "DeepMelLogPonsMaxCNN": DeepMelLogPonsMaxCNN,
    "PonsMultiLayerCNN":   PonsMultiLayerCNN,
    "LogPonsMultiLayerCNN": LogPonsMultiLayerCNN,
}

_mel_tf = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
)
_resample_cache: dict = {}


def full_mel(wav_path: Path) -> np.ndarray:
    data, sr = sf.read(wav_path, always_2d=True)
    waveform = torch.from_numpy(data.mean(axis=1)).float().unsqueeze(0)
    if sr != SAMPLE_RATE:
        if sr not in _resample_cache:
            _resample_cache[sr] = T.Resample(sr, SAMPLE_RATE)
        waveform = _resample_cache[sr](waveform)
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    mel = _mel_tf(waveform)
    return torch.log(mel + 1e-9).squeeze(0).numpy()   # [128, T_full]


def predict_sliding(model, wav_paths, labels, device, threshold):
    model.eval()
    preds, targets = [], []
    for wav_path, label in tqdm(zip(wav_paths, labels), total=len(wav_paths),
                                desc="sliding eval", leave=False):
        mel  = full_mel(wav_path)            # [128, T_full]
        T_f  = mel.shape[1]
        starts = list(range(0, max(1, T_f - WIN_FRAMES + 1), HOP_FRAMES))

        scores = np.zeros(len(CLASSES), dtype=np.float32)
        with torch.no_grad():
            for s in starts:
                w = mel[:, s:s + WIN_FRAMES]
                if w.shape[1] < WIN_FRAMES:
                    w = np.pad(w, ((0, 0), (0, WIN_FRAMES - w.shape[1])),
                               mode="constant", constant_values=np.log(1e-9))
                x = torch.from_numpy(w).unsqueeze(0).unsqueeze(0).float().to(device)
                scores += torch.sigmoid(model(x)).cpu().numpy().squeeze()

        mx = scores.max()
        if mx > 0:
            scores /= mx
        preds.append((scores >= threshold).astype(int))
        targets.append(label.astype(int))

    return np.vstack(preds), np.vstack(targets)


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--test-dir",  type=Path,  default=TEST_DIR)
    parser.add_argument("--only",      type=str,   default=None,
                        help="comma-separated substrings; evaluate only matching checkpoint files")
    args = parser.parse_args()

    device = get_device()
    print(f"Device    : {device}")
    print(f"Threshold : {args.threshold}")
    print(f"Window    : {WIN_FRAMES} frames (3 s), hop {HOP_FRAMES} (50% overlap)\n")

    test_paths, test_labels = _scan_testing_dir(args.test_dir)
    print(f"Test clips: {len(test_paths)}\n")

    ckpts = sorted(CKPT_DIR.glob("*_best.pt"))
    if args.only:
        filters = [f.strip() for f in args.only.split(",")]
        ckpts = [c for c in ckpts if any(f in c.name for f in filters)]

    results = {}
    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cls_name = ckpt.get("model_cls", "")
        if cls_name not in MODEL_MAP:
            print(f"Skipping {ckpt_path.name} ({cls_name} not supported here)")
            continue

        model = MODEL_MAP[cls_name]().to(device)
        model.load_state_dict(ckpt["state_dict"])

        name = ckpt["config_name"] + ("" if ckpt.get("split", "clip") == "clip" else f" [{ckpt['split']}]")
        preds, targets = predict_sliding(model, test_paths, test_labels, device, args.threshold)
        macro   = f1_score(targets, preds, average="macro", zero_division=0)
        micro   = f1_score(targets, preds, average="micro", zero_division=0)
        per_cls = f1_score(targets, preds, average=None,    zero_division=0)

        results[name] = dict(macro=macro, micro=micro, per_cls=per_cls,
                             val_f1=ckpt["f1_macro"])
        print(f"{name:<35}  macro={macro:.4f}  micro={micro:.4f}"
              f"  (val {ckpt['f1_macro']:.4f})")

    # Per-class table
    names = list(results.keys())
    print(f"\nPer-class F1 (sliding window, threshold={args.threshold}):")
    header = f"  {'Class':<6}" + "".join(f"  {n[:16]:>16}" for n in names)
    print(header)
    print("  " + "─" * (8 + 18 * len(names)))
    for i, cls in enumerate(CLASSES):
        vals = [results[n]["per_cls"][i] for n in names]
        best = max(vals)
        row  = f"  {cls:<6}"
        for v in vals:
            marker = "*" if v == best and best > min(vals) + 0.005 else " "
            row   += f"  {v:>15.3f}{marker}"
        print(row)

    print("\n─── Summary ───")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["macro"]):
        print(f"  {name:<35}  macro={r['macro']:.4f}  micro={r['micro']:.4f}")

    print("\nReference (1-second sliding window):")
    print("  Han2017CNN (ours)            macro=0.501  micro=0.593")
    print("  Han et al. 2017 (original)   macro=0.503  micro=0.602")


if __name__ == "__main__":
    main()
