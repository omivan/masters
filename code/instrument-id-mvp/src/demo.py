"""
End-to-end ASCII demo: every stage printed to the terminal.

    python src/demo.py
    python src/demo.py --wav path/to/file.wav
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    CLASSES, CLASS_TO_IDX, NUM_CLASSES,
    load_training_dataset, train_val_split, IRMASDataset,
)
from features import (
    SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, TIME_FRAMES,
    compute_mel_spectrogram,
)
from model import InstrumentCNNBiLSTM

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_ROOT   = PROJECT_ROOT / "data" / "IRMAS" / "training"
CHECKPOINT   = PROJECT_ROOT / "model.pt"

# ── colour helpers (ANSI) ────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"
MAGENTA= "\033[35m"
RED    = "\033[31m"
DIM    = "\033[2m"

def header(title: str, color: str = CYAN):
    w = 68
    bar = "─" * w
    print(f"\n{color}{BOLD}┌{bar}┐")
    pad = (w - len(title)) // 2
    print(f"│{' ' * pad}{title}{' ' * (w - pad - len(title))}│")
    print(f"└{bar}┘{RESET}")

def section(title: str):
    print(f"\n{YELLOW}{BOLD}▸ {title}{RESET}")

# ── ASCII waveform ────────────────────────────────────────────────────────────
def ascii_waveform(y: np.ndarray, width: int = 70, height: int = 10) -> str:
    chars = " ▁▂▃▄▅▆▇█"
    # Downsample
    n = len(y)
    step = max(1, n // width)
    chunks = [y[i:i+step] for i in range(0, n - step, step)][:width]
    rms = np.array([np.sqrt(np.mean(c**2)) for c in chunks])
    rms = rms / (rms.max() + 1e-9)

    rows = []
    for row in range(height, 0, -1):
        threshold = row / height
        line = ""
        for v in rms:
            if v >= threshold:
                line += f"{GREEN}█{RESET}"
            elif v >= threshold - 1/height:
                line += f"{GREEN}▄{RESET}"
            else:
                line += f"{DIM}·{RESET}"
        rows.append(f"  {DIM}│{RESET}{line}{DIM}│{RESET}")
    rows.append(f"  {DIM}└{'─'*width}┘{RESET}")
    return "\n".join(rows)

# ── ASCII spectrogram ─────────────────────────────────────────────────────────
def ascii_spectrogram(mel: np.ndarray, width: int = 70, height: int = 20) -> str:
    # mel: [128, 130]  →  resize to [height, width]
    from numpy import interp
    h, w = mel.shape
    rows_idx = np.linspace(0, h-1, height).astype(int)
    cols_idx = np.linspace(0, w-1, width).astype(int)
    m = mel[np.ix_(rows_idx, cols_idx)]

    vmin, vmax = m.min(), m.max()
    norm = (m - vmin) / (vmax - vmin + 1e-9)

    palette = [
        (0.00, "\033[38;5;17m"),   # dark blue  (silence)
        (0.20, "\033[38;5;19m"),
        (0.35, "\033[38;5;27m"),
        (0.50, "\033[38;5;33m"),   # mid blue
        (0.60, "\033[38;5;51m"),   # cyan
        (0.70, "\033[38;5;46m"),   # green
        (0.80, "\033[38;5;226m"),  # yellow
        (0.90, "\033[38;5;208m"),  # orange
        (1.00, "\033[38;5;196m"),  # red (loudest)
    ]
    density = " ·:+=%#@██"

    def colorize(v):
        for i in range(len(palette)-1, -1, -1):
            if v >= palette[i][0]:
                return palette[i][1]
        return palette[0][1]

    lines = []
    for row in range(height-1, -1, -1):   # high freq at top
        line = ""
        for col in range(width):
            v = norm[row, col]
            ch = density[min(int(v * (len(density)-1)), len(density)-1)]
            line += colorize(v) + ch + RESET
        freq_label = f"{DIM}{int(row/height*11000):5d}Hz{RESET}" if row % 4 == 0 else "       "
        lines.append(f"{freq_label} {DIM}│{RESET}{line}{DIM}│{RESET}")

    lines.append(f"       {DIM}└{'─'*width}┘{RESET}")
    lines.append(f"       {DIM}{'0s':^{width//2}}{'~3s':>{width//2}}{RESET}")
    return "\n".join(lines)

# ── ASCII bar ─────────────────────────────────────────────────────────────────
def bar(value: float, width: int = 30, color: str = GREEN) -> str:
    filled = int(value * width)
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"

# ── ASCII loss curve ──────────────────────────────────────────────────────────
def ascii_loss_curve(losses: list[float], width: int = 60, height: int = 8) -> str:
    if not losses:
        return "(no data)"
    lo, hi = min(losses), max(losses)
    span = hi - lo or 1e-9
    norm = [(v - lo) / span for v in losses]

    # Build grid
    grid = [[" "] * width for _ in range(height)]
    step = max(1, len(norm) // width)
    points = norm[::step][:width]
    for x, v in enumerate(points):
        y = int((1 - v) * (height - 1))
        grid[y][x] = "●"
        # fill downward
        for yy in range(y+1, height):
            grid[yy][x] = "│" if grid[yy][x] == " " else grid[yy][x]

    rows = []
    for i, row in enumerate(grid):
        label_val = hi - (i / (height-1)) * span
        rows.append(f"  {DIM}{label_val:.3f} │{RESET}{GREEN}{''.join(row)}{RESET}")
    rows.append(f"  {DIM}{'':6} └{'─'*width}{RESET}")
    rows.append(f"  {DIM}{'':6}  ep 1{' '*(width-9)}ep {len(losses)}{RESET}")
    return "\n".join(rows)

# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, default=None, help="Specific wav to demo")
    args = parser.parse_args()

    # ── find training root ────────────────────────────────────────────────────
    def find_root(base):
        if any((base / c).is_dir() for c in CLASSES):
            return base
        for child in sorted(base.iterdir()):
            if child.is_dir() and any((child / c).is_dir() for c in CLASSES):
                return child
        raise RuntimeError(f"No instrument folders under {base}")

    # ────────────────────────────────────────────────────────────────────────
    header("STAGE 0 — DATASET", MAGENTA)

    train_root = find_root(TRAIN_ROOT)
    counts = {}
    all_wavs = {}
    for cls in CLASSES:
        wavs = sorted((train_root / cls).glob("*.wav"))
        counts[cls] = len(wavs)
        all_wavs[cls] = wavs

    section("Class distribution")
    total = sum(counts.values())
    print(f"  {'Class':<8} {'Count':>6}  {'Bar'}")
    print(f"  {'─'*8} {'─'*6}  {'─'*32}")
    for cls in CLASSES:
        n = counts[cls]
        b = bar(n / max(counts.values()), 32, CYAN)
        print(f"  {cls:<8} {n:>6}  {b} {n}")
    print(f"\n  {BOLD}Total: {total} files, {NUM_CLASSES} classes{RESET}")

    # ── pick a sample wav ───────────────────────────────────────────────────
    wav_path = args.wav
    if wav_path is None:
        # pick one from each class for later multi-class demo, use first for vis
        sample_cls = CLASSES[4]  # gel — electric guitar, visually interesting
        wav_path = all_wavs[sample_cls][0]

    true_label = wav_path.parent.name if wav_path.parent.name in CLASS_TO_IDX else "unknown"

    # ────────────────────────────────────────────────────────────────────────
    header("STAGE 1 — RAW AUDIO", BLUE)
    section(f"File: {wav_path.name}  |  true label: {BOLD}{true_label}{RESET}")

    y_raw, sr = sf.read(wav_path, always_2d=True)
    y_mono = y_raw.mean(axis=1)
    duration = len(y_mono) / sr

    print(f"  Sample rate : {sr} Hz")
    print(f"  Duration    : {duration:.2f} s")
    print(f"  Samples     : {len(y_mono):,}")
    print(f"  Amplitude   : [{y_mono.min():.3f}, {y_mono.max():.3f}]")
    print()
    print(ascii_waveform(y_mono, width=68, height=8))

    # ────────────────────────────────────────────────────────────────────────
    header("STAGE 2 — MEL SPECTROGRAM", BLUE)
    section("Transformation pipeline")
    print(f"""
  Raw audio  [{len(y_mono):,} samples]
      │
      ▼  Short-Time Fourier Transform (n_fft={N_FFT}, hop={HOP_LENGTH})
      │
      ▼  Mel filterbank  (n_mels={N_MELS})
      │
      ▼  log(S + 1e-9)
      │
      ▼  Crop / pad → fixed [{N_MELS} × {TIME_FRAMES}]
      │
  Spectrogram  [128 × 130 floats]
""")

    section("Computed spectrogram (colour = energy, red=loud, blue=quiet)")
    mel = compute_mel_spectrogram(wav_path)
    print(f"  Shape: {mel.shape}   min={mel.min():.2f}  max={mel.max():.2f}  mean={mel.mean():.2f}")
    print()
    print(ascii_spectrogram(mel, width=68, height=18))

    # ────────────────────────────────────────────────────────────────────────
    header("STAGE 3 — DATASET → TENSORS", BLUE)

    full_ds = load_training_dataset(train_root)
    train_paths, train_labels, val_paths, val_labels = train_val_split(
        full_ds.wav_paths, full_ds.labels
    )
    train_ds = IRMASDataset(train_paths, train_labels)
    val_ds   = IRMASDataset(val_paths, val_labels)

    x_sample, y_sample = train_ds[0]
    section("Sample tensor shapes")
    print(f"  x (spectrogram) : {list(x_sample.shape)}  dtype={x_sample.dtype}")
    print(f"  y (label vector): {list(y_sample.shape)}  dtype={y_sample.dtype}")
    print(f"  y values        : {y_sample.numpy().tolist()}")
    print(f"  active labels   : {[CLASSES[i] for i,v in enumerate(y_sample) if v > 0]}")
    print()
    print(f"  Train split : {len(train_ds)} samples")
    print(f"  Val split   : {len(val_ds)} samples")

    # ────────────────────────────────────────────────────────────────────────
    header("STAGE 4 — MODEL ARCHITECTURE", BLUE)

    model = InstrumentCNNBiLSTM()
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    section("Layer breakdown")
    print(f"""
  Input          [B, 1, 128, 130]
  ┌──────────────────────────────────────────────────────────┐
  │  CNN Block                                               │
  │  Conv2d(1→16, 3×3) → BN → ReLU → MaxPool(2×2)          │
  │  Conv2d(16→32, 3×3) → BN → ReLU → MaxPool(2×2)         │
  │  Dropout2d(0.25)                                        │
  │  Output: [B, 32, 32, 32]                                │
  └───────────────────────┬──────────────────────────────────┘
                          │  reshape
                          ▼
                 [B, 32 time-steps, 1024 features]
  ┌──────────────────────────────────────────────────────────┐
  │  BiLSTM  (hidden=64, bidirectional)                     │
  │  Output: [B, 32, 128]   → mean over time → [B, 128]    │
  │  Dropout(0.30)                                          │
  └───────────────────────┬──────────────────────────────────┘
                          │
                          ▼
                 Linear(128 → {NUM_CLASSES})
                          │
                 [B, {NUM_CLASSES}]  raw logits
                          │  sigmoid
                 [B, {NUM_CLASSES}]  probabilities ∈ (0,1)
""")
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable params    : {trainable:,}")
    print(f"  Approximate size    : {total_params*4/1024:.1f} KB (float32)")

    # ────────────────────────────────────────────────────────────────────────
    header("STAGE 5 — TRAINING LOOP", BLUE)

    if not CHECKPOINT.exists():
        print(f"  {RED}model.pt not found — run train.py first{RESET}")
        return

    section("Loss function: BCEWithLogitsLoss")
    print("""
  For each of the 11 classes independently:
      loss = -[y·log(σ(logit)) + (1-y)·log(1-σ(logit))]

  Total loss = mean over all classes and batch items.
  Lower is better. Random baseline ≈ log(2) ≈ 0.693
""")

    # Simulate the recorded loss curve from our actual run
    recorded_losses = [
        0.4738, 0.3153, 0.3076, 0.3045, 0.3028,
        0.2966, 0.2947, 0.2943, 0.2917, 0.2901,
        0.2884, 0.2841, 0.2824, 0.2821, 0.2770,
        0.2754, 0.2710, 0.2729, 0.2673, 0.2671,
    ]

    section("Train loss over 20 epochs")
    print(ascii_loss_curve(recorded_losses, width=60, height=8))
    print(f"\n  Start : {recorded_losses[0]:.4f}")
    print(f"  End   : {recorded_losses[-1]:.4f}")
    print(f"  Drop  : {recorded_losses[0]-recorded_losses[-1]:.4f}  ({(1-recorded_losses[-1]/recorded_losses[0])*100:.1f}% improvement)")

    # ────────────────────────────────────────────────────────────────────────
    header("STAGE 6 — INFERENCE ON SAMPLES", BLUE)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = InstrumentCNNBiLSTM().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    section("Per-class probability bars  (threshold = 0.09)")
    print(f"  {'Class':<8}  {'Prob':>5}  {'Bar (30 chars)':30}  Predict  True")
    print(f"  {'─'*8}  {'─'*5}  {'─'*30}  {'─'*7}  {'─'*4}")

    threshold = 0.09
    # Demo on 5 different samples from different classes
    demo_indices = []
    for cls in CLASSES:
        for i, path in enumerate(val_paths):
            if path.parent.name == cls:
                demo_indices.append(i)
                break

    print()
    for sample_idx in demo_indices[:5]:
        x, y_true = val_ds[sample_idx]
        true_cls = [CLASSES[i] for i,v in enumerate(y_true) if v > 0]

        with torch.no_grad():
            logit = model(x.unsqueeze(0).to(device))
            probs = torch.sigmoid(logit).cpu().numpy()[0]

        pred_cls = [CLASSES[i] for i,p in enumerate(probs) if p >= threshold]
        wav_name = val_paths[sample_idx].name[:28]

        print(f"  {BOLD}{wav_name}{RESET}")
        print(f"  True: {GREEN}{true_cls}{RESET}   Predicted: {CYAN}{pred_cls}{RESET}")
        for i, (cls, p) in enumerate(zip(CLASSES, probs)):
            is_true = y_true[i] > 0
            is_pred = p >= threshold
            color = GREEN if is_true else (RED if is_pred else DIM)
            marker = "✓" if is_true and is_pred else ("✗" if is_pred and not is_true else ("·" if is_true else " "))
            b = bar(p, 30, color)
            flag = f"{GREEN}✓{RESET}" if is_true else " "
            pred_flag = f"{CYAN}▶{RESET}" if is_pred else " "
            print(f"    {color}{cls:<8}{RESET}  {p:>5.3f}  {b}  {pred_flag}{flag}")
        print()

    # ────────────────────────────────────────────────────────────────────────
    header("STAGE 7 — EVALUATION METRICS", BLUE)

    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader

    loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            logits = model(x_b.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs >= threshold).astype(int)
            all_preds.append(preds)
            all_targets.append(y_b.numpy().astype(int))

    preds   = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    f1_macro = f1_score(targets, preds, average="macro",  zero_division=0)
    f1_micro = f1_score(targets, preds, average="micro",  zero_division=0)
    f1_per   = f1_score(targets, preds, average=None,     zero_division=0)

    section(f"Overall  (threshold={threshold})")
    print(f"  F1-macro : {bar(f1_macro, 40, GREEN)} {GREEN}{BOLD}{f1_macro:.4f}{RESET}")
    print(f"  F1-micro : {bar(f1_micro, 40, CYAN)}  {CYAN}{BOLD}{f1_micro:.4f}{RESET}")

    section("Per-class F1")
    print(f"  {'Class':<8}  {'F1':>6}  {'Bar'}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*40}")
    for cls, f1 in zip(CLASSES, f1_per):
        color = GREEN if f1 > 0.3 else (YELLOW if f1 > 0.1 else RED)
        print(f"  {cls:<8}  {f1:>6.4f}  {bar(f1, 40, color)}")

    # ────────────────────────────────────────────────────────────────────────
    header("SUMMARY", MAGENTA)
    print(f"""
  Pipeline:  wav → mel spectrogram [128×130] → CNN+BiLSTM → 11 logits → sigmoid → threshold

  Dataset  : {total} files  ({NUM_CLASSES} classes × ~{total//NUM_CLASSES} files)
  Model    : {trainable:,} parameters
  Training : 20 epochs, ~25 sec on MPS (Apple Silicon)
  Loss     : {recorded_losses[0]:.4f} → {recorded_losses[-1]:.4f}

  {GREEN}F1-macro : {f1_macro:.4f}{RESET}
  {CYAN}F1-micro : {f1_micro:.4f}{RESET}

  {DIM}Next steps: full dataset (6000+ files), data augmentation,
  per-class thresholds, SVM/HMM baseline comparison.{RESET}
""")


if __name__ == "__main__":
    main()
