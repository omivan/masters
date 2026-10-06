"""
Save matplotlib plots for every pipeline stage.

Output directory: plots/
    00_dataset_distribution.png
    01_waveform.png
    02_mel_spectrogram.png
    03_training_loss.png
    04_inference_probabilities.png
    05_per_class_f1.png

Usage:
    python src/plot.py
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    CLASSES, CLASS_TO_IDX, NUM_CLASSES,
    load_training_dataset, train_val_split, IRMASDataset,
)
from features import compute_mel_spectrogram, SAMPLE_RATE
from model import InstrumentCNNBiLSTM

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_ROOT   = PROJECT_ROOT / "data" / "IRMAS" / "training"
CHECKPOINT   = PROJECT_ROOT / "model.pt"
PLOTS_DIR    = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.09

STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4d",
    "axes.labelcolor":  "#c0c4d8",
    "xtick.color":      "#7a7e94",
    "ytick.color":      "#7a7e94",
    "text.color":       "#c0c4d8",
    "grid.color":       "#2a2d3d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
}
plt.rcParams.update(STYLE)

ACCENT   = "#4fc3f7"   # cyan
ACCENT2  = "#81c784"   # green
ACCENT3  = "#ff8a65"   # orange
WARN     = "#ef5350"   # red
MUTED    = "#546e7a"   # grey-blue

CLASS_LABELS = {
    "cel": "Cello", "cla": "Clarinet", "flu": "Flute",
    "gac": "Ac. Guitar", "gel": "El. Guitar", "org": "Organ",
    "pia": "Piano", "sax": "Saxophone", "tru": "Trumpet",
    "vio": "Violin", "voi": "Voice",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def find_root(base: Path) -> Path:
    if any((base / c).is_dir() for c in CLASSES):
        return base
    for child in sorted(base.iterdir()):
        if child.is_dir() and any((child / c).is_dir() for c in CLASSES):
            return child
    raise RuntimeError(f"No instrument folders under {base}")

def save(fig: plt.Figure, name: str):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path.relative_to(PROJECT_ROOT)}")

# ── 00 dataset distribution ───────────────────────────────────────────────────

def plot_dataset(train_root: Path):
    counts = {cls: len(list((train_root / cls).glob("*.wav"))) for cls in CLASSES}

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.suptitle("Stage 0 — Dataset Distribution", fontsize=14, fontweight="bold", color="white", y=1.01)

    labels = [CLASS_LABELS[c] for c in CLASSES]
    values = [counts[c] for c in CLASSES]
    colors = plt.cm.cool(np.linspace(0.2, 0.8, len(CLASSES)))

    bars = ax.barh(labels, values, color=colors, edgecolor="#0f1117", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                str(v), va="center", ha="left", fontsize=9, color="#c0c4d8")

    ax.set_xlabel("Number of .wav files")
    ax.set_xlim(0, max(values) * 1.12)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_title(f"Total: {sum(values)} files · {NUM_CLASSES} classes · ~{sum(values)//NUM_CLASSES} per class",
                 fontsize=9, color=MUTED)
    fig.tight_layout()
    save(fig, "00_dataset_distribution.png")

# ── 01 waveform ───────────────────────────────────────────────────────────────

def plot_waveform(wav_path: Path):
    y, sr = sf.read(wav_path, always_2d=True)
    y = y.mean(axis=1)
    t = np.linspace(0, len(y)/sr, len(y))

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Stage 1 — Raw Audio Waveform\n{wav_path.name}  ·  label: {wav_path.parent.name}  ·  sr={sr} Hz",
                 fontsize=12, fontweight="bold", color="white")

    # Waveform
    ax = axes[0]
    ax.plot(t, y, color=ACCENT, linewidth=0.4, alpha=0.85)
    ax.fill_between(t, y, alpha=0.15, color=ACCENT)
    ax.axhline(0, color=MUTED, linewidth=0.5)
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1.05, 1.05)
    ax.grid()

    # RMS energy envelope
    frame = sr // 100
    rms = np.array([np.sqrt(np.mean(y[i:i+frame]**2))
                    for i in range(0, len(y)-frame, frame)])
    t_rms = np.linspace(0, len(y)/sr, len(rms))
    ax2 = axes[1]
    ax2.fill_between(t_rms, rms, color=ACCENT3, alpha=0.7)
    ax2.set_ylabel("RMS energy")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim(0, t[-1])
    ax2.grid()

    fig.tight_layout()
    save(fig, "01_waveform.png")

# ── 02 mel spectrogram ────────────────────────────────────────────────────────

def plot_mel(wav_path: Path):
    mel = compute_mel_spectrogram(wav_path)   # [128, 130]

    # Custom dark "magma-like" colormap
    cmap = LinearSegmentedColormap.from_list(
        "dark_heat",
        ["#0f1117", "#1a237e", "#283593", "#1565c0",
         "#0277bd", "#00838f", "#2e7d32", "#f9a825",
         "#e65100", "#b71c1c"],
    )

    hop = 512
    n_frames = mel.shape[1]
    t_max = n_frames * hop / SAMPLE_RATE

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})
    fig.suptitle(f"Stage 2 — Log-Mel Spectrogram  [{mel.shape[0]} mel-bins × {mel.shape[1]} frames]\n"
                 f"{wav_path.name}  ·  label: {wav_path.parent.name}",
                 fontsize=12, fontweight="bold", color="white")

    # Spectrogram
    ax = axes[0]
    img = ax.imshow(mel, aspect="auto", origin="lower", cmap=cmap,
                    extent=[0, t_max, 0, SAMPLE_RATE/2/1000])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    plt.colorbar(img, ax=ax, label="Log energy (dB-like)")

    # Mean spectrum (frequency profile)
    ax2 = axes[1]
    mean_spec = mel.mean(axis=1)
    freqs = np.linspace(0, SAMPLE_RATE/2/1000, mel.shape[0])
    ax2.plot(mean_spec, freqs, color=ACCENT3, linewidth=1.2)
    ax2.fill_betweenx(freqs, mean_spec, alpha=0.25, color=ACCENT3)
    ax2.set_xlabel("Mean log energy")
    ax2.set_ylabel("")
    ax2.set_yticks([])
    ax2.set_ylim(0, SAMPLE_RATE/2/1000)
    ax2.set_title("Freq profile", fontsize=9, color=MUTED)
    ax2.grid()

    fig.tight_layout()
    save(fig, "02_mel_spectrogram.png")

# ── 02b spectrogram grid (one per class) ─────────────────────────────────────

def plot_mel_grid(train_root: Path):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle("Stage 2b — Mel Spectrograms by Instrument Class",
                 fontsize=13, fontweight="bold", color="white")

    cmap = LinearSegmentedColormap.from_list(
        "dark_heat",
        ["#0f1117", "#1a237e", "#0277bd", "#00838f",
         "#2e7d32", "#f9a825", "#e65100", "#b71c1c"],
    )

    all_axes = axes.flatten()
    for idx, cls in enumerate(CLASSES):
        wav = sorted((train_root / cls).glob("*.wav"))[0]
        mel = compute_mel_spectrogram(wav)
        ax = all_axes[idx]
        ax.imshow(mel, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(f"{CLASS_LABELS[cls]}  ({cls})", fontsize=9, color="white")
        ax.set_xticks([])
        ax.set_yticks([])

    # hide last empty subplot
    all_axes[-1].set_visible(False)
    fig.tight_layout()
    save(fig, "02b_mel_spectrogram_grid.png")

# ── 03 training loss ──────────────────────────────────────────────────────────

def plot_loss():
    train_losses = [
        0.4738, 0.3153, 0.3076, 0.3045, 0.3028,
        0.2966, 0.2947, 0.2943, 0.2917, 0.2901,
        0.2884, 0.2841, 0.2824, 0.2821, 0.2770,
        0.2754, 0.2710, 0.2729, 0.2673, 0.2671,
    ]
    val_losses = [
        0.3299, 0.3036, 0.3021, 0.2993, 0.2969,
        0.2956, 0.2942, 0.2931, 0.2907, 0.2881,
        0.2868, 0.2859, 0.2848, 0.2837, 0.2793,
        0.2804, 0.2789, 0.2746, 0.2731, 0.2743,
    ]
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Stage 3 — Training & Validation Loss (BCEWithLogitsLoss)",
                 fontsize=13, fontweight="bold", color="white")

    ax.plot(epochs, train_losses, color=ACCENT,  linewidth=2, marker="o",
            markersize=4, label="Train loss")
    ax.plot(epochs, val_losses,   color=ACCENT2, linewidth=2, marker="s",
            markersize=4, label="Val loss", linestyle="--")
    ax.fill_between(epochs, train_losses, val_losses, alpha=0.08, color=ACCENT3)

    # annotations
    ax.annotate(f"start\n{train_losses[0]:.4f}", xy=(1, train_losses[0]),
                xytext=(2, train_losses[0]+0.02),
                color=ACCENT, fontsize=8, arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax.annotate(f"end\n{train_losses[-1]:.4f}", xy=(len(epochs), train_losses[-1]),
                xytext=(len(epochs)-4, train_losses[-1]+0.02),
                color=ACCENT, fontsize=8, arrowprops=dict(arrowstyle="->", color=ACCENT))

    ax.axhline(np.log(2), color=WARN, linewidth=0.8, linestyle=":",
               label=f"Random baseline ≈ {np.log(2):.3f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid()
    ax.set_xlim(1, len(epochs))

    fig.tight_layout()
    save(fig, "03_training_loss.png")

# ── 04 inference probabilities ────────────────────────────────────────────────

def plot_inference(model, val_ds, val_paths, device):
    model.eval()

    # pick one sample per class
    samples = []
    used_classes = set()
    for i, path in enumerate(val_paths):
        cls = path.parent.name
        if cls not in used_classes and cls in CLASS_TO_IDX:
            samples.append(i)
            used_classes.add(cls)
        if len(samples) == 6:
            break

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"Stage 4 — Model Inference  (threshold={THRESHOLD})",
                 fontsize=13, fontweight="bold", color="white")

    with torch.no_grad():
        for ax, idx in zip(axes.flatten(), samples):
            x, y_true = val_ds[idx]
            logit = model(x.unsqueeze(0).to(device))
            probs = torch.sigmoid(logit).cpu().numpy()[0]

            true_cls  = [CLASSES[i] for i, v in enumerate(y_true) if v > 0]
            pred_cls  = [CLASSES[i] for i, p in enumerate(probs) if p >= THRESHOLD]
            labels    = [CLASS_LABELS[c] for c in CLASSES]

            colors = []
            for i, (cls, p) in enumerate(zip(CLASSES, probs)):
                is_true = y_true[i] > 0
                is_pred = p >= THRESHOLD
                if is_true and is_pred:
                    colors.append(ACCENT2)   # TP — green
                elif is_pred and not is_true:
                    colors.append(WARN)      # FP — red
                elif is_true and not is_pred:
                    colors.append(ACCENT3)   # FN — orange
                else:
                    colors.append(MUTED)     # TN — grey

            bars = ax.barh(labels, probs, color=colors, edgecolor="#0f1117", linewidth=0.5)
            ax.axvline(THRESHOLD, color="white", linewidth=0.8, linestyle="--", alpha=0.6,
                       label=f"threshold={THRESHOLD}")
            ax.set_xlim(0, 0.35)
            ax.invert_yaxis()
            ax.grid(axis="x")

            wav_name = val_paths[idx].name[:35]
            hit = "✓" if set(true_cls) & set(pred_cls) else "✗"
            ax.set_title(f"{hit}  true: {true_cls}  pred: {pred_cls}\n{wav_name}",
                         fontsize=7.5, color="white")

            # add value labels
            for bar, p in zip(bars, probs):
                if p > 0.02:
                    ax.text(p + 0.003, bar.get_y() + bar.get_height()/2,
                            f"{p:.3f}", va="center", fontsize=6.5, color="#c0c4d8")

    # legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(color=ACCENT2, label="True Positive"),
        Patch(color=WARN,    label="False Positive"),
        Patch(color=ACCENT3, label="False Negative (missed)"),
        Patch(color=MUTED,   label="True Negative"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               framealpha=0.2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    save(fig, "04_inference_probabilities.png")

# ── 05 per-class F1 ───────────────────────────────────────────────────────────

def plot_f1(model, val_ds, device):
    from sklearn.metrics import f1_score, confusion_matrix

    loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for x_b, y_b in loader:
            probs = torch.sigmoid(model(x_b.to(device))).cpu().numpy()
            all_preds.append((probs >= THRESHOLD).astype(int))
            all_targets.append(y_b.numpy().astype(int))

    preds   = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    f1_per   = f1_score(targets, preds, average=None,    zero_division=0)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    f1_micro = f1_score(targets, preds, average="micro", zero_division=0)

    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(f"Stage 5 — Evaluation Metrics  (threshold={THRESHOLD}  ·  val set)",
                 fontsize=13, fontweight="bold", color="white")

    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.3], wspace=0.35)

    # Left: per-class F1 bar chart
    ax1 = fig.add_subplot(gs[0])
    labels = [CLASS_LABELS[c] for c in CLASSES]
    bar_colors = [ACCENT2 if f > 0.3 else (ACCENT if f > 0.1 else WARN) for f in f1_per]
    bars = ax1.barh(labels, f1_per, color=bar_colors, edgecolor="#0f1117", linewidth=0.7)
    ax1.axvline(f1_macro, color=ACCENT3, linewidth=1.5, linestyle="--",
                label=f"F1-macro = {f1_macro:.4f}")
    ax1.axvline(f1_micro, color=ACCENT2, linewidth=1.5, linestyle=":",
                label=f"F1-micro = {f1_micro:.4f}")
    ax1.set_xlim(0, 1.0)
    ax1.invert_yaxis()
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(axis="x")
    ax1.set_xlabel("F1 score")
    ax1.set_title("Per-class F1", color="white")
    for bar, f in zip(bars, f1_per):
        ax1.text(f + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{f:.3f}", va="center", fontsize=8.5, color="#c0c4d8")

    # Right: confusion matrix (multi-label → per-class TP/FP/FN/TN summary)
    ax2 = fig.add_subplot(gs[1])
    tp = (preds * targets).sum(axis=0)
    fp = (preds * (1 - targets)).sum(axis=0)
    fn = ((1 - preds) * targets).sum(axis=0)

    x = np.arange(NUM_CLASSES)
    w = 0.28
    ax2.bar(x - w, tp, w, label="TP", color=ACCENT2,  edgecolor="#0f1117")
    ax2.bar(x,     fp, w, label="FP", color=WARN,     edgecolor="#0f1117")
    ax2.bar(x + w, fn, w, label="FN", color=ACCENT3,  edgecolor="#0f1117")
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y")
    ax2.set_ylabel("Count")
    ax2.set_title("TP / FP / FN per class", color="white")

    fig.tight_layout()
    save(fig, "05_per_class_f1.png")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    train_root = find_root(TRAIN_ROOT)
    full_ds = load_training_dataset(train_root)
    _, _, val_paths, val_labels = train_val_split(full_ds.wav_paths, full_ds.labels)
    val_ds = IRMASDataset(val_paths, val_labels)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = InstrumentCNNBiLSTM().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')}), device={device}\n")

    # pick a visually interesting sample (electric guitar)
    sample_wav = sorted((train_root / "gel").glob("*.wav"))[0]

    print("Generating plots …")
    plot_dataset(train_root)
    plot_waveform(sample_wav)
    plot_mel(sample_wav)
    plot_mel_grid(train_root)
    plot_loss()
    plot_inference(model, val_ds, val_paths, device)
    plot_f1(model, val_ds, device)

    print(f"\nAll plots saved to {PLOTS_DIR}/")

if __name__ == "__main__":
    main()
