"""
Side-by-side comparison of mel / Δmel / Δ²mel across 5 instruments.

Each column = one instrument. Each row = one feature:
  row 0 : mel spectrogram         (harmonic content)
  row 1 : Δmel  (1st derivative)  (attack / decay envelope)
  row 2 : Δ²mel (2nd derivative)  (acceleration of change)

The goal: show visually why DeltaHarmonicCNN can distinguish instruments
that look similar in raw mel but differ in their temporal dynamics.

Usage:
    python visualize_delta_comparison.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import compute_mel_spectrogram

# ── Instruments to compare ─────────────────────────────────────────────────────
DATA_ROOT = (
    Path(__file__).parent
    / "data/IRMAS/training/IRMAS-TrainingData"
)

INSTRUMENTS = [
    ("pia", "Piano",  "027__[pia][nod][cla]1398__2.wav",
     "різкий attack, швидкий decay"),
    ("org", "Organ",  "[org][pop_roc]1069__3.wav",
     "повільний attack, рівний sustained"),
    ("vio", "Violin", "[vio][jaz_blu]2092__2.wav",
     "vibrato → осцилюючий Δmel"),
    ("flu", "Flute",  "[flu][cla]0441__1.wav",
     "слабкі обертони, м'який attack"),
    ("tru", "Trumpet","[tru][cla]2014__2.wav",
     "яскраві обертони, різкі ноти"),
]

OUT_PATH = Path(__file__).parent / "plots" / "delta_comparison.png"

HOP_LENGTH  = 512
SAMPLE_RATE = 22050
TIME_FRAMES = 130
BG, FG      = "#0f1117", "#e0e0e0"

ROW_LABELS  = ["mel", "Δmel", "Δ²mel"]
ROW_CMAPS   = ["viridis", "RdBu_r", "PuOr_r"]
ROW_NOTES   = [
    "гармонічний зміст",
    "швидкість зміни (червоний=зростає, синій=спадає)",
    "прискорення зміни",
]


def delta(x: torch.Tensor) -> torch.Tensor:
    xp = F.pad(x.unsqueeze(0), (1, 1, 0, 0), mode="replicate").squeeze(0)
    return (xp[:, :, 2:] - xp[:, :, :-2]) * 0.5


def load_features(inst_code: str, filename: str):
    path = DATA_ROOT / inst_code / filename
    mel_np = compute_mel_spectrogram(path)
    mel_t  = torch.tensor(mel_np).unsqueeze(0)
    d1     = delta(mel_t)
    d2     = delta(d1)
    return [
        mel_np,
        d1.squeeze(0).numpy(),
        d2.squeeze(0).numpy(),
    ]


def main():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "axes.edgecolor":   FG, "axes.labelcolor": FG,
        "xtick.color": FG,      "ytick.color": FG,
        "text.color":  FG,      "font.size": 10,
    })

    n_inst = len(INSTRUMENTS)
    n_rows = 3

    times = np.arange(TIME_FRAMES) * HOP_LENGTH / SAMPLE_RATE

    fig = plt.figure(figsize=(4.2 * n_inst, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Mel spectrogram vs temporal derivatives — порівняння інструментів",
        color=FG, fontsize=13, y=1.01,
    )

    gs = gridspec.GridSpec(
        n_rows, n_inst,
        figure=fig,
        hspace=0.55,
        wspace=0.25,
    )

    # Pre-load all features
    all_feats = []
    for inst_code, label, filename, _ in INSTRUMENTS:
        print(f"  Loading {label} …")
        all_feats.append(load_features(inst_code, filename))

    # Compute symmetric clims per row (shared across instruments for fair comparison)
    clims = []
    for row in range(n_rows):
        data_all = np.concatenate([f[row].ravel() for f in all_feats])
        if row == 0:
            clims.append((data_all.min(), data_all.max()))
        else:
            vmax = np.percentile(np.abs(data_all), 98)
            clims.append((-vmax, vmax))

    # Draw panels
    for col, ((inst_code, label, _, note), feats) in enumerate(
        zip(INSTRUMENTS, all_feats)
    ):
        for row, (data, cmap, (vmin, vmax)) in enumerate(
            zip(feats, ROW_CMAPS, clims)
        ):
            ax = fig.add_subplot(gs[row, col])

            ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                extent=[times[0], times[-1], 0, data.shape[0]],
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                interpolation="nearest",
            )

            # Column header (instrument name + description) — only top row
            if row == 0:
                ax.set_title(
                    f"{label}\n"
                    + r"$\it{" + note.replace(" ", r"\ ") + r"}$",
                    color=FG, fontsize=10, pad=6,
                )

            # Row label — only leftmost column
            if col == 0:
                ax.set_ylabel(ROW_LABELS[row], color=FG, fontsize=11,
                              fontweight="bold")

            # X-axis label — only bottom row
            if row == n_rows - 1:
                ax.set_xlabel("Time (s)", color=FG, fontsize=9)
            else:
                ax.set_xticks([])

            ax.tick_params(labelsize=8)

    # Row annotations on the right side
    for row, note in enumerate(ROW_NOTES):
        ax_last = fig.add_subplot(gs[row, -1])
        ax_last.set_visible(False)
        fig.text(
            0.99, gs[row, -1].get_position(fig).y0
            + gs[row, -1].get_position(fig).height / 2,
            note,
            va="center", ha="right", color=FG, fontsize=8,
            alpha=0.7, rotation=0,
        )

    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
