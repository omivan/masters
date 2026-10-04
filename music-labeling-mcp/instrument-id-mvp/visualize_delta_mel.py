"""
Visualisation of mel spectrogram and its temporal derivatives (Δmel, Δ²mel).

Shows how DeltaHarmonicCNN perceives the same audio through 3 lenses:
  • mel    — harmonic content (what frequencies are present)
  • Δmel   — rate of change  (attack, sustain, decay envelope)
  • Δ²mel  — acceleration    (where the sound starts, peaks, fades)

Usage:
    python visualize_delta_mel.py                        # uses built-in piano sample
    python visualize_delta_mel.py path/to/any_audio.wav
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ── Allow importing from src/ ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import compute_mel_spectrogram

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_WAV = (
    Path(__file__).parent
    / "data/IRMAS/training/IRMAS-TrainingData/pia"
    / "027__[pia][nod][cla]1398__2.wav"
)
OUT_PATH = Path(__file__).parent / "plots" / "delta_mel_visualisation.png"

TIME_FRAMES = 130   # matches features.py
HOP_LENGTH  = 512
SAMPLE_RATE = 22050

BG     = "#0f1117"
FG     = "#e0e0e0"


# ── Delta computation (same as DeltaHarmonicCNN._delta) ───────────────────────

def delta(x: torch.Tensor) -> torch.Tensor:
    """Central finite difference along time axis. [1, F, T] → [1, F, T]"""
    xp = F.pad(x.unsqueeze(0), (1, 1, 0, 0), mode="replicate").squeeze(0)  # [1, F, T+2]
    return (xp[:, :, 2:] - xp[:, :, :-2]) * 0.5


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    wav_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WAV
    if not wav_path.exists():
        print(f"File not found: {wav_path}")
        sys.exit(1)

    print(f"Loading: {wav_path.name}")

    # Compute mel and derivatives
    mel_np  = compute_mel_spectrogram(wav_path)             # [128, 130]
    mel_t   = torch.tensor(mel_np).unsqueeze(0)            # [1, 128, 130]
    dmel_t  = delta(mel_t)                                  # [1, 128, 130]
    d2mel_t = delta(dmel_t)                                 # [1, 128, 130]

    mel   = mel_np                                          # [128, 130]
    dmel  = dmel_t.squeeze(0).numpy()                      # [128, 130]
    d2mel = d2mel_t.squeeze(0).numpy()                     # [128, 130]

    # Time axis in seconds
    times = np.arange(TIME_FRAMES) * HOP_LENGTH / SAMPLE_RATE   # [130]
    freqs = np.arange(mel.shape[0])                              # [128]

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "axes.edgecolor": FG,   "axes.labelcolor": FG,
        "xtick.color": FG,      "ytick.color": FG,
        "text.color": FG,       "font.size": 11,
    })

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Mel spectrogram and temporal derivatives\n{wav_path.name}",
        color=FG, fontsize=13, y=0.98,
    )

    panels = [
        (mel,   "viridis",  "mel spectrogram",
         "log-amplitude",
         "Harmonic content — кожна горизонтальна смуга = обертон"),
        (dmel,  "RdBu_r",   "Δmel  (перша похідна по часу)",
         "Δ amplitude",
         "Червоний = енергія зростає (attack)   Синій = енергія спадає (decay)"),
        (d2mel, "PuOr_r",   "Δ²mel  (друга похідна по часу)",
         "Δ² amplitude",
         "Фіолетовий = пік зростання (початок удару)   Помаранчевий = пік спадання"),
    ]

    for ax, (data, cmap, title, cbar_label, note) in zip(axes, panels):
        # symmetric clim for derivative panels
        if "Δ" in title:
            vmax = np.percentile(np.abs(data), 98)
            vmin = -vmax
        else:
            vmin, vmax = data.min(), data.max()

        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title, color=FG, fontsize=12, loc="left", pad=6)
        ax.set_ylabel("Mel bin", color=FG)
        ax.text(0.01, 0.04, note, transform=ax.transAxes,
                color=FG, fontsize=9, alpha=0.8,
                bbox=dict(facecolor=BG, edgecolor="none", alpha=0.6))

        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
        cb.set_label(cbar_label, color=FG, fontsize=9)
        cb.ax.yaxis.set_tick_params(color=FG)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG, fontsize=8)

    axes[-1].set_xlabel("Time (s)", color=FG)

    # ── Envelope annotations ──────────────────────────────────────────────────
    # Find approximate attack peak in Δ²mel (large positive values early on)
    mean_d2 = d2mel.mean(axis=0)           # [130]  averaged over freq
    attack_frame = int(np.argmax(mean_d2[:30]))   # look in first 30 frames
    t_attack = times[attack_frame]

    for ax in axes:
        ax.axvline(t_attack, color="#ffd700", lw=1.2, ls="--", alpha=0.7)

    axes[0].text(t_attack + 0.03, freqs[-1] * 0.9, "attack",
                 color="#ffd700", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT_PATH}")

    # ── Print quick stats ─────────────────────────────────────────────────────
    print(f"\nQuick stats:")
    print(f"  mel   range: [{mel.min():.2f}, {mel.max():.2f}]")
    print(f"  Δmel  range: [{dmel.min():.2f}, {dmel.max():.2f}]  "
          f"(|mean|={np.abs(dmel).mean():.3f})")
    print(f"  Δ²mel range: [{d2mel.min():.2f}, {d2mel.max():.2f}]  "
          f"(|mean|={np.abs(d2mel).mean():.3f})")
    print(f"  Estimated attack frame: {attack_frame}  ({t_attack:.2f}s)")


if __name__ == "__main__":
    main()
