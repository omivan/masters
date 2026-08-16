"""
Visualization of HarmonicCNN input channels.
Shows how one mel spectrogram is shifted by octaves to create 4 channels.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
NPY_FILE    = Path(__file__).parent.parent / "data/IRMAS/training/IRMAS-TrainingData/pia/001__[pia][nod][cla]1389__1.npy"
SR          = 22050
N_MELS      = 128
TIME_FRAMES = 130
N_OCTAVES   = 4

# Мел-шкала: +1 октава ≈ 16 бінів при 128 мелах
OCTAVE_BINS = N_MELS // (N_OCTAVES * 2)  # ~16

# Приблизні частоти для мел-бінів (без librosa)
def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def bin_to_hz(b, n_mels=128, fmax=11025):
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel = b / n_mels * mel_max
    return mel_to_hz(mel)


def load_mel(path):
    return np.load(path)  # (128, 130), already log-mel


def octave_shift(mel, n_octaves):
    """Shift mel spectrogram up by n_octaves (roll rows, zero-pad bottom)."""
    shift = n_octaves * OCTAVE_BINS
    shifted = np.zeros_like(mel)
    if shift < N_MELS:
        shifted[shift:, :] = mel[:N_MELS - shift, :]
    return shifted


def main():
    mel = load_mel(NPY_FILE)
    channels = [octave_shift(mel, i) for i in range(N_OCTAVES)]

    # ── Мел-бін → Hz для осі Y ────────────────────────────────────────────────
    ytick_bins   = [0, 16, 32, 48, 64, 80, 96, 112, 127]
    ytick_labels = [f"{bin_to_hz(b):.0f}" for b in ytick_bins]

    vmin, vmax = mel.min(), mel.max()

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("HarmonicCNN: 4 вхідних канали\n(одна мел-спектрограма, зсунута на 0/1/2/3 октави)",
                 fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.88, bottom=0.08)

    channel_colors = ['Blues', 'Greens', 'Oranges', 'Reds']
    channel_names  = ['Канал 0\n(без зсуву)', 'Канал 1\n(+1 октава)',
                      'Канал 2\n(+2 октави)', 'Канал 3\n(+3 октави)']

    # ── Рядок 1: всі 4 канали ─────────────────────────────────────────────────
    axes = []
    for i in range(N_OCTAVES):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(channels[i], aspect='auto', origin='lower',
                       cmap=channel_colors[i], vmin=vmin, vmax=vmax)
        ax.set_title(channel_names[i], fontsize=11, fontweight='bold')
        ax.set_xlabel("Час (фрейми)", fontsize=9)
        if i == 0:
            ax.set_ylabel("Мел-бін (частота)", fontsize=9)
            ax.set_yticks(ytick_bins)
            ax.set_yticklabels(ytick_labels, fontsize=7)
        else:
            ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")
    axes.append(ax)

    # ── Рядок 2: підсвічування зсуву на каналах 0 і 1 ────────────────────────
    shift = OCTAVE_BINS  # скільки рядків = 1 октава

    ax0 = fig.add_subplot(gs[1, 0])
    ax0.imshow(channels[0], aspect='auto', origin='lower',
               cmap='Blues', vmin=vmin, vmax=vmax, alpha=0.8)
    ax0.axhspan(0, shift - 1, color='yellow', alpha=0.35, label=f'0–{shift} бін')
    ax0.set_title("Канал 0: виділена нижня октава", fontsize=10)
    ax0.set_xlabel("Час (фрейми)", fontsize=9)
    ax0.set_ylabel("Мел-бін", fontsize=9)
    ax0.set_yticks(ytick_bins)
    ax0.set_yticklabels(ytick_labels, fontsize=7)
    ax0.legend(fontsize=8, loc='upper right')

    ax1 = fig.add_subplot(gs[1, 1])
    ax1.imshow(channels[1], aspect='auto', origin='lower',
               cmap='Greens', vmin=vmin, vmax=vmax, alpha=0.8)
    ax1.axhspan(shift, 2 * shift - 1, color='yellow', alpha=0.35,
                label=f'{shift}–{2*shift} бін')
    ax1.set_title("Канал 1: та сама октава після зсуву", fontsize=10)
    ax1.set_xlabel("Час (фрейми)", fontsize=9)
    ax1.set_yticks([])
    ax1.legend(fontsize=8, loc='upper right')

    # Стрілка між ax0 і ax1
    fig.text(0.415, 0.465, "→ зсув\n+1 окт", ha='center', va='center',
             fontsize=12, color='darkorange', fontweight='bold')

    # Пояснення
    ax_txt = fig.add_subplot(gs[1, 2:])
    ax_txt.axis('off')
    explanation = (
        "Ідея зсуву:\n\n"
        "Якщо інструмент грає ноту C3 (130 Гц),\n"
        "його обертони — C4, C5, C6\n"
        "(260, 520, 1040 Гц).\n\n"
        "Після зсуву на +1 октаву:\n"
        "C4 (обертон 1) опиняється там,\n"
        "де був C3 у каналі 0.\n\n"
        "CNN-фільтр 3×3 тепер бачить\n"
        "фундаментал і обертон\n"
        "в одному місці по осі Y."
    )
    ax_txt.text(0.05, 0.95, explanation, transform=ax_txt.transAxes,
                fontsize=10, va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── Рядок 3: порівняння CNN vs HarmonicCNN ────────────────────────────────
    ax_cnn = fig.add_subplot(gs[2, :2])
    ax_cnn.imshow(channels[0], aspect='auto', origin='lower',
                  cmap='Blues', vmin=vmin, vmax=vmax)

    # Позначити фундаментал і обертони (приклад для фортепіано)
    fund_bin = 10   # ~130 Гц (C3)
    for j, (obin, label) in enumerate([(fund_bin, 'фундаментал'),
                                        (fund_bin + 16, 'обертон 1'),
                                        (fund_bin + 32, 'обертон 2'),
                                        (fund_bin + 48, 'обертон 3')]):
        if obin < N_MELS:
            ax_cnn.axhline(obin, color='red', linewidth=1.2, alpha=0.7,
                           linestyle='--')
            ax_cnn.text(TIME_FRAMES + 1, obin, label, fontsize=7,
                        va='center', color='red')
    ax_cnn.set_title("Звичайний CNN: фундаментал і обертони — далеко один від одного",
                     fontsize=10)
    ax_cnn.set_xlabel("Час (фрейми)", fontsize=9)
    ax_cnn.set_ylabel("Мел-бін", fontsize=9)
    ax_cnn.set_yticks(ytick_bins)
    ax_cnn.set_yticklabels(ytick_labels, fontsize=7)
    ax_cnn.set_xlim(0, TIME_FRAMES + 25)

    ax_harm = fig.add_subplot(gs[2, 2:])
    stacked_vis = np.stack(channels, axis=0).mean(axis=0)  # усереднення для візуалізації
    ax_harm.imshow(stacked_vis, aspect='auto', origin='lower',
                   cmap='viridis', vmin=vmin, vmax=vmax)
    ax_harm.axhline(fund_bin, color='yellow', linewidth=1.5, alpha=0.9,
                    linestyle='--', label='всі обертони → один рівень')
    ax_harm.set_title("HarmonicCNN: після стекання каналів обертони вирівняні",
                      fontsize=10)
    ax_harm.set_xlabel("Час (фрейми)", fontsize=9)
    ax_harm.set_ylabel("Мел-бін", fontsize=9)
    ax_harm.set_yticks(ytick_bins)
    ax_harm.set_yticklabels(ytick_labels, fontsize=7)
    ax_harm.legend(fontsize=8, loc='upper right')

    out_path = Path(__file__).parent.parent / "plots" / "harmonic_channels_viz.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
