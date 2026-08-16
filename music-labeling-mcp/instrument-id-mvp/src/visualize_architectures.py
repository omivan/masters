"""
Visualize InstrumentCNN (baseline) and HarmonicCNN architectures.
Saves two PNG files to plots/.

Usage:
    python src/visualize_architectures.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).parent.parent
PLOTS_DIR    = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

BG    = "#0f1117"
FG    = "#e0e0e0"
GRID  = "#2a2a3a"

# Layer colours
C_CONV   = "#4c8eff"   # blue   — Conv2d
C_BN     = "#78909c"   # grey   — BatchNorm
C_RELU   = "#26c6a2"   # green  — ReLU
C_POOL   = "#ff7043"   # orange — MaxPool / AdaptiveAvgPool
C_DROP   = "#ce93d8"   # purple — Dropout
C_FLAT   = "#78909c"   # grey   — Flatten
C_FC     = "#ffd54f"   # yellow — Linear
C_INPUT  = "#546e7a"   # dark   — Input
C_OUTPUT = "#ef5350"   # red    — Output
C_STACK  = "#00bcd4"   # cyan   — HarmonicStack (special)


def draw_layer(ax, x, y, w, h, color, label, sublabel="", fontsize=8):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=FG, linewidth=0.8, alpha=0.92,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y + (0.04 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            color="white", fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.1, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                color="#cccccc", zorder=4)


def arrow(ax, x, y1, y2, color=FG):
    ax.annotate("", xy=(x, y2 + 0.01), xytext=(x, y1 - 0.01),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
                zorder=2)


def bracket_label(ax, x_left, x_right, y, label, color=FG):
    """Horizontal bracket with label above two columns."""
    mid = (x_left + x_right) / 2
    ax.annotate("", xy=(x_right, y), xytext=(x_left, y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.0))
    ax.text(mid, y + 0.07, label, ha="center", va="bottom",
            fontsize=7.5, color=color)


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 1: InstrumentCNN (baseline)
# ─────────────────────────────────────────────────────────────────────────────
def plot_cnn_baseline():
    fig, ax = plt.subplots(figsize=(5, 14))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 14.5)
    ax.axis("off")
    ax.set_title("InstrumentCNN  (baseline)\n288,747 parameters",
                 color=FG, fontsize=12, fontweight="bold", pad=10)

    cx = 0.5
    W  = 0.55
    H  = 0.30

    layers = [
        # (y,      color,   label,              sublabel)
        (14.0, C_INPUT,  "Input",             "[B, 1, 128, 130]"),
        (13.2, C_CONV,   "Conv2d  1→16",      "3×3, pad=1"),
        (12.5, C_BN,     "BatchNorm2d",       ""),
        (12.0, C_RELU,   "ReLU",              ""),
        (11.5, C_POOL,   "MaxPool2d",         "2×2  →  [B,16,64,65]"),
        (10.5, C_CONV,   "Conv2d  16→32",     "3×3, pad=1"),
        ( 9.8, C_BN,     "BatchNorm2d",       ""),
        ( 9.3, C_RELU,   "ReLU",              ""),
        ( 8.8, C_POOL,   "MaxPool2d",         "2×2  →  [B,32,32,32]"),
        ( 7.8, C_CONV,   "Conv2d  32→64",     "3×3, pad=1"),
        ( 7.1, C_BN,     "BatchNorm2d",       ""),
        ( 6.6, C_RELU,   "ReLU",              ""),
        ( 6.1, C_POOL,   "AdaptiveAvgPool",   "→  [B, 64, 4, 4]"),
        ( 5.4, C_DROP,   "Dropout2d",         "p=0.25"),
        ( 4.7, C_FLAT,   "Flatten",           "→  [B, 1024]"),
        ( 3.9, C_FC,     "Linear",            "1024 → 256"),
        ( 3.2, C_RELU,   "ReLU",              ""),
        ( 2.7, C_DROP,   "Dropout",           "p=0.30"),
        ( 1.9, C_FC,     "Linear",            "256 → 11"),
        ( 1.0, C_OUTPUT, "Output",            "[B, 11]  raw logits"),
    ]

    for y, col, lbl, sub in layers:
        draw_layer(ax, cx, y, W, H, col, lbl, sub)

    # arrows between consecutive layers
    ys = [l[0] for l in layers]
    for i in range(len(ys) - 1):
        arrow(ax, cx, ys[i] - H/2, ys[i+1] + H/2)

    # Block brackets on the right
    def side_bracket(ax, y_top, y_bot, label, x=0.82):
        mid = (y_top + y_bot) / 2
        ax.annotate("", xy=(x, y_bot - 0.05), xytext=(x, y_top + 0.05),
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=1.5,
                                   connectionstyle="arc3,rad=0"))
        # left tick
        for yy in [y_top + 0.05, y_bot - 0.05]:
            ax.plot([x - 0.02, x], [yy, yy], color="#aaaaaa", lw=1.2)
        ax.text(x + 0.03, mid, label, va="center", ha="left",
                fontsize=7.5, color="#aaaaaa")

    side_bracket(ax, 14.0 - H/2, 11.5 - H/2, "Block 1")
    side_bracket(ax, 10.5 - H/2,  8.8 - H/2, "Block 2")
    side_bracket(ax,  7.8 - H/2,  5.4 - H/2, "Block 3")
    side_bracket(ax,  4.7 - H/2,  1.9 - H/2, "Classifier")

    plt.tight_layout()
    out = PLOTS_DIR / "arch_baseline_cnn.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 2: HarmonicCNN
# ─────────────────────────────────────────────────────────────────────────────
def plot_harmonic_cnn():
    fig, ax = plt.subplots(figsize=(7, 16))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 16.5)
    ax.axis("off")
    ax.set_title("HarmonicCNN\n621,355 parameters",
                 color=FG, fontsize=12, fontweight="bold", pad=10)

    cx = 0.5
    W  = 0.62
    H  = 0.30

    # ── Harmonic stack illustration ───────────────────────────────────────────
    # Show 4 shifted copies stacking into 4-channel input
    stack_ys = [15.8, 15.3, 14.8, 14.3]
    stack_xs = [0.25, 0.35, 0.45, 0.55]
    labels_h = ["ch 0: x[f]", "ch 1: x[f+12]", "ch 2: x[f+24]", "ch 3: x[f+36]"]
    for sx, sy, sl in zip(stack_xs, stack_ys, labels_h):
        box = FancyBboxPatch((sx - 0.10, sy - 0.10), 0.20, 0.18,
                             boxstyle="round,pad=0.02",
                             facecolor=C_STACK, edgecolor=FG, linewidth=0.7,
                             alpha=0.85, zorder=3)
        ax.add_patch(box)
        ax.text(sx, sy, sl, ha="center", va="center",
                fontsize=6.5, color="white", zorder=4)

    ax.text(0.5, 16.2, "Input  [B, 1, 128, 130]",
            ha="center", va="center", fontsize=9, color=FG,
            bbox=dict(facecolor=C_INPUT, edgecolor=FG, boxstyle="round,pad=0.3",
                      linewidth=0.8))

    # arrow from input down into stack
    ax.annotate("", xy=(0.5, 15.9), xytext=(0.5, 16.0),
                arrowprops=dict(arrowstyle="-|>", color=FG, lw=1.0), zorder=2)

    ax.text(0.5, 14.0,
            "_harmonic_stack()\n→  [B, 4, 128, 130]",
            ha="center", va="center", fontsize=8, color="white",
            bbox=dict(facecolor=C_STACK, edgecolor=FG,
                      boxstyle="round,pad=0.35", linewidth=0.9, alpha=0.95))

    arrow(ax, cx, 13.75, 13.35)

    layers = [
        (13.2, C_CONV,  "Conv2d  4→32",      "3×3, pad=1   (4-channel input)"),
        (12.5, C_BN,    "BatchNorm2d",       ""),
        (12.0, C_RELU,  "ReLU",              ""),
        (11.5, C_POOL,  "MaxPool2d",         "2×2  →  [B,32,64,65]"),
        (10.5, C_CONV,  "Conv2d  32→64",     "3×3, pad=1"),
        ( 9.8, C_BN,    "BatchNorm2d",       ""),
        ( 9.3, C_RELU,  "ReLU",              ""),
        ( 8.8, C_POOL,  "MaxPool2d",         "2×2  →  [B,64,32,32]"),
        ( 7.8, C_CONV,  "Conv2d  64→128",    "3×3, pad=1"),
        ( 7.1, C_BN,    "BatchNorm2d",       ""),
        ( 6.6, C_RELU,  "ReLU",              ""),
        ( 6.1, C_POOL,  "AdaptiveAvgPool",   "→  [B, 128, 4, 4]"),
        ( 5.4, C_DROP,  "Dropout2d",         "p=0.25"),
        ( 4.7, C_FLAT,  "Flatten",           "→  [B, 2048]"),
        ( 3.9, C_FC,    "Linear",            "2048 → 256"),
        ( 3.2, C_RELU,  "ReLU",              ""),
        ( 2.7, C_DROP,  "Dropout",           "p=0.30"),
        ( 1.9, C_FC,    "Linear",            "256 → 11"),
        ( 1.0, C_OUTPUT,"Output",            "[B, 11]  raw logits"),
    ]

    for y, col, lbl, sub in layers:
        draw_layer(ax, cx, y, W, H, col, lbl, sub)

    ys = [l[0] for l in layers]
    for i in range(len(ys) - 1):
        arrow(ax, cx, ys[i] - H/2, ys[i+1] + H/2)

    def side_bracket(ax, y_top, y_bot, label, x=0.86):
        mid = (y_top + y_bot) / 2
        ax.annotate("", xy=(x, y_bot - 0.05), xytext=(x, y_top + 0.05),
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=1.5))
        for yy in [y_top + 0.05, y_bot - 0.05]:
            ax.plot([x - 0.02, x], [yy, yy], color="#aaaaaa", lw=1.2)
        ax.text(x + 0.03, mid, label, va="center", ha="left",
                fontsize=7.5, color="#aaaaaa")

    side_bracket(ax, 13.2 - H/2, 11.5 - H/2, "Block 1")
    side_bracket(ax, 10.5 - H/2,  8.8 - H/2, "Block 2")
    side_bracket(ax,  7.8 - H/2,  5.4 - H/2, "Block 3")
    side_bracket(ax,  4.7 - H/2,  1.9 - H/2, "Classifier")

    # Key difference annotation
    ax.text(0.01, 14.05,
            "KEY: 3×3 conv at (f,t)\nsees energy at\nf, f+1oct, f+2oct, f+3oct\nsimultaneously",
            ha="left", va="center", fontsize=7, color=C_STACK,
            style="italic")

    plt.tight_layout()
    out = PLOTS_DIR / "arch_harmonic_cnn.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 3: Side-by-side comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(11, 14))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Architecture Comparison", color=FG, fontsize=13, fontweight="bold")

    titles = [
        "InstrumentCNN (baseline)\n288,747 params",
        "HarmonicCNN\n621,355 params",
    ]

    # (label, sublabel, color) for each model
    cnn_layers = [
        ("Input",           "[B, 1, 128, 130]",    C_INPUT),
        ("Conv2d 1→16",     "3×3",                 C_CONV),
        ("BN + ReLU",       "",                    C_BN),
        ("MaxPool 2×2",     "→ [B,16,64,65]",      C_POOL),
        ("Conv2d 16→32",    "3×3",                 C_CONV),
        ("BN + ReLU",       "",                    C_BN),
        ("MaxPool 2×2",     "→ [B,32,32,32]",      C_POOL),
        ("Conv2d 32→64",    "3×3",                 C_CONV),
        ("BN + ReLU",       "",                    C_BN),
        ("AdaptAvgPool",    "→ [B,64,4,4]",        C_POOL),
        ("Dropout2d 0.25",  "",                    C_DROP),
        ("Flatten",         "→ [B,1024]",          C_FLAT),
        ("Linear 1024→256", "",                    C_FC),
        ("ReLU + Drop 0.3", "",                    C_RELU),
        ("Linear 256→11",   "",                    C_FC),
        ("Output",          "[B, 11] logits",      C_OUTPUT),
    ]

    harm_layers = [
        ("Input",            "[B, 1, 128, 130]",   C_INPUT),
        ("HarmonicStack",    "→ [B, 4, 128, 130]", C_STACK),
        ("Conv2d 4→32",      "3×3  (4-ch input)",  C_CONV),
        ("BN + ReLU",        "",                   C_BN),
        ("MaxPool 2×2",      "→ [B,32,64,65]",     C_POOL),
        ("Conv2d 32→64",     "3×3",                C_CONV),
        ("BN + ReLU",        "",                   C_BN),
        ("MaxPool 2×2",      "→ [B,64,32,32]",     C_POOL),
        ("Conv2d 64→128",    "3×3",                C_CONV),
        ("BN + ReLU",        "",                   C_BN),
        ("AdaptAvgPool",     "→ [B,128,4,4]",      C_POOL),
        ("Dropout2d 0.25",   "",                   C_DROP),
        ("Flatten",          "→ [B,2048]",         C_FLAT),
        ("Linear 2048→256",  "",                   C_FC),
        ("ReLU + Drop 0.3",  "",                   C_RELU),
        ("Linear 256→11",    "",                   C_FC),
        ("Output",           "[B, 11] logits",     C_OUTPUT),
    ]

    all_layers = [cnn_layers, harm_layers]

    for ax, title, layers in zip(axes, titles, all_layers):
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        n = len(layers)
        ax.set_ylim(-0.5, n + 0.5)
        ax.axis("off")
        ax.set_title(title, color=FG, fontsize=10, fontweight="bold", pad=8)

        cx = 0.5
        W, H = 0.80, 0.42
        step = 1.0

        for i, (lbl, sub, col) in enumerate(reversed(layers)):
            y = i * step
            box = FancyBboxPatch(
                (cx - W/2, y - H/2), W, H,
                boxstyle="round,pad=0.03",
                facecolor=col, edgecolor=FG, linewidth=0.7,
                alpha=0.90, zorder=3,
            )
            ax.add_patch(box)
            ax.text(cx, y + (0.07 if sub else 0), lbl,
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold", zorder=4)
            if sub:
                ax.text(cx, y - 0.10, sub,
                        ha="center", va="center", fontsize=6.5,
                        color="#cccccc", zorder=4)
            if i < n - 1:
                ax.annotate("", xy=(cx, i * step + H/2 + 0.02),
                            xytext=(cx, (i-1) * step - H/2 - 0.02),
                            arrowprops=dict(arrowstyle="-|>", color=FG, lw=1.0),
                            zorder=2)

    # Legend
    legend_items = [
        mpatches.Patch(color=C_STACK,  label="HarmonicStack (novel)"),
        mpatches.Patch(color=C_CONV,   label="Conv2d"),
        mpatches.Patch(color=C_POOL,   label="Pooling"),
        mpatches.Patch(color=C_BN,     label="BatchNorm / Flatten"),
        mpatches.Patch(color=C_FC,     label="Linear (FC)"),
        mpatches.Patch(color=C_DROP,   label="Dropout"),
        mpatches.Patch(color=C_RELU,   label="ReLU"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               facecolor="#1e1e2e", edgecolor=FG,
               labelcolor=FG, fontsize=8, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = PLOTS_DIR / "arch_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_cnn_baseline()
    plot_harmonic_cnn()
    plot_comparison()
    print("\nDone. Check plots/")
