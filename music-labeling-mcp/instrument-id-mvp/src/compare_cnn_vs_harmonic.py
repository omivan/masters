"""
Detailed comparison: InstrumentCNN (baseline) vs HarmonicCNN.

Runs both models with N_SEEDS different random seeds, then reports:
  • mean ± std F1-macro / F1-micro
  • per-class precision, recall, F1
  • per-instrument-family aggregated F1
  • 5 plots + metrics.json + report.txt saved to plots/harmonic_detail/

Usage:
    python src/compare_cnn_vs_harmonic.py
    python src/compare_cnn_vs_harmonic.py --epochs 30 --seeds 3
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import CLASSES, NUM_CLASSES, IRMASDataset, load_training_dataset, train_val_split
from model import InstrumentCNN, HarmonicCNN

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "data" / "IRMAS" / "training"

# ── Instrument families ───────────────────────────────────────────────────────
# Grouping by how prominently harmonic overtones appear
FAMILIES = {
    "Bowed strings": ["cel", "vio"],
    "Woodwinds":     ["cla", "flu", "sax"],
    "Brass":         ["tru"],
    "Plucked":       ["gac", "gel"],
    "Keyboard":      ["org", "pia"],
    "Voice":         ["voi"],
}
CLASS_TO_FAMILY = {cls: fam for fam, clss in FAMILIES.items() for cls in clss}

# Dark theme colours
BG     = "#0f1117"
FG     = "#e0e0e0"
BLUE   = "#4c8eff"
GREEN  = "#26c6a2"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_training_root(base: Path) -> Path:
    instrument_names = set(CLASSES)
    if any((base / c).is_dir() for c in instrument_names):
        return base
    for child in sorted(base.iterdir()):
        if child.is_dir() and any((child / c).is_dir() for c in instrument_names):
            return child
    raise RuntimeError(f"Cannot find instrument folders under {base}.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer, device, desc):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate_detailed(model, loader, device, threshold):
    """Returns preds and targets arrays for flexible metric computation."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            all_preds.append((probs >= threshold).astype(int))
            all_targets.append(y.numpy().astype(int))
    return np.vstack(all_preds), np.vstack(all_targets)


def compute_metrics(preds, targets):
    return {
        "f1_macro":    f1_score(targets, preds, average="macro",  zero_division=0),
        "f1_micro":    f1_score(targets, preds, average="micro",  zero_division=0),
        "f1_per":      f1_score(targets, preds, average=None,     zero_division=0),
        "prec_per":    precision_score(targets, preds, average=None, zero_division=0),
        "rec_per":     recall_score(targets, preds, average=None,  zero_division=0),
    }


def train_one_seed(model_cls, train_ds, val_ds, args, device, seed):
    set_seed(seed)

    model = model_cls().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    history = {"train": [], "val": []}
    for epoch in range(1, args.epochs + 1):
        tl = run_epoch(model, train_loader, criterion, optimizer, device,
                       f"seed={seed} E{epoch:>2} train")
        vl = run_epoch(model, val_loader, criterion, None, device,
                       f"seed={seed} E{epoch:>2} val")
        history["train"].append(tl)
        history["val"].append(vl)

    preds, targets = evaluate_detailed(model, val_loader, device, args.threshold)
    metrics = compute_metrics(preds, targets)
    metrics["history"] = history
    return metrics


def run_model(name, model_cls, train_ds, val_ds, args, device, seeds):
    print(f"\n{'='*60}")
    print(f"  {name}  —  {len(seeds)} seed(s): {seeds}")
    print(f"{'='*60}")
    n_params = sum(p.numel() for p in model_cls().parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    all_metrics = []
    t0 = time.time()
    for seed in seeds:
        print(f"\n  [seed {seed}]")
        m = train_one_seed(model_cls, train_ds, val_ds, args, device, seed)
        all_metrics.append(m)
        print(f"  → F1-macro={m['f1_macro']:.4f}  F1-micro={m['f1_micro']:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # Aggregate across seeds
    result = {
        "name":    name,
        "params":  n_params,
        "f1_macro_all": [m["f1_macro"] for m in all_metrics],
        "f1_micro_all": [m["f1_micro"] for m in all_metrics],
        "f1_macro": np.mean([m["f1_macro"] for m in all_metrics]),
        "f1_micro": np.mean([m["f1_micro"] for m in all_metrics]),
        "f1_macro_std": np.std([m["f1_macro"] for m in all_metrics]),
        "f1_micro_std": np.std([m["f1_micro"] for m in all_metrics]),
        "f1_per":   np.mean([m["f1_per"]   for m in all_metrics], axis=0),
        "f1_per_std": np.std([m["f1_per"]  for m in all_metrics], axis=0),
        "prec_per": np.mean([m["prec_per"] for m in all_metrics], axis=0),
        "rec_per":  np.mean([m["rec_per"]  for m in all_metrics], axis=0),
        "histories": [m["history"] for m in all_metrics],
    }
    return result


def print_comparison_table(cnn_r, harm_r, threshold):
    print(f"\n\n{'='*80}")
    print("  CNN (baseline)  vs  HarmonicCNN — Detailed Comparison")
    print(f"  Threshold: {threshold}")
    print(f"{'='*80}")
    print(f"\n  {'Metric':<16} {'CNN':>14} {'HarmonicCNN':>16} {'Δ':>10}")
    print(f"  {'-'*58}")

    seeds = len(cnn_r["f1_macro_all"])
    for key, label in [("f1_macro", "F1-macro"), ("f1_micro", "F1-micro")]:
        cv, hv = cnn_r[key], harm_r[key]
        cs, hs = cnn_r[f"{key}_std"], harm_r[f"{key}_std"]
        delta = hv - cv
        sign = "▲" if delta > 0 else "▼"
        if seeds > 1:
            print(f"  {label:<16} {cv:.4f}±{cs:.4f}   {hv:.4f}±{hs:.4f}   {sign}{abs(delta):.4f}")
        else:
            print(f"  {label:<16} {cv:.4f}        {hv:.4f}        {sign}{abs(delta):.4f}")

    params_d = harm_r["params"] - cnn_r["params"]
    print(f"  {'Parameters':<16} {cnn_r['params']:>14,} {harm_r['params']:>16,} {params_d:>+10,}")

    print(f"\n  Per-class  (mean over {seeds} seed(s)):")
    print(f"  {'Class':<6} {'Family':<16} "
          f"{'CNN P':>6} {'CNN R':>6} {'CNN F1':>7} "
          f"{'Hrm P':>6} {'Hrm R':>6} {'Hrm F1':>7} {'ΔF1':>7}")
    print(f"  {'-'*76}")

    for i, cls in enumerate(CLASSES):
        fam = CLASS_TO_FAMILY.get(cls, "?")
        cp, cr, cf1 = cnn_r["prec_per"][i],  cnn_r["rec_per"][i],  cnn_r["f1_per"][i]
        hp, hr, hf1 = harm_r["prec_per"][i], harm_r["rec_per"][i], harm_r["f1_per"][i]
        delta = hf1 - cf1
        sign = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else " ")
        best = "*" if hf1 > cf1 + 0.005 else (" " if cf1 > hf1 + 0.005 else "=")
        print(f"  {cls:<6} {fam:<16} "
              f"{cp:>6.3f} {cr:>6.3f} {cf1:>7.3f} "
              f"{hp:>6.3f} {hr:>6.3f} {hf1:>7.3f} "
              f"{sign}{abs(delta):.3f}{best}")

    # Family aggregates
    print(f"\n  Per-family average F1:")
    print(f"  {'Family':<16} {'CNN':>8} {'HarmCNN':>10} {'Δ':>7}")
    print(f"  {'-'*44}")
    for fam, clss in FAMILIES.items():
        idxs = [CLASSES.index(c) for c in clss]
        cf1 = float(np.mean([cnn_r["f1_per"][i]  for i in idxs]))
        hf1 = float(np.mean([harm_r["f1_per"][i] for i in idxs]))
        delta = hf1 - cf1
        sign = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else " ")
        print(f"  {fam:<16} {cf1:>8.3f} {hf1:>10.3f} {sign}{abs(delta):.3f}")

    print(f"\n{'='*80}\n")


def save_metrics(cnn_r, harm_r, args, total, out_dir):
    """Save metrics to JSON and a human-readable text report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = len(cnn_r["histories"])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── JSON ──────────────────────────────────────────────────────────────────
    data = {
        "meta": {
            "timestamp": ts,
            "epochs": args.epochs,
            "seeds": seeds,
            "threshold": args.threshold,
            "total_samples": total,
        },
        "CNN": {
            "params": cnn_r["params"],
            "f1_macro_mean": float(cnn_r["f1_macro"]),
            "f1_macro_std":  float(cnn_r["f1_macro_std"]),
            "f1_micro_mean": float(cnn_r["f1_micro"]),
            "f1_micro_std":  float(cnn_r["f1_micro_std"]),
            "f1_macro_per_seed": [float(v) for v in cnn_r["f1_macro_all"]],
            "per_class": {
                cls: {
                    "f1":        float(cnn_r["f1_per"][i]),
                    "f1_std":    float(cnn_r["f1_per_std"][i]),
                    "precision": float(cnn_r["prec_per"][i]),
                    "recall":    float(cnn_r["rec_per"][i]),
                }
                for i, cls in enumerate(CLASSES)
            },
        },
        "HarmonicCNN": {
            "params": harm_r["params"],
            "f1_macro_mean": float(harm_r["f1_macro"]),
            "f1_macro_std":  float(harm_r["f1_macro_std"]),
            "f1_micro_mean": float(harm_r["f1_micro"]),
            "f1_micro_std":  float(harm_r["f1_micro_std"]),
            "f1_macro_per_seed": [float(v) for v in harm_r["f1_macro_all"]],
            "per_class": {
                cls: {
                    "f1":        float(harm_r["f1_per"][i]),
                    "f1_std":    float(harm_r["f1_per_std"][i]),
                    "precision": float(harm_r["prec_per"][i]),
                    "recall":    float(harm_r["rec_per"][i]),
                }
                for i, cls in enumerate(CLASSES)
            },
        },
        "delta": {
            "f1_macro": float(harm_r["f1_macro"] - cnn_r["f1_macro"]),
            "f1_micro": float(harm_r["f1_micro"] - cnn_r["f1_micro"]),
            "per_class": {
                cls: float(harm_r["f1_per"][i] - cnn_r["f1_per"][i])
                for i, cls in enumerate(CLASSES)
            },
        },
        "per_family": {
            fam: {
                "CNN":        float(np.mean([cnn_r["f1_per"][CLASSES.index(c)]  for c in clss])),
                "HarmonicCNN": float(np.mean([harm_r["f1_per"][CLASSES.index(c)] for c in clss])),
                "delta":      float(np.mean([harm_r["f1_per"][CLASSES.index(c)] - cnn_r["f1_per"][CLASSES.index(c)] for c in clss])),
            }
            for fam, clss in FAMILIES.items()
        },
    }

    json_path = out_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {json_path.name}")

    # ── Text report ───────────────────────────────────────────────────────────
    lines = []
    lines.append(f"CNN vs HarmonicCNN — Comparison Report")
    lines.append(f"Generated: {ts}")
    lines.append(f"Epochs: {args.epochs}  |  Seeds: {seeds}  |  Threshold: {args.threshold}  |  Samples: {total}")
    lines.append("=" * 72)

    lines.append("")
    lines.append("OVERALL METRICS")
    lines.append(f"  {'Metric':<16} {'CNN':>18} {'HarmonicCNN':>20} {'Delta':>10}")
    lines.append(f"  {'-'*66}")
    for key, label in [("f1_macro", "F1-macro"), ("f1_micro", "F1-micro")]:
        cv, cs = cnn_r[key],  cnn_r[f"{key}_std"]
        hv, hs = harm_r[key], harm_r[f"{key}_std"]
        delta = hv - cv
        sign = "+" if delta >= 0 else ""
        lines.append(f"  {label:<16} {cv:.4f} ± {cs:.4f}    {hv:.4f} ± {hs:.4f}    {sign}{delta:.4f}")
    lines.append(f"  {'Parameters':<16} {cnn_r['params']:>18,} {harm_r['params']:>20,}")

    lines.append("")
    lines.append("PER-CLASS (mean over seeds)")
    lines.append(f"  {'Class':<6} {'Family':<16} "
                 f"{'CNN P':>6} {'CNN R':>6} {'CNN F1':>7} "
                 f"{'Hrm P':>6} {'Hrm R':>6} {'Hrm F1':>7} {'Delta':>7}")
    lines.append(f"  {'-'*76}")
    for i, cls in enumerate(CLASSES):
        fam = CLASS_TO_FAMILY.get(cls, "?")
        cp, cr, cf1 = cnn_r["prec_per"][i],  cnn_r["rec_per"][i],  cnn_r["f1_per"][i]
        hp, hr, hf1 = harm_r["prec_per"][i], harm_r["rec_per"][i], harm_r["f1_per"][i]
        delta = hf1 - cf1
        sign = "+" if delta >= 0 else ""
        lines.append(f"  {cls:<6} {fam:<16} "
                     f"{cp:>6.3f} {cr:>6.3f} {cf1:>7.3f} "
                     f"{hp:>6.3f} {hr:>6.3f} {hf1:>7.3f} {sign}{abs(delta):.3f}")

    lines.append("")
    lines.append("PER-FAMILY")
    lines.append(f"  {'Family':<16} {'CNN':>8} {'HarmonicCNN':>13} {'Delta':>8}")
    lines.append(f"  {'-'*48}")
    for fam, clss in FAMILIES.items():
        idxs = [CLASSES.index(c) for c in clss]
        cf1 = float(np.mean([cnn_r["f1_per"][i]  for i in idxs]))
        hf1 = float(np.mean([harm_r["f1_per"][i] for i in idxs]))
        delta = hf1 - cf1
        sign = "+" if delta >= 0 else ""
        lines.append(f"  {fam:<16} {cf1:>8.3f} {hf1:>13.3f} {sign}{abs(delta):.3f}")

    lines.append("")
    lines.append("PER-SEED F1-MACRO")
    lines.append(f"  {'Seed':<6} {'CNN':>8} {'HarmonicCNN':>14}")
    lines.append(f"  {'-'*30}")
    for s, (cv, hv) in enumerate(zip(cnn_r["f1_macro_all"], harm_r["f1_macro_all"])):
        lines.append(f"  {s:<6} {cv:>8.4f} {hv:>14.4f}")

    txt_path = out_dir / "report.txt"
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {txt_path.name}")


def save_plots(cnn_r, harm_r, args, total, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
        "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
        "text.color": FG, "grid.color": "#2a2a3a", "grid.linestyle": "--",
        "font.size": 11,
    })

    seeds = len(cnn_r["histories"])
    epochs = list(range(1, args.epochs + 1))

    # ── Plot 1: Training loss curves (mean ± std band) ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Training loss  ({seeds} seed(s), {total} samples, {args.epochs} epochs)",
                 color=FG, fontsize=13)

    for ax, split in zip(axes, ["train", "val"]):
        for r, col, label in [(cnn_r, BLUE, "CNN (baseline)"), (harm_r, GREEN, "HarmonicCNN")]:
            mat = np.array([h[split] for h in r["histories"]])  # [seeds, epochs]
            mean = mat.mean(axis=0)
            std  = mat.std(axis=0)
            ax.plot(epochs, mean, color=col, lw=2, label=label)
            if seeds > 1:
                ax.fill_between(epochs, mean - std, mean + std, color=col, alpha=0.2)
        ax.set_title(f"{split.capitalize()} loss", color=FG)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCEWithLogitsLoss")
        ax.legend(facecolor="#1e1e2e", edgecolor=FG)
        ax.grid(True)

    plt.tight_layout()
    out = out_dir / "h1_loss_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 2: Per-class F1 side-by-side ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    x = np.arange(len(CLASSES))
    w = 0.35
    bars_cnn  = ax.bar(x - w/2, cnn_r["f1_per"],  w, color=BLUE,  label="CNN (baseline)", zorder=3)
    bars_harm = ax.bar(x + w/2, harm_r["f1_per"], w, color=GREEN, label="HarmonicCNN",     zorder=3)

    if seeds > 1:
        ax.errorbar(x - w/2, cnn_r["f1_per"],  yerr=cnn_r["f1_per_std"],
                    fmt="none", color=FG, capsize=3, lw=1.2)
        ax.errorbar(x + w/2, harm_r["f1_per"], yerr=harm_r["f1_per_std"],
                    fmt="none", color=FG, capsize=3, lw=1.2)

    for bar in list(bars_cnn) + list(bars_harm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color=FG)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 score")
    ax.set_title(f"Per-class F1  (threshold={args.threshold})", color=FG)
    ax.legend(facecolor="#1e1e2e", edgecolor=FG)
    ax.grid(axis="y", zorder=0)

    plt.tight_layout()
    out = out_dir / "h2_per_class_f1.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 3: ΔF1 HarmonicCNN - CNN ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(BG)

    deltas = harm_r["f1_per"] - cnn_r["f1_per"]
    bar_colors = [GREEN if d >= 0 else "#e57373" for d in deltas]
    bars = ax.bar(CLASSES, deltas, color=bar_colors, zorder=3)

    if seeds > 1:
        delta_std = np.sqrt(cnn_r["f1_per_std"]**2 + harm_r["f1_per_std"]**2)
        ax.errorbar(range(len(CLASSES)), deltas, yerr=delta_std,
                    fmt="none", color=FG, capsize=4, lw=1.2)

    ax.axhline(0, color=FG, lw=0.9)
    for i, (d, bar) in enumerate(zip(deltas, bars)):
        ax.text(i, d + (0.006 if d >= 0 else -0.015),
                f"{d:+.3f}", ha="center",
                va="bottom" if d >= 0 else "top", fontsize=9, color=FG)

    macro_d = harm_r["f1_macro"] - cnn_r["f1_macro"]
    ax.set_title(f"ΔF1 (HarmonicCNN − CNN baseline)  macro Δ={macro_d:+.4f}", color=FG)
    ax.set_ylabel("ΔF1")
    ax.grid(axis="y", zorder=0)

    plt.tight_layout()
    out = out_dir / "h3_delta_f1.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 4: Precision / Recall / F1 radar for each model ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Precision / Recall / F1 per class", color=FG, fontsize=13)

    for ax, r, col in [(axes[0], cnn_r, BLUE), (axes[1], harm_r, GREEN)]:
        x = np.arange(len(CLASSES))
        w = 0.25
        ax.bar(x - w, r["prec_per"], w, color=col,    alpha=0.9, label="Precision", zorder=3)
        ax.bar(x,     r["rec_per"],  w, color="#ffb74d", alpha=0.9, label="Recall",    zorder=3)
        ax.bar(x + w, r["f1_per"],   w, color=FG,     alpha=0.7, label="F1",         zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_title(r["name"], color=FG)
        ax.legend(facecolor="#1e1e2e", edgecolor=FG, fontsize=9)
        ax.grid(axis="y", zorder=0)

    plt.tight_layout()
    out = out_dir / "h4_prec_rec_f1.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 5: Per-family F1 bar ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(BG)

    fam_names = list(FAMILIES.keys())
    fam_cnn, fam_harm = [], []
    for fam, clss in FAMILIES.items():
        idxs = [CLASSES.index(c) for c in clss]
        fam_cnn.append(float(np.mean([cnn_r["f1_per"][i]  for i in idxs])))
        fam_harm.append(float(np.mean([harm_r["f1_per"][i] for i in idxs])))

    xf = np.arange(len(fam_names))
    wf = 0.35
    b1 = ax.bar(xf - wf/2, fam_cnn,  wf, color=BLUE,  label="CNN (baseline)", zorder=3)
    b2 = ax.bar(xf + wf/2, fam_harm, wf, color=GREEN, label="HarmonicCNN",     zorder=3)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, color=FG)

    ax.set_xticks(xf)
    ax.set_xticklabels(fam_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean F1 score")
    ax.set_title("Average F1 by instrument family", color=FG)
    ax.legend(facecolor="#1e1e2e", edgecolor=FG)
    ax.grid(axis="y", zorder=0)

    plt.tight_layout()
    out = out_dir / "h5_family_f1.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    print(f"\n  All plots saved to: {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="CNN vs HarmonicCNN detailed comparison")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--threshold",  type=float, default=0.3)
    parser.add_argument("--seeds",      type=int,   default=3,
                        help="Number of random seeds (runs) — use 1 for a quick test")
    parser.add_argument("--data-dir",   type=Path,  default=DEFAULT_TRAIN_ROOT)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    train_root = find_training_root(args.data_dir)
    full_ds    = load_training_dataset(train_root)
    total      = len(full_ds)
    print(f"Dataset: {total} samples  ({total // 11} avg/class)")

    # Fixed split — same for all seeds (seed=42 for the split itself)
    train_paths, train_labels, val_paths, val_labels = train_val_split(
        full_ds.wav_paths, full_ds.labels, seed=42
    )
    train_ds = IRMASDataset(train_paths, train_labels)
    val_ds   = IRMASDataset(val_paths,   val_labels)
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    seeds = list(range(args.seeds))

    cnn_r  = run_model("CNN (baseline)", InstrumentCNN,  train_ds, val_ds, args, device, seeds)
    harm_r = run_model("HarmonicCNN",    HarmonicCNN,    train_ds, val_ds, args, device, seeds)

    print_comparison_table(cnn_r, harm_r, args.threshold)

    out_dir = PROJECT_ROOT / "plots" / "harmonic_detail"
    save_metrics(cnn_r, harm_r, args, total, out_dir)
    save_plots(cnn_r, harm_r, args, total, out_dir)


if __name__ == "__main__":
    main()
