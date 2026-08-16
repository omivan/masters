"""
Train and compare InstrumentCNN (baseline) vs InstrumentCNNBiLSTM on the same data.

Usage:
    python src/compare.py
    python src/compare.py --epochs 30 --batch-size 32
"""

import argparse
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

BG     = "#0f1117"
FG     = "#e0e0e0"
BLUE   = "#4c8eff"
ORANGE = "#ff7043"
GREEN  = "#26c6a2"
PURPLE = "#ce93d8"

sys.path.insert(0, str(Path(__file__).parent))

from dataset import CLASSES, NUM_CLASSES, IRMASDataset, load_training_dataset, train_val_split
from model import InstrumentCNN, InstrumentCNNBiLSTM, HarmonicCNN, ProtoNet, ContrastiveCNN, AttackSustainCNN

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "data" / "IRMAS" / "training"


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


def evaluate(model, loader, device, threshold):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            all_preds.append((probs >= threshold).astype(int))
            all_targets.append(y.numpy().astype(int))
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    return (
        f1_score(targets, preds, average="macro", zero_division=0),
        f1_score(targets, preds, average="micro", zero_division=0),
        f1_score(targets, preds, average=None, zero_division=0),
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── ProtoNet-specific helpers ─────────────────────────────────────────────────

def proto_loss(embeddings: torch.Tensor, labels: torch.Tensor,
               temperature: float = 10.0) -> torch.Tensor:
    """
    Multi-label prototypical loss.

    For every class k that has at least one positive example in the batch:
      1. Compute the batch prototype = mean of embeddings where label_k == 1.
      2. Compute cosine similarity of ALL embeddings to that prototype.
         (Embeddings are already L2-normalised → dot product = cosine sim.)
      3. Apply BCE: positive examples should score high, negatives low.

    Averaging over all active classes gives the final scalar loss.
    """
    loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    count = 0
    for k in range(labels.shape[1]):
        pos_mask = labels[:, k] > 0.5
        if pos_mask.sum() < 1:
            continue
        proto_k = embeddings[pos_mask].mean(dim=0)          # [D]
        sims_k  = (embeddings * proto_k).sum(dim=1)         # [B]  cosine sim
        loss = loss + F.binary_cross_entropy_with_logits(
            sims_k * temperature, labels[:, k]
        )
        count += 1
    return loss / max(count, 1)


def run_proto_epoch(model, loader, optimizer, device, desc):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device), y.to(device)
            embs = model(x)
            loss = proto_loss(embs, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def compute_prototypes(model, loader, device) -> torch.Tensor:
    """
    One full pass over the dataset to compute per-class prototypes.
    prototype[k] = L2-normalised mean of embeddings where label_k == 1.
    Returns: [NUM_CLASSES, embed_dim] on CPU.
    """
    model.eval()
    accum  = torch.zeros(NUM_CLASSES, model.embed_dim)
    counts = torch.zeros(NUM_CLASSES)
    with torch.no_grad():
        for x, y in loader:
            embs = model(x.to(device)).cpu()
            for k in range(NUM_CLASSES):
                mask = y[:, k] > 0.5
                if mask.sum() > 0:
                    accum[k]  += embs[mask].sum(dim=0)
                    counts[k] += mask.sum()
    prototypes = accum / counts.unsqueeze(1).clamp(min=1)
    return F.normalize(prototypes, dim=1)


def evaluate_proto(model, prototypes, loader, device, threshold=0.0):
    model.eval()
    all_preds, all_targets = [], []
    proto = prototypes.to(device)
    with torch.no_grad():
        for x, y in loader:
            embs = model(x.to(device))        # [B, D]
            sims = embs @ proto.T             # [B, NUM_CLASSES]  cosine sim
            preds = (sims >= threshold).cpu().int().numpy()
            all_preds.append(preds)
            all_targets.append(y.int().numpy())
    preds   = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    return (
        f1_score(targets, preds, average="macro", zero_division=0),
        f1_score(targets, preds, average="micro", zero_division=0),
        f1_score(targets, preds, average=None,    zero_division=0),
    )


def best_proto_threshold(model, prototypes, loader, device):
    """Sweep cosine-sim thresholds and return the one with best val F1-macro."""
    best_t, best_f1 = 0.0, 0.0
    for t in [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]:
        f1, _, _ = evaluate_proto(model, prototypes, loader, device, threshold=t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ── Contrastive pre-training helpers ─────────────────────────────────────────

def augment_mel(x: torch.Tensor) -> torch.Tensor:
    """
    Random spectrogram augmentation for contrastive learning.
    x: [1, F, T] — single sample (no batch dim).
    Returns an independently augmented view of the same sample.
    """
    x = x.clone()
    F_bins, T_bins = x.shape[-2], x.shape[-1]
    fill = -20.7  # log(1e-9) — log-zero sentinel

    # Frequency masking: zero out a random band of mel bins
    if random.random() > 0.2:
        w = random.randint(5, 25)
        s = random.randint(0, F_bins - w)
        x[..., s:s + w, :] = fill

    # Time masking: zero out a random time window
    if random.random() > 0.2:
        w = random.randint(5, 35)
        s = random.randint(0, T_bins - w)
        x[..., s:s + w] = fill

    # Time shift: circularly roll along time axis
    if random.random() > 0.3:
        x = torch.roll(x, random.randint(-15, 15), dims=-1)

    # Additive Gaussian noise
    if random.random() > 0.3:
        x = x + torch.randn_like(x) * 0.1

    return x


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss — SimCLR.

    z1, z2: [B, D]  L2-normalised embeddings.
    Positive pairs: (z1[i], z2[i]).
    All other 2(B-1) pairs per sample are negatives.
    """
    B   = z1.shape[0]
    z   = torch.cat([z1, z2], dim=0)        # [2B, D]
    sim = (z @ z.T) / temperature            # [2B, 2B]
    sim.fill_diagonal_(float("-inf"))        # mask self-similarity

    # z1[i]'s positive is z2[i] at position i+B, and vice-versa
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device),
    ])
    return F.cross_entropy(sim, labels)


def run_pretrain_epoch(model, loader, optimizer, device, desc):
    model.train()
    total_loss = 0.0
    for x, _ in tqdm(loader, desc=desc, leave=False):
        # Two independent augmented views of the same batch
        x1 = torch.stack([augment_mel(xi) for xi in x]).to(device)
        x2 = torch.stack([augment_mel(xi) for xi in x]).to(device)
        z1, z2 = model(x1), model(x2)
        loss = nt_xent_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def train_contrastive(name, model, train_loader, val_loader, epochs, lr, device):
    print(f"\n{'='*55}")
    print(f"  Training: {name}  ({count_params(model):,} params)")
    print(f"{'='*55}")

    history = {"train": [], "val": []}
    t0 = time.time()

    # ── Stage 1: contrastive pre-training ────────────────────────────────────
    print(f"  [Stage 1] Contrastive pre-training  ({epochs} epochs) …")
    model.set_mode(ContrastiveCNN.PRETRAIN)
    opt_pt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        tl = run_pretrain_epoch(model, train_loader, opt_pt, device, f"PT E{epoch:>2}")
        print(f"  PT Epoch {epoch:>3}/{epochs}  train_loss={tl:.4f}")

    # ── Stage 2: supervised fine-tuning ──────────────────────────────────────
    print(f"\n  [Stage 2] Supervised fine-tuning  ({epochs} epochs) …")
    model.set_mode(ContrastiveCNN.FINETUNE)
    criterion = nn.BCEWithLogitsLoss()
    # Lower LR for fine-tuning — encoder weights are already meaningful
    opt_ft = torch.optim.Adam(model.parameters(), lr=lr * 0.1)

    for epoch in range(1, epochs + 1):
        tl = run_epoch(model, train_loader, criterion, opt_ft, device, f"FT E{epoch:>2} train")
        vl = run_epoch(model, val_loader,   criterion, None,   device, f"FT E{epoch:>2} val  ")
        history["train"].append(tl)
        history["val"].append(vl)
        print(f"  FT Epoch {epoch:>3}/{epochs}  train={tl:.4f}  val={vl:.4f}")

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s  (pretrain + finetune)")
    return history, elapsed


def train_proto_model(name, model, train_loader, val_loader, epochs, lr, device):
    print(f"\n{'='*55}")
    print(f"  Training: {name}  ({count_params(model):,} params)")
    print(f"{'='*55}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train": [], "val": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tl = run_proto_epoch(model, train_loader, optimizer, device, f"E{epoch:>2} train")
        vl = run_proto_epoch(model, val_loader,   None,      device, f"E{epoch:>2} val  ")
        history["train"].append(tl)
        history["val"].append(vl)
        print(f"  Epoch {epoch:>3}/{epochs}  train={tl:.4f}  val={vl:.4f}")

    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")
    return history, elapsed


def train_model(name, model, train_loader, val_loader, epochs, lr, device):
    print(f"\n{'='*55}")
    print(f"  Training: {name}  ({count_params(model):,} params)")
    print(f"{'='*55}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train": [], "val": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tl = run_epoch(model, train_loader, criterion, optimizer, device, f"E{epoch:>2} train")
        vl = run_epoch(model, val_loader, criterion, None, device, f"E{epoch:>2} val  ")
        history["train"].append(tl)
        history["val"].append(vl)
        print(f"  Epoch {epoch:>3}/{epochs}  train={tl:.4f}  val={vl:.4f}")

    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")
    return history, elapsed


def main():
    parser = argparse.ArgumentParser(description="Compare CNN vs CNN+BiLSTM on IRMAS")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Sigmoid threshold for binary prediction (lower = more predictions)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # ── Load data (shared split) ──────────────────────────────────────────────
    train_root = find_training_root(args.data_dir)
    full_ds = load_training_dataset(train_root)
    total = len(full_ds)
    print(f"Dataset: {total} samples  ({total // 11} avg per class)")

    train_paths, train_labels, val_paths, val_labels = train_val_split(
        full_ds.wav_paths, full_ds.labels
    )
    train_ds = IRMASDataset(train_paths, train_labels)
    val_ds = IRMASDataset(val_paths, val_labels)
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    results = {}

    # ── Attack / Sustain CNN ──────────────────────────────────────────────────
    atksu = AttackSustainCNN().to(device)
    hist_atksu, time_atksu = train_model(
        "AttackSustainCNN", atksu, train_loader, val_loader, args.epochs, args.lr, device
    )
    f1_mac_as, f1_mic_as, f1_per_as = evaluate(atksu, val_loader, device, args.threshold)
    results["AttackSustain"] = {
        "params":         count_params(atksu),
        "train_time":     time_atksu,
        "f1_macro":       f1_mac_as,
        "f1_micro":       f1_mic_as,
        "f1_per_class":   f1_per_as,
        "final_val_loss": hist_atksu["val"][-1],
        "history":        hist_atksu,
    }

    # ── ContrastiveCNN ────────────────────────────────────────────────────────
    contra = ContrastiveCNN().to(device)
    hist_contra, time_contra = train_contrastive(
        "ContrastiveCNN", contra, train_loader, val_loader, args.epochs, args.lr, device
    )
    f1_mac_c, f1_mic_c, f1_per_c = evaluate(contra, val_loader, device, args.threshold)
    results["ContrastiveCNN"] = {
        "params": count_params(contra),
        "train_time": time_contra,
        "f1_macro": f1_mac_c,
        "f1_micro": f1_mic_c,
        "f1_per_class": f1_per_c,
        "final_val_loss": hist_contra["val"][-1],
        "history": hist_contra,
    }

    # ── ProtoNet ──────────────────────────────────────────────────────────────
    proto_model = ProtoNet(embed_dim=128).to(device)
    hist_proto, time_proto = train_proto_model(
        "ProtoNet (metric learning)", proto_model,
        train_loader, val_loader, args.epochs, args.lr, device
    )
    # Compute prototypes from full training split, then find best threshold
    train_loader_seq = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)
    prototypes = compute_prototypes(proto_model, train_loader_seq, device)
    best_t, _ = best_proto_threshold(proto_model, prototypes, val_loader, device)
    print(f"  Best cosine-sim threshold: {best_t}")
    f1_mac_p, f1_mic_p, f1_per_p = evaluate_proto(
        proto_model, prototypes, val_loader, device, threshold=best_t
    )
    results["ProtoNet"] = {
        "params": count_params(proto_model),
        "train_time": time_proto,
        "f1_macro": f1_mac_p,
        "f1_micro": f1_mic_p,
        "f1_per_class": f1_per_p,
        "final_val_loss": hist_proto["val"][-1],
        "history": hist_proto,
    }

    # ── Baseline CNN ──────────────────────────────────────────────────────────
    cnn = InstrumentCNN().to(device)
    hist_cnn, time_cnn = train_model(
        "CNN (baseline)", cnn, train_loader, val_loader, args.epochs, args.lr, device
    )
    f1_mac_cnn, f1_mic_cnn, f1_per_cnn = evaluate(cnn, val_loader, device, args.threshold)
    results["CNN"] = {
        "params": count_params(cnn),
        "train_time": time_cnn,
        "f1_macro": f1_mac_cnn,
        "f1_micro": f1_mic_cnn,
        "f1_per_class": f1_per_cnn,
        "final_val_loss": hist_cnn["val"][-1],
        "history": hist_cnn,
    }

    # ── Harmonic CNN ──────────────────────────────────────────────────────────
    harm = HarmonicCNN().to(device)
    hist_harm, time_harm = train_model(
        "HarmonicCNN (ours)", harm, train_loader, val_loader, args.epochs, args.lr, device
    )
    f1_mac_h, f1_mic_h, f1_per_h = evaluate(harm, val_loader, device, args.threshold)
    results["HarmonicCNN"] = {
        "params": count_params(harm),
        "train_time": time_harm,
        "f1_macro": f1_mac_h,
        "f1_micro": f1_mic_h,
        "f1_per_class": f1_per_h,
        "final_val_loss": hist_harm["val"][-1],
        "history": hist_harm,
    }

    # ── CNN + BiLSTM ──────────────────────────────────────────────────────────
    bilstm = InstrumentCNNBiLSTM().to(device)
    hist_bilstm, time_bilstm = train_model(
        "CNN + BiLSTM", bilstm, train_loader, val_loader, args.epochs, args.lr, device
    )
    f1_mac_b, f1_mic_b, f1_per_b = evaluate(bilstm, val_loader, device, args.threshold)
    results["CNN+BiLSTM"] = {
        "params": count_params(bilstm),
        "train_time": time_bilstm,
        "f1_macro": f1_mac_b,
        "f1_micro": f1_mic_b,
        "f1_per_class": f1_per_b,
        "final_val_loss": hist_bilstm["val"][-1],
        "history": hist_bilstm,
    }

    # ── Comparison table ──────────────────────────────────────────────────────
    models = ["CNN", "HarmonicCNN", "AttackSustain", "CNN+BiLSTM", "ContrastiveCNN", "ProtoNet"]
    shorts = {"CNN":"CNN","HarmonicCNN":"HarmCNN","AttackSustain":"AtkSus",
              "CNN+BiLSTM":"BiLSTM","ContrastiveCNN":"ContraCNN","ProtoNet":"ProtoNet"}

    print(f"\n\n{'='*104}")
    print("  COMPARISON RESULTS")
    print(f"  Threshold: {args.threshold}  |  Epochs: {args.epochs}  |  Data: {total} samples")
    print(f"  Note: ContrastiveCNN runs {args.epochs} pretrain + {args.epochs} finetune epochs")
    print(f"{'='*104}")

    col_w = 11
    header = f"  {'Metric':<20}" + "".join(f" {shorts[m]:>{col_w}}" for m in models)
    print(header)
    print(f"  {'-' * (20 + (col_w + 1) * len(models))}")

    def best_marker(vals):
        m = max(vals)
        return ["←" if v == m else " " for v in vals]

    def rowN(label, key, fmt=".4f"):
        vals = [results[m][key] for m in models]
        marks = best_marker(vals)
        if fmt == "int":
            cols = [f"{v:>{col_w},}" for v in vals]
        elif fmt == "time":
            cols = [f"{v:>{col_w-1}.1f}s" for v in vals]
        else:
            cols = [f"{v:>{col_w}{fmt}}" for v in vals]
        line = f"  {label:<20}"
        for col, mark in zip(cols, marks):
            line += f" {col}{mark}"
        return line

    print(rowN("Parameters",     "params",         "int"))
    print(rowN("Train time",     "train_time",     "time"))
    print(rowN("Final val loss", "final_val_loss",  ".4f"))
    print(rowN("F1-macro",       "f1_macro"))
    print(rowN("F1-micro",       "f1_micro"))

    print(f"\n  Per-class F1  (* = best):")
    hdr = f"  {'Class':<7}" + "".join(f" {shorts[m]:>10}" for m in models)
    print(hdr)
    print(f"  {'-'*70}")
    for i, cls in enumerate(CLASSES):
        vals = [results[m]["f1_per_class"][i] for m in models]
        best = max(vals)
        line = f"  {cls:<7}"
        for v in vals:
            line += f" {v:>9.3f}{'*' if v == best else ' '}"
        print(line)

    print(f"\n{'='*104}\n")

    # ── Save plots ────────────────────────────────────────────────────────────
    plots_dir = PROJECT_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    _save_plots(results, args, total, plots_dir)


def _save_plots(results, args, total, plots_dir):
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
        "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
        "text.color": FG, "grid.color": "#2a2a3a", "grid.linestyle": "--",
        "font.size": 11,
    })

    CYAN   = "#00bcd4"
    YELLOW = "#ffd54f"
    models = ["CNN", "HarmonicCNN", "AttackSustain", "CNN+BiLSTM", "ContrastiveCNN", "ProtoNet"]
    colors = {"CNN": BLUE, "HarmonicCNN": GREEN, "AttackSustain": YELLOW,
              "CNN+BiLSTM": ORANGE, "ContrastiveCNN": CYAN, "ProtoNet": PURPLE}
    labels = {"CNN": "CNN (baseline)", "HarmonicCNN": "HarmonicCNN",
              "AttackSustain": "AttackSustain", "CNN+BiLSTM": "CNN + BiLSTM",
              "ContrastiveCNN": "ContrastiveCNN", "ProtoNet": "ProtoNet"}
    epochs = list(range(1, args.epochs + 1))

    # ── Plot 1: Training curves ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Training curves  ({total} samples, {args.epochs} epochs)", color=FG, fontsize=13)

    for ax, split in zip(axes, ["train", "val"]):
        for m in models:
            ax.plot(epochs, results[m]["history"][split],
                    color=colors[m], lw=2, label=labels[m])
        ax.set_title(f"{split.capitalize()} loss", color=FG)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCEWithLogitsLoss")
        ax.legend(facecolor="#1e1e2e", edgecolor=FG)
        ax.grid(True)

    plt.tight_layout()
    out = plots_dir / "cmp_01_loss_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 2: F1-macro / F1-micro bar chart ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(BG)

    metrics     = ["F1-macro", "F1-micro"]
    metric_keys = ["f1_macro", "f1_micro"]
    x = np.arange(len(metrics))
    w = 0.12
    offsets = [-2.5*w, -1.5*w, -0.5*w, 0.5*w, 1.5*w, 2.5*w]

    for m, off in zip(models, offsets):
        vals = [results[m][k] for k in metric_keys]
        bars = ax.bar(x + off, vals, w, color=colors[m], label=labels[m], zorder=3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, color=FG)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Overall F1  (threshold={args.threshold})", color=FG)
    ax.legend(facecolor="#1e1e2e", edgecolor=FG)
    ax.grid(axis="y", zorder=0)

    plt.tight_layout()
    out = plots_dir / "cmp_02_f1_overall.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 3: Per-class F1 grouped bar ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor(BG)

    x = np.arange(len(CLASSES))
    w = 0.11
    offsets = [-2.5*w, -1.5*w, -0.5*w, 0.5*w, 1.5*w, 2.5*w]

    for m, off in zip(models, offsets):
        vals = results[m]["f1_per_class"]
        bars = ax.bar(x + off, vals, w, color=colors[m], label=labels[m], zorder=3)
        for bar in bars:
            if bar.get_height() > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7, color=FG)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 score")
    ax.set_title(f"Per-class F1  (threshold={args.threshold}, {total} samples)", color=FG)
    ax.legend(facecolor="#1e1e2e", edgecolor=FG)
    ax.grid(axis="y", zorder=0)

    plt.tight_layout()
    out = plots_dir / "cmp_03_f1_per_class.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 4: Delta HarmonicCNN vs CNN baseline ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor(BG)
    fig.suptitle("ΔF1 vs CNN baseline  (positive = improvement)", color=FG, fontsize=13)

    for ax, m, col in zip(axes, ["HarmonicCNN", "AttackSustain"], [GREEN, YELLOW]):
        deltas = [results[m]["f1_per_class"][i] - results["CNN"]["f1_per_class"][i]
                  for i in range(len(CLASSES))]
        bar_colors = [col if d >= 0 else "#555555" for d in deltas]
        ax.bar(CLASSES, deltas, color=bar_colors, zorder=3)
        ax.axhline(0, color=FG, lw=0.8)
        for i, d in enumerate(deltas):
            ax.text(i, d + (0.005 if d >= 0 else -0.013),
                    f"{d:+.2f}", ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=8, color=FG)
        macro_d = results[m]["f1_macro"] - results["CNN"]["f1_macro"]
        ax.set_title(f"{labels[m]}  (macro Δ={macro_d:+.3f})", color=FG)
        ax.set_ylabel("ΔF1")
        ax.grid(axis="y", zorder=0)

    plt.tight_layout()
    out = plots_dir / "cmp_04_delta.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    print(f"\n  All plots saved to: {plots_dir}/")


if __name__ == "__main__":
    main()
