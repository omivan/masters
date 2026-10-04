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

from dataset import CLASSES, IRMASDataset, load_training_dataset, train_val_split
from model import HarmonicCNN, DeltaHarmonicCNN

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "data" / "IRMAS" / "training"

MIXUP_ALPHA = 0.4   # Beta(α, α) parameter for mixup


def get_device() -> torch.device:
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


# ── Training loops ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    with torch.set_grad_enabled(True):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
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


# ── Generic seed runner ────────────────────────────────────────────────────────

def train_seeds(name, model_cls, train_paths, train_labels, val_paths, val_labels,
                args, device,
                pitch_shift_bins: int = 0,
                spec_augment: bool = False,
                use_mixup: bool = False,
                cosine_lr: bool = False):

    print(f"\n=== {name} ({args.seeds} seeds × {args.epochs} epochs) ===")
    tags = []
    if pitch_shift_bins: tags.append(f"pitch±{pitch_shift_bins}")
    if spec_augment:     tags.append("SpecAugment")
    if use_mixup:        tags.append(f"Mixup(α={MIXUP_ALPHA})")
    if cosine_lr:        tags.append("CosineLR")
    if tags:
        print(f"    [{', '.join(tags)}]")
    n_params = sum(p.numel() for p in model_cls().parameters() if p.requires_grad)
    print(f"    params: {n_params:,}")

    val_ds     = IRMASDataset(val_paths, val_labels)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    scores, per_class_all = [], []

    for seed in range(args.seeds):
        set_seed(seed)

        train_ds = IRMASDataset(
            train_paths, train_labels,
            pitch_shift_bins=pitch_shift_bins,
            spec_augment=spec_augment,
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

        for epoch in range(1, args.epochs + 1):
            if use_mixup:
                tl = run_epoch_mixup(model, train_loader, criterion, optimizer, device)
            else:
                tl = run_epoch(model, train_loader, criterion, optimizer, device)
            vl = run_val_epoch(model, val_loader, criterion, device)
            if scheduler:
                scheduler.step()
            if epoch % 5 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  [{name} seed {seed}] ep {epoch:>2}/{args.epochs}"
                      f"  train={tl:.4f}  val={vl:.4f}  lr={lr_now:.2e}")

        f1_macro, f1_per = evaluate(model, val_loader, device)
        scores.append(f1_macro)
        per_class_all.append(f1_per)
        print(f"  [{name} seed {seed}] saved — F1-macro: {f1_macro:.4f}")

    return scores, np.mean(per_class_all, axis=0)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seeds",      type=int,   default=5)
    parser.add_argument("--data-dir",   type=Path,  default=DEFAULT_TRAIN_ROOT)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    train_root = find_training_root(args.data_dir)
    full_ds    = load_training_dataset(train_root)
    train_paths, train_labels, val_paths, val_labels = train_val_split(
        full_ds.wav_paths, full_ds.labels, seed=42
    )
    print(f"Train: {len(train_paths)}  Val: {len(val_paths)}")

    configs = [
        dict(name="HarmonicCNN",
             model_cls=HarmonicCNN),
        dict(name="HarmonicCNN+Mixup",
             model_cls=HarmonicCNN,
             use_mixup=True),
        dict(name="HarmonicCNN+pitch2+SpecAug",
             model_cls=HarmonicCNN,
             pitch_shift_bins=2, spec_augment=True),
        dict(name="HarmonicCNN+CosineLR",
             model_cls=HarmonicCNN,
             cosine_lr=True),
        dict(name="DeltaHarmonicCNN+CosineLR",
             model_cls=DeltaHarmonicCNN,
             cosine_lr=True),
    ]

    all_results = {}
    for cfg in configs:
        name = cfg.pop("name")
        model_cls = cfg.pop("model_cls")
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


if __name__ == "__main__":
    main()
