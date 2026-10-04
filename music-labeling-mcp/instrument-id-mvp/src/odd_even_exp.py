"""
Odd/Even harmonic branch experiment.

Compares HarmonicCNN (fixed octave shifts) vs OddEvenHarmonicCNN (dual branch
that separately processes odd harmonics f,3f and even harmonics 2f,4f).

Usage:
    python src/odd_even_exp.py
    python src/odd_even_exp.py --epochs 30 --seeds 5
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
from model import HarmonicCNN, OddEvenHarmonicCNN

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "data" / "IRMAS" / "training"


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


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


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


def train_seeds(name, model_cls, train_paths, train_labels, val_paths, val_labels,
                args, device):
    print(f"\n=== {name} ({args.seeds} seeds × {args.epochs} epochs) ===")
    n_params = sum(p.numel() for p in model_cls().parameters() if p.requires_grad)
    print(f"    params: {n_params:,}")

    val_ds     = IRMASDataset(val_paths, val_labels)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    scores, per_class_all = [], []

    for seed in range(args.seeds):
        set_seed(seed)

        train_ds     = IRMASDataset(train_paths, train_labels)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)

        model     = model_cls().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            tl = run_epoch(model, train_loader, criterion, optimizer, device)
            vl = run_epoch(model, val_loader,   criterion, None,      device)
            if epoch % 5 == 0:
                print(f"  [{name} seed {seed}] ep {epoch:>2}/{args.epochs}"
                      f"  train={tl:.4f}  val={vl:.4f}")

        f1_macro, f1_per = evaluate(model, val_loader, device)
        scores.append(f1_macro)
        per_class_all.append(f1_per)
        print(f"  [{name} seed {seed}] saved — F1-macro: {f1_macro:.4f}")

    return scores, np.mean(per_class_all, axis=0)


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

    harm_scores,   harm_per   = train_seeds("HarmonicCNN",
                                            HarmonicCNN,
                                            train_paths, train_labels,
                                            val_paths, val_labels, args, device)
    oddeven_scores, oddeven_per = train_seeds("OddEvenHarmonicCNN",
                                              OddEvenHarmonicCNN,
                                              train_paths, train_labels,
                                              val_paths, val_labels, args, device)

    print("\n─── Final comparison ───")
    for name, scores in [("HarmonicCNN        ", harm_scores),
                         ("OddEvenHarmonicCNN ", oddeven_scores)]:
        mean = np.mean(scores)
        std  = np.std(scores)
        strs = [f"{s:.4f}" for s in scores]
        print(f"{name}  {mean:.4f} ± {std:.4f}  {strs}")

    print(f"\nPer-class F1 (mean over {args.seeds} seeds):")
    print(f"  {'Class':<6}  {'HarmonicCNN':>12}  {'OddEven':>12}  {'Δ':>8}")
    print(f"  {'-'*44}")
    for i, cls in enumerate(CLASSES):
        h = harm_per[i]
        o = oddeven_per[i]
        delta = o - h
        sign = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else " ")
        print(f"  {cls:<6}  {h:>12.3f}  {o:>12.3f}  {sign}{abs(delta):.3f}")


if __name__ == "__main__":
    main()
