"""
Train the CNN+BiLSTM instrument classifier on IRMAS training data.

Usage:
    python src/train.py
    python src/train.py --epochs 10 --batch-size 16 --max-per-class 50
    python src/train.py --data-dir data/IRMAS/training/IRMAS-TrainingData
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow imports from src/ when called as a script
sys.path.insert(0, str(Path(__file__).parent))

from dataset import (
    CLASSES,
    load_training_dataset,
    train_val_split,
    IRMASDataset,
)
from model import InstrumentCNNBiLSTM

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "data" / "IRMAS" / "training"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "model.pt"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_training_root(base: Path) -> Path:
    """Resolve the actual folder that contains instrument sub-directories."""
    instrument_names = set(CLASSES)
    if any((base / c).is_dir() for c in instrument_names):
        return base
    # One level of nesting (IRMAS zips often add an extra folder)
    for child in sorted(base.iterdir()):
        if child.is_dir() and any((child / c).is_dir() for c in instrument_names):
            return child
    raise RuntimeError(
        f"Cannot find instrument folders under {base}. "
        "Run download_data.py first."
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> float:
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


def main():
    parser = argparse.ArgumentParser(description="Train CNN+BiLSTM on IRMAS")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_TRAIN_ROOT,
        help="Path to IRMAS training root (contains instrument sub-folders)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Where to save model.pt",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 = main process, safe on all platforms)",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_root = find_training_root(args.data_dir)
    print(f"Loading dataset from {train_root} …")
    full_ds = load_training_dataset(train_root)
    print(f"Total samples: {len(full_ds)}")

    train_paths, train_labels, val_paths, val_labels = train_val_split(
        full_ds.wav_paths, full_ds.labels
    )
    train_ds = IRMASDataset(train_paths, train_labels)
    val_ds = IRMASDataset(val_paths, val_labels)
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # ── Model, loss, optimiser ───────────────────────────────────────────────
    model = InstrumentCNNBiLSTM().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epoch(s) …\n")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, f"Epoch {epoch} train")
        val_loss = run_epoch(model, val_loader, criterion, None, device, f"Epoch {epoch} val  ")
        print(f"Epoch {epoch:>3}/{args.epochs}  |  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "epoch": args.epochs,
        },
        args.checkpoint,
    )
    print(f"\nCheckpoint saved → {args.checkpoint}")


if __name__ == "__main__":
    main()
