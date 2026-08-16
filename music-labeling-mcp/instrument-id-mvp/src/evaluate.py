"""
Evaluate the trained model on a held-out validation split or IRMAS testing data.

Usage:
    # Evaluate on the 20% val split of training data (default):
    python src/evaluate.py

    # Evaluate on downloaded IRMAS testing data (multi-label):
    python src/evaluate.py --test-dir data/IRMAS/testing
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import (
    CLASSES,
    load_training_dataset,
    load_testing_dataset,
    train_val_split,
    IRMASDataset,
)
from model import InstrumentCNNBiLSTM

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "model.pt"
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


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Inference"):
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_preds.append(preds)
            all_targets.append(y.numpy().astype(int))
    return np.vstack(all_preds), np.vstack(all_targets)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN+BiLSTM on IRMAS")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_TRAIN_ROOT,
        help="IRMAS training root (used when --test-dir is not provided)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="IRMAS testing directory (multi-label .wav+.txt pairs)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run train.py first."
        )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = InstrumentCNNBiLSTM().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    if args.test_dir is not None:
        print(f"Using IRMAS testing data: {args.test_dir}")
        ds = load_testing_dataset(args.test_dir)
    else:
        train_root = find_training_root(args.data_dir)
        full_ds = load_training_dataset(train_root)
        _, _, val_paths, val_labels = train_val_split(full_ds.wav_paths, full_ds.labels)
        ds = IRMASDataset(val_paths, val_labels)
        print(f"Using validation split ({len(ds)} samples)")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    preds, targets = predict(model, loader, device, threshold=args.threshold)

    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    f1_micro = f1_score(targets, preds, average="micro", zero_division=0)

    print(f"\nThreshold : {args.threshold}")
    print(f"F1-macro  : {f1_macro:.4f}")
    print(f"F1-micro  : {f1_micro:.4f}")

    print("\nPer-class report:")
    print(classification_report(targets, preds, target_names=CLASSES, zero_division=0))


if __name__ == "__main__":
    main()
