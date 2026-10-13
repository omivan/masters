"""
Train DeepHarmonicCNN on 100% of IRMAS training data (no val split).
Hyperparams fixed from val experiments: 30 epochs, CosineLR, threshold=0.3.
"""

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

from dataset import CLASSES, IRMASDataset, load_training_dataset
from evaluate_test_sliding import predict_sliding, MODEL_MAP
from dataset import _scan_testing_dir
from model import DeepHarmonicCNN

PROJECT_ROOT = Path(__file__).parent.parent
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
TEST_DIR     = PROJECT_ROOT / "data" / "IRMAS" / "testing"

EPOCHS    = 30
LR        = 1e-3
BATCH     = 32
THRESHOLD = 0.3
SEEDS     = 3


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Device : {device}")

    base = PROJECT_ROOT / "data" / "IRMAS" / "training"
    train_root = base
    for child in sorted(base.iterdir()):
        if child.is_dir() and any((child / c).is_dir() for c in CLASSES):
            train_root = child
            break
    full_ds = load_training_dataset(train_root)
    print(f"Full training set: {len(full_ds.wav_paths)} clips (100%)")

    CKPT_DIR.mkdir(exist_ok=True)
    test_paths, test_labels = _scan_testing_dir(TEST_DIR)
    print(f"Test set: {len(test_paths)} clips\n")

    best_test_macro = -1.0
    best_ckpt_path  = None

    for seed in range(SEEDS):
        set_seed(seed)
        loader = DataLoader(
            IRMASDataset(full_ds.wav_paths, full_ds.labels),
            batch_size=BATCH, shuffle=True, num_workers=0,
        )
        model     = DeepHarmonicCNN().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-5)

        bar = tqdm(range(1, EPOCHS + 1), desc=f"seed {seed}")
        for _ in bar:
            model.train()
            total = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                loss = criterion(model(x), y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += loss.item() * x.size(0)
            scheduler.step()
            bar.set_postfix(loss=f"{total/len(loader.dataset):.4f}")

        # Test eval
        preds, targets = predict_sliding(model, test_paths, test_labels, device, THRESHOLD)
        macro = f1_score(targets, preds, average="macro", zero_division=0)
        micro = f1_score(targets, preds, average="micro", zero_division=0)
        print(f"[seed {seed}] test macro={macro:.4f}  micro={micro:.4f}")

        ckpt_path = CKPT_DIR / f"DeepHarmonicCNN_full_seed{seed}_best.pt"
        torch.save({
            "model_cls":   "DeepHarmonicCNN",
            "config_name": "DeepHarmonicCNN+Full",
            "seed":        seed,
            "test_macro":  macro,
            "epochs":      EPOCHS,
            "state_dict":  {k: v.cpu().clone() for k, v in model.state_dict().items()},
        }, ckpt_path)

        if macro > best_test_macro:
            best_test_macro = macro
            best_ckpt_path  = ckpt_path

    print(f"\nBest test macro: {best_test_macro:.4f} -> {best_ckpt_path.name}")
    print("Reference (80% train): DeepHarmonicCNN+CosineLR  macro=0.4905")


if __name__ == "__main__":
    main()
