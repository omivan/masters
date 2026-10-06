"""Standalone sliding-window evaluation — no need to rerun the full notebook."""
import sys, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             hamming_loss, accuracy_score, roc_auc_score)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import InstrumentCNN, HarmonicCNN
from dataset import load_testing_dataset
from features import compute_mel_windows

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS    = 30
N_SEEDS   = 5
THRESHOLD = 0.3
CKPT_DIR  = PROJECT_ROOT / "checkpoints"
TEST_ROOT = PROJECT_ROOT / "data" / "IRMAS" / "testing"

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Device: {device}")

CLASSES = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]


def ckpt_path(model_name, seed):
    return CKPT_DIR / f"{model_name}_seed{seed}_ep{EPOCHS}.pt"


def compute_metrics(probs, preds, targets):
    return {
        "f1_macro":   f1_score(targets, preds, average="macro",  zero_division=0),
        "f1_micro":   f1_score(targets, preds, average="micro",  zero_division=0),
        "prec_macro": precision_score(targets, preds, average="macro", zero_division=0),
        "rec_macro":  recall_score(targets, preds, average="macro",  zero_division=0),
        "auc_macro":  roc_auc_score(targets, probs, average="macro"),
        "f1_per":     f1_score(targets, preds, average=None, zero_division=0),
    }


def eval_crop(model_cls, model_name, test_loader, targets_np):
    """Original: take first 3s only."""
    all_seed_probs = []
    for seed in range(N_SEEDS):
        path = ckpt_path(model_name, seed)
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = model_cls().to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        probs_list = []
        with torch.no_grad():
            for x, _ in test_loader:
                p = torch.sigmoid(model(x.to(device))).cpu().numpy()
                probs_list.append(p)
        all_seed_probs.append(np.vstack(probs_list))
    mean_probs = np.mean(all_seed_probs, axis=0)
    preds = (mean_probs >= THRESHOLD).astype(int)
    return compute_metrics(mean_probs, preds, targets_np)


def eval_windowed(model_cls, model_name, wav_paths, targets_np, hop_seconds=1.5):
    """Sliding window: average over all 3s windows per file."""
    seed_file_probs = []
    for seed in range(N_SEEDS):
        path = ckpt_path(model_name, seed)
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = model_cls().to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        file_probs = []
        with torch.no_grad():
            for wav in wav_paths:
                windows = compute_mel_windows(wav, hop_seconds=hop_seconds)
                batch = torch.tensor(
                    np.stack(windows)[:, None, :, :], dtype=torch.float32
                ).to(device)
                logits = model(batch)
                probs  = torch.sigmoid(logits).cpu().numpy()
                file_probs.append(probs.mean(axis=0))
        seed_file_probs.append(np.array(file_probs))

    mean_probs = np.mean(seed_file_probs, axis=0)
    preds = (mean_probs >= THRESHOLD).astype(int)
    return compute_metrics(mean_probs, preds, targets_np)


def main():
    if not TEST_ROOT.exists():
        print("Test set not found. Run: python src/download_data.py --include-test")
        return

    test_ds     = load_testing_dataset(TEST_ROOT)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    targets_np  = np.array(test_ds.labels)
    wav_paths   = test_ds.wav_paths
    print(f"Test files: {len(wav_paths)}")

    # Count windows per file for info
    sample_wins = compute_mel_windows(wav_paths[0])
    print(f"Example file: {wav_paths[0].name} → {len(sample_wins)} windows")

    models = [
        (InstrumentCNN, "InstrumentCNN"),
        (HarmonicCNN,   "HarmonicCNN"),
    ]

    results = {}
    for model_cls, name in models:
        print(f"\n── {name} ──────────────────────────────")
        print("  Crop (first 3s)...")
        results[f"{name}_crop"] = eval_crop(model_cls, name, test_loader, targets_np)
        print("  Sliding window (hop=1.5s)...")
        results[f"{name}_win"]  = eval_windowed(model_cls, name, wav_paths, targets_np)

    # ── Summary table ─────────────────────────────────────────────────────────
    metrics = [("F1-macro", "f1_macro"), ("F1-micro", "f1_micro"),
               ("Precision", "prec_macro"), ("Recall", "rec_macro"), ("AUC-ROC", "auc_macro")]

    print("\n" + "═" * 78)
    print(f"{'Metric':<18} {'CNN crop':>10} {'CNN win':>10} {'Δ':>7}   "
          f"{'Harm crop':>10} {'Harm win':>10} {'Δ':>7}")
    print("─" * 78)
    for label, key in metrics:
        cc = results["InstrumentCNN_crop"][key]
        cw = results["InstrumentCNN_win"][key]
        hc = results["HarmonicCNN_crop"][key]
        hw = results["HarmonicCNN_win"][key]
        print(f"{label:<18} {cc:>10.4f} {cw:>10.4f} {cw-cc:>+7.4f}   "
              f"{hc:>10.4f} {hw:>10.4f} {hw-hc:>+7.4f}")

    # ── Per-class F1 comparison ───────────────────────────────────────────────
    print("\n" + "═" * 78)
    print(f"{'Class':<8} {'CNN crop':>10} {'CNN win':>10} {'Δ':>7}   "
          f"{'Harm crop':>10} {'Harm win':>10} {'Δ':>7}")
    print("─" * 78)
    for k, cls in enumerate(CLASSES):
        cc = results["InstrumentCNN_crop"]["f1_per"][k]
        cw = results["InstrumentCNN_win"]["f1_per"][k]
        hc = results["HarmonicCNN_crop"]["f1_per"][k]
        hw = results["HarmonicCNN_win"]["f1_per"][k]
        print(f"{cls:<8} {cc:>10.4f} {cw:>10.4f} {cw-cc:>+7.4f}   "
              f"{hc:>10.4f} {hw:>10.4f} {hw-hc:>+7.4f}")


if __name__ == "__main__":
    main()
