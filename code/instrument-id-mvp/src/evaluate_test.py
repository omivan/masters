"""
Evaluate saved checkpoints on the IRMAS polyphonic test set.

Usage:
    python src/evaluate_test.py
    python src/evaluate_test.py --threshold 0.3
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

from dataset import CLASSES, load_testing_dataset
from model import HarmonicCNN, DeltaHarmonicCNN, Han2017CNN

PROJECT_ROOT = Path(__file__).parent.parent
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
TEST_DIR     = PROJECT_ROOT / "data" / "IRMAS" / "testing"

MODEL_MAP = {
    "HarmonicCNN":    HarmonicCNN,
    "DeltaHarmonicCNN": DeltaHarmonicCNN,
    "Han2017CNN":     Han2017CNN,
}


def get_device():
    if torch.cuda.is_available():    return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def evaluate_ckpt(ckpt_path, loader, device, threshold):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cls_name = ckpt["model_cls"]
    model = MODEL_MAP[cls_name]().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=ckpt["config_name"], leave=False):
            probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
            preds.append((probs >= threshold).astype(int))
            targets.append(y.numpy().astype(int))

    p, t = np.vstack(preds), np.vstack(targets)
    return (
        f1_score(t, p, average="macro", zero_division=0),
        f1_score(t, p, average="micro", zero_division=0),
        f1_score(t, p, average=None,    zero_division=0),
        ckpt,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--test-dir",  type=Path,  default=TEST_DIR)
    args = parser.parse_args()

    device = get_device()
    print(f"Device   : {device}")
    print(f"Test dir : {args.test_dir}")
    print(f"Threshold: {args.threshold}\n")

    ds     = load_testing_dataset(args.test_dir)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test clips: {len(ds)}\n")

    ckpts = sorted(CKPT_DIR.glob("*_best.pt"))
    if not ckpts:
        print(f"No checkpoints in {CKPT_DIR}")
        return

    results = {}
    for ckpt_path in ckpts:
        # Han2017CNN uses 43-frame windows; skip here (evaluated via han_exp.py sliding window)
        ckpt_peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt_peek.get("model_cls") == "Han2017CNN":
            print(f"Skipping {ckpt_path.name} (Han2017CNN — use han_exp.py for sliding-window eval)")
            continue
        macro, micro, per_cls, meta = evaluate_ckpt(ckpt_path, loader, device, args.threshold)
        name = meta["config_name"]
        results[name] = dict(macro=macro, micro=micro, per_cls=per_cls,
                             val_f1=meta["f1_macro"], seed=meta["seed"])
        print(f"{name}")
        print(f"  F1-macro : {macro:.4f}  |  F1-micro : {micro:.4f}"
              f"  (val F1 was {meta['f1_macro']:.4f}, seed {meta['seed']})")

    # Per-class table
    names = list(results.keys())
    print(f"\nPer-class F1 on test set (threshold={args.threshold}):")
    header = f"  {'Class':<6}" + "".join(f"  {n[:16]:>16}" for n in names)
    print(header)
    print("  " + "-" * (8 + 18 * len(names)))
    for i, cls in enumerate(CLASSES):
        vals = [results[n]["per_cls"][i] for n in names]
        best = max(vals)
        row  = f"  {cls:<6}"
        for v in vals:
            marker = "*" if v == best and best > min(vals) + 0.005 else " "
            row   += f"  {v:>15.3f}{marker}"
        print(row)

    print("\n─── Summary ───")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["macro"]):
        print(f"  {name:<35}  macro={r['macro']:.4f}  micro={r['micro']:.4f}")

    print("\nPublished SOTA (test set):")
    print("  Han et al. 2017 (CNN)       macro=0.503  micro=0.602")
    print("  DWT CNN 2024                macro=0.631  micro=0.695")


if __name__ == "__main__":
    main()
