"""
Choose the decision threshold on a *polyphonic* validation set instead of
borrowing 0.5 from Han et al. or tuning on the test set.

The validation split (recording-grouped, no leakage) is monophonic, so
thresholds tuned on it favour predicting a single instrument. Here each val
clip is mixed with 0–2 other val clips from different recordings (linear-power
sum, ±6 dB, same as PolyMix) with a fixed RNG seed; the label is the union.
Every seed of a model is scored with the same S2 rule as on the test set
(sliding windows of the model's own length → sum of sigmoids → / max), the
threshold maximising mean val macro F1 is selected, and only then the cached
test scores are read at that threshold.

Usage:
    python src/tune_threshold_polyval.py --models DeepMelPonsCNN_CosineLR_PolyMix_rec
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))

from dataset import _scan_testing_dir, _scan_training_dir, recording_id, train_val_split
from ensemble_test import SCORE_DIR, clip_scores, window_frames
from evaluate_test_sliding import CKPT_DIR, MODEL_MAP, TEST_DIR, get_device
from features import compute_mel_spectrogram

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_ROOT   = PROJECT_ROOT / "data" / "IRMAS" / "training" / "IRMAS-TrainingData"
THRESHOLDS   = np.round(np.arange(0.30, 0.81, 0.05), 2)

# Number of sources per mixture: equal shares of solo, duo and trio — an
# uninformative choice that does not look at the test-set label statistics.
SOURCE_PROBS = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}
GAIN_DB      = 6.0


def build_poly_val(seed: int = 123):
    tr_p, tr_l = _scan_training_dir(TRAIN_ROOT)
    _, _, val_p, val_l = train_val_split(tr_p, tr_l, seed=42)
    mels = np.stack([compute_mel_spectrogram(p) for p in val_p])          # [N, 128, 130]
    labs = np.stack(val_l).astype(int)
    recs = np.array([recording_id(p) for p in val_p])

    rng = np.random.default_rng(seed)
    ks  = rng.choice(list(SOURCE_PROBS), size=len(val_p), p=list(SOURCE_PROBS.values()))
    X, Y = mels.copy(), labs.copy()
    for i, k in enumerate(ks):
        used = {recs[i]}
        for _ in range(k - 1):
            j = rng.integers(len(val_p))
            while recs[j] in used:
                j = rng.integers(len(val_p))
            used.add(recs[j])
            gain = rng.uniform(-GAIN_DB, GAIN_DB) * np.log(10) / 10
            X[i] = np.logaddexp(X[i], mels[j] + gain)
            Y[i] = np.maximum(Y[i], labs[j])
    return X, Y, ks


def s2_scores(stem: str, X: np.ndarray, device) -> np.ndarray:
    ckpt  = torch.load(CKPT_DIR / f"{stem}.pt", map_location="cpu", weights_only=False)
    model = MODEL_MAP[ckpt["model_cls"]]().to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    win = window_frames(stem, ckpt["model_cls"])
    starts = range(0, X.shape[2] - win + 1, max(1, win // 2))
    out = np.zeros((len(X), 11), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(X), 256):
            xb = torch.from_numpy(X[b:b + 256]).float().unsqueeze(1).to(device)
            s = sum(torch.sigmoid(model(xb[..., t:t + win])) for t in starts).cpu().numpy()
            out[b:b + 256] = s / s.max(axis=1, keepdims=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="checkpoint stems without _seed<n> (e.g. DeepMelPonsCNN_CosineLR_PolyMix_rec)")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    X, Y, ks = build_poly_val()
    print(f"Polyphonic val: {len(X)} mixtures  (solo {np.mean(ks == 1):.0%}, duo {np.mean(ks == 2):.0%},"
          f" trio {np.mean(ks == 3):.0%}), {Y.sum(1).mean():.2f} labels/clip")

    test_paths, test_labels = _scan_testing_dir(TEST_DIR)
    y_test = np.vstack(test_labels).astype(int)

    for model in args.models:
        stems = [f"{model}_seed{s}" for s in range(args.seeds)]
        val_f1 = np.array([[f1_score(Y, (s2_scores(st, X, device) >= t).astype(int),
                                     average="macro", zero_division=0) for t in THRESHOLDS]
                           for st in stems])                                    # [seeds, thr]
        best_t = THRESHOLDS[val_f1.mean(0).argmax()]
        test_f1 = [f1_score(y_test, (clip_scores(st, test_paths, device) >= best_t).astype(int),
                            average="macro", zero_division=0) for st in stems]
        test_mi = [f1_score(y_test, (clip_scores(st, test_paths, device) >= best_t).astype(int),
                            average="micro", zero_division=0) for st in stems]

        print(f"\n{model}")
        print("  poly-val macro F1 by threshold: "
              + "  ".join(f"{t:.2f}:{v:.3f}" for t, v in zip(THRESHOLDS, val_f1.mean(0))))
        print(f"  → selected threshold {best_t:.2f}  |  test macro {np.mean(test_f1):.4f} ± {np.std(test_f1):.4f}"
              f"  [{', '.join(f'{v:.3f}' for v in test_f1)}]   micro {np.mean(test_mi):.4f} ± {np.std(test_mi):.4f}")


if __name__ == "__main__":
    main()
