"""
Ensemble evaluation on the IRMAS polyphonic test set.

Each checkpoint is run once with sliding-window S2 inference at its own window
length (1 s for Han-style / *_1sec models, 3 s otherwise); the per-clip
normalised scores are cached in checkpoints/test_scores/<ckpt>.npy. Ensembles
average those score vectors, renormalise by the max and apply the threshold.

Usage:
    python src/ensemble_test.py --members Han2017CNN_best DeepExactOctaveCNN_CosineLR_rec_best
    python src/ensemble_test.py --members A_best B_best --threshold 0.4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import CLASSES, _scan_testing_dir
from evaluate_test_sliding import MODEL_MAP, TEST_DIR, CKPT_DIR, full_mel, get_device

SCORE_DIR = CKPT_DIR / "test_scores"

_MEL_MEMO: dict = {}


def test_mel(wav: Path) -> np.ndarray:
    """Full-length log-mel of a test clip, computed once per process.

    Decoding + resampling + STFT of 2874 clips on the CPU dominated test
    evaluation (~45 s per checkpoint); with this memo only the first checkpoint
    of a run pays it. Held in RAM only (~1 GB), released when the process exits.
    """
    key = str(wav)
    if key not in _MEL_MEMO:
        _MEL_MEMO[key] = full_mel(wav)
    return _MEL_MEMO[key]


def window_frames(ckpt_stem: str, cls_name: str) -> int:
    one_sec = "1sec" in ckpt_stem or (cls_name.startswith(("Han", "HarmonicHan"))
                                      and "3sec" not in ckpt_stem)
    return 43 if one_sec else 130


def clip_scores(ckpt_stem: str, wav_paths, device) -> np.ndarray:
    """[N_clips, N_classes] S2 scores (sum of window sigmoids / max), cached."""
    cache = SCORE_DIR / f"{ckpt_stem}.npy"
    if cache.exists():
        return np.load(cache)

    ckpt  = torch.load(CKPT_DIR / f"{ckpt_stem}.pt", map_location="cpu", weights_only=False)
    cls   = ckpt["model_cls"]
    model = MODEL_MAP[cls]().to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    win = window_frames(ckpt_stem, cls)
    hop = win // 2

    out = np.zeros((len(wav_paths), len(CLASSES)), dtype=np.float32)
    with torch.no_grad():
        for i, wav in enumerate(tqdm(wav_paths, desc=f"{ckpt_stem} (win {win})", leave=False)):
            mel = test_mel(wav)
            T_f = mel.shape[1]
            if T_f < win:
                mel = np.pad(mel, ((0, 0), (0, win - T_f)), constant_values=np.log(1e-9))
                T_f = win
            starts = range(0, T_f - win + 1, hop)
            x = np.stack([mel[:, s:s + win] for s in starts])[:, None]   # [W, 1, F, win]
            s = torch.sigmoid(model(torch.from_numpy(x).float().to(device))).sum(0).cpu().numpy()
            out[i] = s / s.max() if s.max() > 0 else s

    SCORE_DIR.mkdir(exist_ok=True)
    np.save(cache, out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members",   nargs="+", required=True,
                        help="checkpoint stems (file names without .pt)")
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.3, 0.4, 0.5])
    parser.add_argument("--test-dir",  type=Path, default=TEST_DIR)
    args = parser.parse_args()

    device = get_device()
    paths, labels = _scan_testing_dir(args.test_dir)
    y = np.vstack(labels).astype(int)

    scores = {m: clip_scores(m, paths, device) for m in args.members}

    def report(name, s):
        cells = []
        for thr in args.threshold:
            p = (s >= thr).astype(int)
            cells.append(f"thr {thr:.2f}: macro={f1_score(y, p, average='macro', zero_division=0):.4f}"
                         f" micro={f1_score(y, p, average='micro', zero_division=0):.4f}")
        print(f"{name:<45}  " + "  |  ".join(cells))

    for m, s in scores.items():
        report(m, s)
    if len(scores) > 1:
        ens = np.mean(list(scores.values()), axis=0)
        ens = ens / ens.max(axis=1, keepdims=True)
        report("ENSEMBLE(" + " + ".join(args.members) + ")", ens)


if __name__ == "__main__":
    main()
