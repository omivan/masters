"""
PyTorch Dataset for IRMAS instrument classification.

Training data: one-hot labels derived from the folder name.
Testing data:  multi-hot labels read from companion .txt files.

Folder / label mapping (IRMAS official):
    cel  → cello
    cla  → clarinet
    flu  → flute
    gac  → acoustic guitar
    gel  → electric guitar
    org  → organ
    pia  → piano
    sax  → saxophone
    tru  → trumpet
    vio  → violin
    voi  → human voice
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from features import compute_mel_spectrogram, compute_cqt_spectrogram, compute_hpss_spectrogram

CLASSES = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)


def _scan_training_dir(train_root: Path) -> Tuple[List[Path], List[np.ndarray]]:
    """Return (wav_paths, one_hot_labels) from an IRMAS training directory."""
    wav_paths, labels = [], []
    for class_dir in sorted(train_root.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in CLASS_TO_IDX:
            continue
        idx = CLASS_TO_IDX[class_dir.name]
        for wav in sorted(class_dir.glob("*.wav")):
            label = np.zeros(NUM_CLASSES, dtype=np.float32)
            label[idx] = 1.0
            wav_paths.append(wav)
            labels.append(label)
    return wav_paths, labels


def _scan_testing_dir(test_root: Path) -> Tuple[List[Path], List[np.ndarray]]:
    """Return (wav_paths, multi_hot_labels) from an IRMAS testing directory.

    Each wav has a companion .txt file listing active instrument abbreviations.
    """
    wav_paths, labels = [], []
    for wav in sorted(test_root.rglob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        for line in txt.read_text().splitlines():
            token = line.strip().lower()
            if token in CLASS_TO_IDX:
                label[CLASS_TO_IDX[token]] = 1.0
        wav_paths.append(wav)
        labels.append(label)
    return wav_paths, labels


def train_val_split(
    wav_paths: List[Path],
    labels: List[np.ndarray],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Path], List[np.ndarray], List[Path], List[np.ndarray]]:
    """Stratified 80/20 split by dominant class label."""
    random.seed(seed)
    # Group indices by class
    class_indices: dict[int, list[int]] = {i: [] for i in range(NUM_CLASSES)}
    for idx, label in enumerate(labels):
        dominant = int(np.argmax(label))
        class_indices[dominant].append(idx)

    train_idx, val_idx = [], []
    for cls_idxs in class_indices.values():
        random.shuffle(cls_idxs)
        split = max(1, int(len(cls_idxs) * (1 - val_ratio)))
        train_idx.extend(cls_idxs[:split])
        val_idx.extend(cls_idxs[split:])

    return (
        [wav_paths[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [wav_paths[i] for i in val_idx],
        [labels[i] for i in val_idx],
    )


_LOG_ZERO = float(np.log(1e-9))  # -20.72 — silence sentinel


class IRMASDataset(Dataset):
    def __init__(
        self,
        wav_paths: List[Path],
        labels: List[np.ndarray],
        pitch_shift_bins: int = 0,
        spec_augment: bool = False,
        spec_freq_mask: int = 20,   # max mel bins to zero out per mask
        spec_time_mask: int = 30,   # max time frames to zero out per mask
    ):
        self.wav_paths = wav_paths
        self.labels = labels
        self.pitch_shift_bins = pitch_shift_bins
        self.spec_augment = spec_augment
        self.spec_freq_mask = spec_freq_mask
        self.spec_time_mask = spec_time_mask

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mel = compute_mel_spectrogram(self.wav_paths[idx])  # [128, 130]
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # [1, 128, 130]

        if self.pitch_shift_bins > 0:
            shift = random.randint(-self.pitch_shift_bins, self.pitch_shift_bins)
            if shift != 0:
                x = torch.roll(x, shift, dims=1)
                if shift > 0:
                    x[:, :shift, :] = _LOG_ZERO
                else:
                    x[:, shift:, :] = _LOG_ZERO

        if self.spec_augment:
            _, F, T = x.shape
            # Frequency mask
            f = random.randint(0, self.spec_freq_mask)
            f0 = random.randint(0, F - f)
            x[:, f0:f0 + f, :] = _LOG_ZERO
            # Time mask
            t = random.randint(0, self.spec_time_mask)
            t0 = random.randint(0, T - t)
            x[:, :, t0:t0 + t] = _LOG_ZERO

        y = torch.tensor(self.labels[idx], dtype=torch.float32)   # [NUM_CLASSES]
        return x, y


class HPSSIRMASDataset(Dataset):
    """IRMAS dataset using HPSS harmonic log-mel as input [1, 128, 130].

    Only the harmonic component is returned — percussive noise is filtered out.
    """

    def __init__(self, wav_paths: List[Path], labels: List[np.ndarray]):
        self.wav_paths = wav_paths
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        harm, _ = compute_hpss_spectrogram(self.wav_paths[idx])   # [128, 130]
        x = torch.tensor(harm, dtype=torch.float32).unsqueeze(0)  # [1, 128, 130]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class HPSS2ChIRMASDataset(Dataset):
    """IRMAS dataset with both HPSS components stacked as [2, 128, 130].

    ch 0 = harmonic log-mel  (tonal timbre)
    ch 1 = percussive log-mel (attack / onset)
    """

    def __init__(self, wav_paths: List[Path], labels: List[np.ndarray]):
        self.wav_paths = wav_paths
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        harm, perc = compute_hpss_spectrogram(self.wav_paths[idx])  # each [128, 130]
        x = torch.tensor(np.stack([harm, perc]), dtype=torch.float32)  # [2, 128, 130]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class CQTIRMASDataset(Dataset):
    """Same as IRMASDataset but uses log-CQT features instead of log-mel."""

    def __init__(self, wav_paths: List[Path], labels: List[np.ndarray]):
        self.wav_paths = wav_paths
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cqt = compute_cqt_spectrogram(self.wav_paths[idx])   # [168, 130]
        x   = torch.tensor(cqt, dtype=torch.float32).unsqueeze(0)  # [1, 168, 130]
        y   = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def load_training_dataset(train_root: Path) -> IRMASDataset:
    wav_paths, labels = _scan_training_dir(train_root)
    if not wav_paths:
        raise RuntimeError(f"No training wav files found under {train_root}")
    return IRMASDataset(wav_paths, labels)


def load_testing_dataset(test_root: Path) -> IRMASDataset:
    wav_paths, labels = _scan_testing_dir(test_root)
    if not wav_paths:
        raise RuntimeError(f"No testing wav files (with .txt) found under {test_root}")
    return IRMASDataset(wav_paths, labels)
