"""
Mel-spectrogram extraction with on-disk caching.

Uses torchaudio (no lzma dependency) instead of librosa.
Each .wav gets a companion .npy file (same path, different extension).
Shape of cached array: [128, 130]  (n_mels × time_frames)
"""

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from pathlib import Path

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
TIME_FRAMES = 130  # ~3 s at 22050 Hz / 512 hop

_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)
_resample_cache: dict[int, T.Resample] = {}


def compute_mel_spectrogram(wav_path: str | Path, use_cache: bool = True) -> np.ndarray:
    """Return log-mel spectrogram of shape [N_MELS, TIME_FRAMES].

    Loads from .npy cache if available; otherwise computes and saves it.
    """
    wav_path = Path(wav_path)
    cache_path = wav_path.with_suffix(".npy")

    if use_cache and cache_path.exists():
        return np.load(cache_path)

    data, sr = sf.read(wav_path, always_2d=True)   # [T, C]
    # Mix down to mono, convert to [1, T] tensor
    waveform = torch.from_numpy(data.mean(axis=1, keepdims=False)).float().unsqueeze(0)

    # Resample if needed
    if sr != SAMPLE_RATE:
        if sr not in _resample_cache:
            _resample_cache[sr] = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = _resample_cache[sr](waveform)

    # Amplitude normalisation
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    mel = _mel_transform(waveform)          # [1, N_MELS, T]
    log_mel = torch.log(mel + 1e-9)        # [1, N_MELS, T]
    log_mel = log_mel.squeeze(0).numpy()   # [N_MELS, T]

    # Crop or pad along time axis to exactly TIME_FRAMES
    t = log_mel.shape[1]
    if t >= TIME_FRAMES:
        log_mel = log_mel[:, :TIME_FRAMES]
    else:
        pad_width = TIME_FRAMES - t
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=np.log(1e-9))

    if use_cache:
        np.save(cache_path, log_mel)

    return log_mel  # [128, 130]


def compute_mel_windows(wav_path: str | Path, hop_seconds: float = 1.5) -> list[np.ndarray]:
    """Split audio into overlapping 3-second windows, return list of [N_MELS, TIME_FRAMES].

    hop_seconds controls the step between windows (default: 1.5s = 50% overlap).
    Always returns at least one window (even for short files).
    """
    wav_path = Path(wav_path)
    data, sr = sf.read(wav_path, always_2d=True)
    waveform = torch.from_numpy(data.mean(axis=1)).float().unsqueeze(0)

    if sr != SAMPLE_RATE:
        if sr not in _resample_cache:
            _resample_cache[sr] = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = _resample_cache[sr](waveform)

    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    window_samples = TIME_FRAMES * HOP_LENGTH          # ~66560 samples = 3s
    hop_samples    = int(hop_seconds * SAMPLE_RATE)    # ~33075 samples = 1.5s
    total_samples  = waveform.shape[1]

    # Collect start positions
    starts = list(range(0, max(1, total_samples - window_samples + 1), hop_samples))
    if not starts:
        starts = [0]

    windows = []
    for start in starts:
        chunk = waveform[:, start: start + window_samples]
        # Pad if last chunk is short
        if chunk.shape[1] < window_samples:
            chunk = torch.nn.functional.pad(chunk, (0, window_samples - chunk.shape[1]))
        mel     = _mel_transform(chunk)               # [1, N_MELS, T]
        log_mel = torch.log(mel + 1e-9).squeeze(0).numpy()  # [N_MELS, T]
        log_mel = log_mel[:, :TIME_FRAMES]
        windows.append(log_mel)

    return windows  # list of [128, 130]
