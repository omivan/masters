"""
Feature extraction with on-disk caching.

Mel spectrogram  — uses torchaudio, cached as <file>.npy
CQT spectrogram  — uses torch.stft,  cached as <file>.cqt.npy
HPSS             — harmonic+percussive mel via median filter, cached as <file>.hpss.npy

Mel shape:  [N_MELS,     TIME_FRAMES]    = [128, 130]
CQT shape:  [CQT_N_BINS, TIME_FRAMES]    = [144, 130]
HPSS shape: [2, N_MELS,  TIME_FRAMES]    = [2, 128, 130]  (harmonic, percussive)
"""

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from pathlib import Path

SAMPLE_RATE = 22050
N_MELS      = 128
N_FFT       = 1024
HOP_LENGTH  = 512
TIME_FRAMES = 130  # ~3 s at 22050 Hz / 512 hop

# CQT parameters
# 24 bins/octave → exact integer shifts for harmonic series:
#   2f = +24 bins (1 octave), 3f = +38 bins, 4f = +48 bins (2 octaves)
# 7 octaves starting at C1 (32.7 Hz) covers full instrument range C1–C8
CQT_BINS_PER_OCT = 24
CQT_OCTAVES      = 6
CQT_N_BINS       = CQT_BINS_PER_OCT * CQT_OCTAVES  # 144  (C1=32.7 Hz → C7=2093 Hz)
CQT_FMIN         = 32.70  # C1 in Hz
# 144 → MaxPool2d×2 → 36 → AdaptiveAvgPool2d(4,4) → 9  (divisible on MPS)

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


# ── HPSS ──────────────────────────────────────────────────────────────────────
# Harmonic-Percussive Source Separation via median filtering (Fitzgerald 2010).
#
# In a mel spectrogram:
#   • Harmonic sounds (sustained notes) → horizontal stripes (stable over time)
#   • Percussive sounds (onsets, attacks) → vertical stripes (broadband, short)
#
# Median filter along time  → keeps horizontal structure  → harmonic component H
# Median filter along freq  → keeps vertical structure    → percussive component P
# Wiener soft masks: h_mask = H² / (H² + P²),  p_mask = P² / (H² + P²)
#
# Result: two clean components that separate tonal timbre from attack/rhythm.

HPSS_HARM_WIN = 31   # time frames  ≈ 0.72 s  (31 × 512 / 22050)
HPSS_PERC_WIN = 17   # mel freq bins ≈ covers typical percussive bandwidth


def compute_hpss_spectrogram(
    wav_path: str | Path,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (harmonic, percussive) log-mel spectrograms, each [N_MELS, TIME_FRAMES].

    Cached as <name>.hpss.npy with shape [2, 128, 130]:
        arr[0] = harmonic log-mel
        arr[1] = percussive log-mel
    """
    from scipy.ndimage import median_filter

    wav_path   = Path(wav_path)
    cache_path = wav_path.with_suffix(".hpss.npy")

    if use_cache and cache_path.exists():
        arr = np.load(cache_path)
        return arr[0], arr[1]

    # ── Load and normalise audio (same pipeline as mel) ───────────────────────
    data, sr = sf.read(wav_path, always_2d=True)
    waveform = torch.from_numpy(data.mean(axis=1)).float().unsqueeze(0)

    if sr != SAMPLE_RATE:
        if sr not in _resample_cache:
            _resample_cache[sr] = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = _resample_cache[sr](waveform)

    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    # ── Amplitude mel (HPSS works on amplitudes, not log) ────────────────────
    mel_amp = _mel_transform(waveform).squeeze(0).numpy()   # [128, T]
    mel_amp = np.maximum(mel_amp, 0.0)

    t = mel_amp.shape[1]
    if t >= TIME_FRAMES:
        mel_amp = mel_amp[:, :TIME_FRAMES]
    else:
        mel_amp = np.pad(mel_amp, ((0, 0), (0, TIME_FRAMES - t)), mode="constant")

    # ── Median filtering ─────────────────────────────────────────────────────
    H = median_filter(mel_amp, size=(1,             HPSS_HARM_WIN))  # horizontal
    P = median_filter(mel_amp, size=(HPSS_PERC_WIN, 1            ))  # vertical

    # ── Wiener soft masks ─────────────────────────────────────────────────────
    H2, P2 = H ** 2, P ** 2
    eps    = 1e-9
    harm   = mel_amp * (H2 / (H2 + P2 + eps))
    perc   = mel_amp * (P2 / (H2 + P2 + eps))

    log_harm = np.log(harm + 1e-9)   # [128, 130]
    log_perc = np.log(perc + 1e-9)   # [128, 130]

    arr = np.stack([log_harm, log_perc])   # [2, 128, 130]
    if use_cache:
        np.save(cache_path, arr)

    return log_harm, log_perc


# ── CQT ───────────────────────────────────────────────────────────────────────
# Implemented without librosa (which requires _lzma, absent from this pyenv build).
# Approach: high-resolution STFT → average STFT bins within each constant-Q band.
#
# Each CQT bin k has centre frequency f_k = CQT_FMIN * 2^(k / CQT_BINS_PER_OCT)
# and edges f_k * 2^(±0.5 / CQT_BINS_PER_OCT), giving a constant Q-factor of
#   Q = 1 / (2^(1/CQT_BINS_PER_OCT) - 1) ≈ 34  for 24 bins/oct.
#
# N_FFT_CQT=4096 gives ~5.4 Hz STFT resolution, which is adequate for f ≥ 100 Hz.
# Centre-padded (reflect) so output is exactly TIME_FRAMES columns.

_N_FFT_CQT    = 4096
_CQT_WIN      = torch.hann_window(_N_FFT_CQT)
_STFT_FREQS   = np.linspace(0, SAMPLE_RATE / 2, _N_FFT_CQT // 2 + 1)  # [2049]

# Precompute which STFT bins map to each CQT bin (constant across all files)
_cqt_f_centre = CQT_FMIN * 2 ** (np.arange(CQT_N_BINS) / CQT_BINS_PER_OCT)
_cqt_f_lo     = _cqt_f_centre * 2 ** (-0.5 / CQT_BINS_PER_OCT)
_cqt_f_hi     = _cqt_f_centre * 2 ** ( 0.5 / CQT_BINS_PER_OCT)

# For each CQT bin: list of STFT bin indices that fall inside [f_lo, f_hi]
# Falls back to nearest STFT bin when the CQT band is narrower than one STFT bin.
_cqt_stft_idx: list[np.ndarray] = []
for k in range(CQT_N_BINS):
    inside = np.where((_STFT_FREQS >= _cqt_f_lo[k]) & (_STFT_FREQS < _cqt_f_hi[k]))[0]
    if len(inside) == 0:
        inside = np.array([int(np.argmin(np.abs(_STFT_FREQS - _cqt_f_centre[k])))])
    _cqt_stft_idx.append(inside)


def compute_cqt_spectrogram(wav_path: str | Path, use_cache: bool = True) -> np.ndarray:
    """Return pseudo-CQT of shape [CQT_N_BINS, TIME_FRAMES] = [168, 130].

    Uses torch.stft (no librosa / lzma needed).
    Cached as <name>.cqt.npy alongside the wav file.
    """
    wav_path   = Path(wav_path)
    cache_path = wav_path.with_suffix(".cqt.npy")

    if use_cache and cache_path.exists():
        return np.load(cache_path)

    data, sr = sf.read(wav_path, always_2d=True)
    waveform  = torch.from_numpy(data.mean(axis=1)).float()

    if sr != SAMPLE_RATE:
        if sr not in _resample_cache:
            _resample_cache[sr] = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = _resample_cache[sr](waveform)

    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    # Centre-pad so the output has the same number of frames as the mel pipeline
    pad      = _N_FFT_CQT // 2
    waveform = torch.nn.functional.pad(waveform.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)

    stft_out = torch.stft(
        waveform,
        n_fft      = _N_FFT_CQT,
        hop_length = HOP_LENGTH,
        window     = _CQT_WIN,
        return_complex = True,
    )                                       # [2049, T]
    S = stft_out.abs().numpy()              # [2049, T]

    # Aggregate STFT bins into CQT bins
    cqt = np.stack([S[idx].mean(axis=0) for idx in _cqt_stft_idx])  # [168, T]

    log_cqt = np.log(cqt + 1e-9)

    t = log_cqt.shape[1]
    if t >= TIME_FRAMES:
        log_cqt = log_cqt[:, :TIME_FRAMES]
    else:
        log_cqt = np.pad(log_cqt, ((0, 0), (0, TIME_FRAMES - t)),
                         mode="constant", constant_values=np.log(1e-9))

    if use_cache:
        np.save(cache_path, log_cqt)

    return log_cqt  # [168, 130]
