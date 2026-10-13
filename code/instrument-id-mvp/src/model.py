"""
Models for multi-label instrument identification.

Input:  [batch, 1, 128, 130]   (channels, n_mels, time_frames)
Output: [batch, NUM_CLASSES]   raw logits (apply sigmoid for probabilities)

Architectures:
  BasicCNN         — HarmonicCNN backbone without harmonic stacking (ablation)
  HarmonicCNN      — domain-aware CNN with harmonic stacking (baseline)
  DeltaHarmonicCNN — HarmonicCNN + Δmel + Δ²mel channels for attack/decay
  DeepHarmonicCNN  — DeltaHarmonicCNN + 4th conv block (128→256 filters)
  DeepExactHarmonicCNN — DeepHarmonicCNN with frequency-exact harmonics 1–5
  DeepExactOctaveCNN   — ablation: exact octaves 1, 2, 4, 8
  DeepMelCNN           — ablation: same backbone, no harmonic views
  DeepExactOctaveMaxCNN / DeepMelMaxCNN — the same with max pooling in block4
  DeepMelFreqCNN       — DeepMelCNN + frequency-position channel (instrument register)
  DeepMelPonsCNN       — DeepMelCNN with musically shaped first-layer filters
  DeepMelLogPonsCNN    — Pons filters with the vertical branch on a log-frequency axis
  DeepMelHPSPonsCNN    — DeepMelPonsCNN + soft-min harmonic (HPS-like) channel
  DeepMelMultiLogPonsCNN — log-frequency vertical filters of ½, 2 and 4 octaves
  DeepMelLogPonsMaxCNN — DeepMelLogPonsCNN with max pooling in block4
  PonsMultiLayerCNN    — re-implementation of the Pons et al. 2017 IRMAS model
  LogPonsMultiLayerCNN — the same with its front end on a log-frequency axis
  Han2017CNN       — reproduction of Han et al. arXiv 1605.09507
                     Input: [batch, 1, 128, 43]  (1-second window)
  HarmonicHan2017CNN — Han2017CNN with harmonic stacking (4 channels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import NUM_CLASSES
from features import N_MELS, SAMPLE_RATE


class BasicCNN(nn.Module):
    """
    HarmonicCNN backbone without harmonic stacking — ablation baseline.

    Identical to HarmonicCNN except the first conv takes 1 raw mel channel
    instead of 4 harmonic-shifted channels. Used to isolate the contribution
    of the harmonic inductive bias.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


class HarmonicCNN(nn.Module):
    """
    Harmonic-aware CNN for instrument identification.

    Musical instruments produce overtones at f, 2f, 3f, 4f...
    On a mel spectrogram each octave spans ~BINS_PER_OCT bins.

    _harmonic_stack() shifts the spectrogram by 0, 1, 2, 3 octaves and
    stacks the results as separate input channels — so a standard 3×3 conv
    at position (freq, time) simultaneously sees energy at four harmonic
    frequencies.  The network can trivially learn "fundamental + overtones
    present → instrument X".
    """

    NUM_HARMONICS = 4   # fundamental + 3 overtones
    BINS_PER_OCT  = 12  # ≈ mel bins per octave for n_mels=128, sr=22050

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(self.NUM_HARMONICS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # [32, 64, 65]
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # [64, 32, 32]
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # [128, 4, 4]
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 128 * 4 * 4 = 2048
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _harmonic_stack(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1, F, T]
        out: [B, NUM_HARMONICS, F, T]

        Channel h contains the spectrogram shifted upward by h octaves:
          ch=0 → x[f]                    (fundamental view)
          ch=1 → x[f + 1*BINS_PER_OCT]  (2nd harmonic view)
          ch=2 → x[f + 2*BINS_PER_OCT]  (4th harmonic view)
          ch=3 → x[f + 3*BINS_PER_OCT]  (8th harmonic view)
        Out-of-range bins are filled with the log-zero sentinel (-20.7).
        """
        B, C, F, T = x.shape
        sentinel = torch.full((B, C, 1, T), fill_value=-20.7,
                              device=x.device, dtype=x.dtype)
        channels = [x]
        for h in range(1, self.NUM_HARMONICS):
            shift   = h * self.BINS_PER_OCT
            pad     = sentinel.expand(B, C, shift, T)
            shifted = torch.cat([x[:, :, shift:, :], pad], dim=2)
            channels.append(shifted)
        return torch.cat(channels, dim=1)  # [B, NUM_HARMONICS, F, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._harmonic_stack(x)   # [B, 4, 128, 130]
        x = self.block1(x)            # [B, 32,  64,  65]
        x = self.block2(x)            # [B, 64,  32,  32]
        x = self.block3(x)            # [B, 128,  4,   4]
        return self.classifier(x)     # [B, 11]


class DeltaHarmonicCNN(nn.Module):
    """
    HarmonicCNN augmented with temporal-derivative channels.

    Musical instruments differ not only in their harmonic content but in how
    that content changes over time — their amplitude envelope:
      • piano / gac  : sharp attack, fast decay          → large |Δ| at onset
      • org          : slow attack, flat sustain          → near-zero Δ throughout
      • vio / cel    : bow-sustained, possible vibrato    → gentle, oscillating Δ
      • tru / sax    : tongue-articulated, variable decay → moderate Δ with bursts

    Input channels (6 total):
      ch 0–3 : harmonic stack (mel shifted by 0/1/2/3 octaves) — same as HarmonicCNN
      ch 4   : Δmel   — central finite difference along the time axis
      ch 5   : Δ²mel  — second temporal derivative (rate-of-change of the envelope)

    No changes to the CNN backbone or classifier — only the first Conv2d now
    takes 6 channels instead of 4.
    """

    NUM_HARMONICS = 4
    BINS_PER_OCT  = 12
    IN_CHANNELS   = 6   # 4 harmonic + Δ + Δ²

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 128 * 4 * 4 = 2048
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _delta(x: torch.Tensor) -> torch.Tensor:
        """Central finite difference along the time axis. [B, 1, F, T] → [B, 1, F, T]."""
        xp = F.pad(x, (1, 1, 0, 0), mode='replicate')  # [B, 1, F, T+2]
        return (xp[:, :, :, 2:] - xp[:, :, :, :-2]) * 0.5

    def _harmonic_stack(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        sentinel = torch.full((B, C, 1, T), -20.7, device=x.device, dtype=x.dtype)
        channels = [x]
        for h in range(1, self.NUM_HARMONICS):
            shift   = h * self.BINS_PER_OCT
            pad     = sentinel.expand(B, C, shift, T)
            shifted = torch.cat([x[:, :, shift:, :], pad], dim=2)
            channels.append(shifted)
        return torch.cat(channels, dim=1)   # [B, 4, F, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 128, 130]
        harmonic = self._harmonic_stack(x)              # [B, 4, 128, 130]
        delta    = self._delta(x)                       # [B, 1, 128, 130]
        delta2   = self._delta(delta)                   # [B, 1, 128, 130]
        x = torch.cat([harmonic, delta, delta2], dim=1) # [B, 6, 128, 130]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


class DeepHarmonicCNN(nn.Module):
    """
    DeltaHarmonicCNN with a 4th conv block (128→256 filters).

    Identical 6-channel harmonic+delta input. The extra block adds representational
    depth without changing the inductive bias — same BN+ReLU+MaxPool stack.

    Architecture:
      block1: Conv(6→32)  + BN + ReLU + MaxPool(2,2)   → [32, 64, 65]
      block2: Conv(32→64) + BN + ReLU + MaxPool(2,2)   → [64, 32, 32]
      block3: Conv(64→128)+ BN + ReLU + MaxPool(2,2)   → [128, 16, 16]
      block4: Conv(128→256)+BN + ReLU + AdaptiveAvgPool(2,2) + Dropout2d → [256, 2, 2]
      classifier: Linear(1024→256→11)
    """

    NUM_HARMONICS = 4
    BINS_PER_OCT  = 12
    IN_CHANNELS   = 6

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 256 * 2 * 2 = 1024
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _delta(x: torch.Tensor) -> torch.Tensor:
        xp = F.pad(x, (1, 1, 0, 0), mode='replicate')
        return (xp[:, :, :, 2:] - xp[:, :, :, :-2]) * 0.5

    def _harmonic_stack(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        sentinel = torch.full((B, C, 1, T), -20.7, device=x.device, dtype=x.dtype)
        channels = [x]
        for h in range(1, self.NUM_HARMONICS):
            shift   = h * self.BINS_PER_OCT
            pad     = sentinel.expand(B, C, shift, T)
            shifted = torch.cat([x[:, :, shift:, :], pad], dim=2)
            channels.append(shifted)
        return torch.cat(channels, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        harmonic = self._harmonic_stack(x)
        delta    = self._delta(x)
        delta2   = self._delta(delta)
        x = torch.cat([harmonic, delta, delta2], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)


class ExactHarmonicStack(nn.Module):
    """
    Frequency-exact harmonic stacking on a log-mel spectrogram.

    The fixed BINS_PER_OCT=12 shift used above is only right around 300–600 Hz:
    on the HTK mel scale (128 bins, 0–11025 Hz) an octave spans ~3 bins at
    55–110 Hz but ~28 bins at 3.5–7 kHz. Here, channel h at bin k holds the
    log-mel value at frequency h·f_k (f_k = centre of bin k), linearly
    interpolated between neighbouring mel bins — the mel analogue of the
    harmonic CQT (Bittner et al. 2017).

    x  : [B, 1, F, T]
    out: [B, len(harmonics), F, T]; bins whose h·f_k exceeds Nyquist hold the
         log-zero sentinel.
    """

    def __init__(self, harmonics, n_mels: int = N_MELS, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        hz_to_mel = lambda f: 2595.0 * torch.log10(1.0 + f / 700.0)
        mel_max  = hz_to_mel(torch.tensor(sample_rate / 2.0, dtype=torch.float64))
        mel_step = mel_max / (n_mels + 1)
        # torchaudio places n_mels+2 equally spaced mel points on [0, mel_max];
        # bin k is centred on point k+1
        centre_mel = mel_step * torch.arange(1, n_mels + 1, dtype=torch.float64)
        centre_hz  = 700.0 * (10.0 ** (centre_mel / 2595.0) - 1.0)

        h   = torch.tensor(harmonics, dtype=torch.float64).unsqueeze(1)   # [H, 1]
        pos = hz_to_mel(h * centre_hz) / mel_step - 1.0                   # [H, F] fractional bin
        valid = pos <= n_mels - 1 + 1e-6
        pos   = pos.clamp(0, n_mels - 1)
        lo    = pos.floor().long().clamp(max=n_mels - 2)
        w     = pos - lo

        self.register_buffer("lo",    lo,                          persistent=False)
        self.register_buffer("w",     w.float()[None, :, :, None], persistent=False)
        self.register_buffer("valid", valid[None, :, :, None],     persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0]                          # [B, F, T]
        lo_val = x0[:, self.lo]               # [B, H, F, T]
        hi_val = x0[:, self.lo + 1]
        out = lo_val * (1.0 - self.w) + hi_val * self.w
        return torch.where(self.valid, out, torch.full_like(out, -20.7))


class DeepExactHarmonicCNN(DeepHarmonicCNN):
    """
    DeepHarmonicCNN with frequency-exact harmonic stacking (ExactHarmonicStack)
    instead of the fixed 12-bin octave shift. Harmonics 1–5 include the odd
    partials 3f and 5f that the octave stack (1, 2, 4, 8) cannot see.

    Input channels: len(HARMONICS) harmonic views + Δ + Δ².
    """

    HARMONICS   = (1, 2, 3, 4, 5)
    IN_CHANNELS = len(HARMONICS) + 2

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__(num_classes)
        self.stack = ExactHarmonicStack(self.HARMONICS)

    def _harmonic_stack(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class DeepExactOctaveCNN(DeepExactHarmonicCNN):
    """Ablation: same octaves as DeepHarmonicCNN (1, 2, 4, 8), but frequency-exact."""

    HARMONICS   = (1, 2, 4, 8)
    IN_CHANNELS = len(HARMONICS) + 2


class DeepMelCNN(DeepExactHarmonicCNN):
    """Ablation: DeepHarmonicCNN backbone with no harmonic views (mel + Δ + Δ²)."""

    HARMONICS   = (1,)
    IN_CHANNELS = len(HARMONICS) + 2


class DeepMelFreqCNN(DeepMelCNN):
    """
    DeepMelCNN + a frequency-position channel (CoordConv, Liu et al. 2018).

    Convolutions are translation-invariant along frequency, but instruments
    have fixed registers (cello ~65 Hz–1 kHz, flute ~260 Hz–2 kHz). A channel
    holding the normalised mel-bin index in [-1, 1] lets every filter know
    where in the spectrum it is. Unlike harmonic stacking it stays correct
    in polyphonic mixtures.

    Input channels: mel + freq-position + Δ + Δ².
    """

    IN_CHANNELS = 4

    def _harmonic_stack(self, x: torch.Tensor) -> torch.Tensor:
        B, _, F, T = x.shape
        pos = torch.linspace(-1.0, 1.0, F, device=x.device, dtype=x.dtype)
        return torch.cat([x, pos.view(1, 1, F, 1).expand(B, 1, F, T)], dim=1)


class PonsConv(nn.Module):
    """
    Musically motivated first layer (Pons et al. 2017, "Timbre analysis of
    music audio signals with convolutional neural networks").

    Three filter shapes run in parallel and are concatenated:
      3×3      — local patterns (as in the rest of the network)
      31×1     — vertical: ~1–2 octaves at one instant → spectral envelope / timbre
      1×31     — horizontal: one band over ~0.7 s → attack, decay, vibrato
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        n_sq = out_ch // 2
        n_v  = out_ch // 4
        n_h  = out_ch - n_sq - n_v
        self.square     = nn.Conv2d(in_ch, n_sq, kernel_size=3,       padding=1)
        self.vertical   = nn.Conv2d(in_ch, n_v,  kernel_size=(31, 1), padding=(15, 0))
        self.horizontal = nn.Conv2d(in_ch, n_h,  kernel_size=(1, 31), padding=(0, 15))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.square(x), self.vertical(x), self.horizontal(x)], dim=1)


class DeepMelPonsCNN(DeepMelCNN):
    """DeepMelCNN whose first conv is PonsConv (square + vertical + horizontal filters)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__(num_classes)
        self.block1[0] = PonsConv(self.IN_CHANNELS, 32)


def _interp_matrix(src_pos: torch.Tensor, n_src: int) -> torch.Tensor:
    """[len(src_pos), n_src] linear-interpolation weights reading fractional
    positions src_pos (in source-bin units, clamped to range) from a source axis."""
    pos = src_pos.clamp(0, n_src - 1)
    lo  = pos.floor().long().clamp(max=n_src - 2)
    w   = (pos - lo).to(torch.float64)
    m   = torch.zeros(len(pos), n_src, dtype=torch.float64)
    idx = torch.arange(len(pos))
    m[idx, lo]     += 1.0 - w
    m[idx, lo + 1] += w
    return m.float()


class LogFreqPonsConv(PonsConv):
    """
    PonsConv whose vertical (timbre) branch runs on a log-frequency axis.

    On the mel axis an octave spans ~3 bins at 55–110 Hz but ~28 bins at
    3.5–7 kHz, so a fixed 31×1 filter covers ~5 octaves in the bass and ~1 in
    the treble: the same instrument's overtone pattern looks different
    depending on the note played. Here the input is resampled to
    BINS_PER_OCT bins per octave, the vertical filter (OCTAVES tall) is applied
    there — the same musical interval at every pitch, so a learned timbre
    pattern is transposition-invariant — and the result is resampled back
    onto the mel grid to join the square and horizontal branches.
    """

    BINS_PER_OCT = 24
    OCTAVES      = 2
    F_MIN        = 32.70   # C1

    def __init__(self, in_ch: int, out_ch: int,
                 n_mels: int = N_MELS, sample_rate: int = SAMPLE_RATE):
        super().__init__(in_ch, out_ch)
        hz_to_mel = lambda f: 2595.0 * torch.log10(1.0 + f / 700.0)
        mel_step  = hz_to_mel(torch.tensor(sample_rate / 2.0, dtype=torch.float64)) / (n_mels + 1)
        mel_hz    = 700.0 * (10.0 ** (mel_step * torch.arange(1, n_mels + 1, dtype=torch.float64)
                                      / 2595.0) - 1.0)                     # mel bin centres

        n_log  = int(self.BINS_PER_OCT * torch.log2(mel_hz[-1] / self.F_MIN)) + 1
        log_hz = self.F_MIN * 2.0 ** (torch.arange(n_log, dtype=torch.float64) / self.BINS_PER_OCT)

        to_log = _interp_matrix(hz_to_mel(log_hz) / mel_step - 1.0, n_mels)          # [L, F]
        to_mel = _interp_matrix(self.BINS_PER_OCT * torch.log2(mel_hz / self.F_MIN), n_log)  # [F, L]
        self.register_buffer("to_log", to_log, persistent=False)
        self.register_buffer("to_mel", to_mel, persistent=False)

        k = self.BINS_PER_OCT * self.OCTAVES + 1
        self.vertical = nn.Conv2d(in_ch, self.vertical.out_channels,
                                  kernel_size=(k, 1), padding=(k // 2, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_log = torch.einsum("lf,bcft->bclt", self.to_log, x)
        v     = torch.einsum("fl,bclt->bcft", self.to_mel, self.vertical(x_log))
        return torch.cat([self.square(x), v, self.horizontal(x)], dim=1)


class DeepMelLogPonsCNN(DeepMelCNN):
    """DeepMelCNN whose first conv is LogFreqPonsConv (octave-consistent timbre filters)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__(num_classes)
        self.block1[0] = LogFreqPonsConv(self.IN_CHANNELS, 32)


class MultiLogPonsConv(LogFreqPonsConv):
    """
    LogFreqPonsConv with several vertical filter heights, each a fixed musical
    interval on the log-frequency axis:
      ½ octave — formant-scale detail (e.g. vowel formants, reed resonances)
      2 octaves — envelope of the first overtones
      4 octaves — overall spectral tilt of the instrument
    The vertical channels are split between the three heights.
    """

    INTERVALS_OCT = (0.5, 2, 4)

    def __init__(self, in_ch: int, out_ch: int, **kw):
        super().__init__(in_ch, out_ch, **kw)
        n_v   = self.vertical.out_channels
        sizes = [n_v // len(self.INTERVALS_OCT)] * len(self.INTERVALS_OCT)
        sizes[0] += n_v - sum(sizes)
        heights = [int(self.BINS_PER_OCT * o) + 1 for o in self.INTERVALS_OCT]
        self.verticals = nn.ModuleList(
            nn.Conv2d(in_ch, n, kernel_size=(h, 1), padding=(h // 2, 0))
            for n, h in zip(sizes, heights)
        )
        del self.vertical

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_log = torch.einsum("lf,bcft->bclt", self.to_log, x)
        v_log = torch.cat([conv(x_log) for conv in self.verticals], dim=1)
        v     = torch.einsum("fl,bclt->bcft", self.to_mel, v_log)
        return torch.cat([self.square(x), v, self.horizontal(x)], dim=1)


class DeepMelMultiLogPonsCNN(DeepMelCNN):
    """DeepMelCNN whose first conv is MultiLogPonsConv (½, 2 and 4-octave timbre filters)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__(num_classes)
        self.block1[0] = MultiLogPonsConv(self.IN_CHANNELS, 32)


class PonsFrontEnd(nn.Module):
    """
    First layer of the Pons et al. 2017 multi-layer IRMAS model (Sec. III-B):
    for each of two filter heights (5/96 and 80/96 of the frequency axis —
    "small" and "high" timbre filters) 128 filters of width 1, 64 of width 3
    and 32 of width 5; same padding. The paper's wording also allows 224
    filters in total; 448 (each count per height) is used here.

    log_freq=True runs all filters on a log-frequency axis (24 bins/octave,
    as in LogFreqPonsConv) so each filter height is a fixed musical interval,
    then resamples the responses back onto the mel grid.
    """

    COUNTS_WIDTHS = ((128, 1), (64, 3), (32, 5))
    HEIGHT_FRACS  = (5 / 96, 80 / 96)

    def __init__(self, in_ch: int = 1, n_mels: int = N_MELS, log_freq: bool = False):
        super().__init__()
        self.log_freq = log_freq
        if log_freq:
            lf = LogFreqPonsConv(in_ch, 4, n_mels=n_mels)      # only borrow its resampling matrices
            self.register_buffer("to_log", lf.to_log, persistent=False)
            self.register_buffer("to_mel", lf.to_mel, persistent=False)
            axis = lf.to_log.shape[0]
        else:
            axis = n_mels
        heights = [int(round(f * axis)) | 1 for f in self.HEIGHT_FRACS]   # odd → symmetric padding
        self.convs = nn.ModuleList(
            nn.Conv2d(in_ch, n, kernel_size=(h, w), padding=(h // 2, w // 2))
            for h in heights for n, w in self.COUNTS_WIDTHS
        )
        self.out_channels = sum(c.out_channels for c in self.convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.log_freq:
            x = torch.einsum("lf,bcft->bclt", self.to_log, x)
        y = torch.cat([conv(x) for conv in self.convs], dim=1)
        if self.log_freq:
            y = torch.einsum("fl,bclt->bcft", self.to_mel, y)
        return y


class PonsMultiLayerCNN(nn.Module):
    """
    Re-implementation of the Pons et al. 2017 multi-layer instrument model
    (reported on IRMAS: macro F1 0.516, micro 0.589, 743k params) on our
    128-band log-mel input:

      PonsFrontEnd (448 filters) + BN + ELU + MaxPool(16, 16) → 8 bands × 8 frames
      Conv 3×3 (128) + BN + ELU + MaxPool(2, 2)
      Conv 3×3 (128) + BN + ELU + MaxPool(2, 2)
      FC 256 + ELU → FC 11
      Dropout 25% after conv layers, 50% on the dense layer.

    Deviations from the paper: 128 mel bands at 22.05 kHz instead of 96 at
    12 kHz (filter heights scaled), sigmoid + BCE (multi-label, needed for
    PolyMix) instead of softmax, Adam + cosine LR instead of SGD.
    """

    LOG_FREQ = False

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.front = PonsFrontEnd(1, log_freq=self.LOG_FREQ)
        c = self.front.out_channels
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(c), nn.ELU(inplace=True),
            nn.MaxPool2d((N_MELS // 8, 16)), nn.Dropout2d(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ELU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ELU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 128 × 2 × 2 = 512
            nn.Linear(512, 256), nn.ELU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(self.front(x))          # [B, 448, 8, 8]
        x = self.block2(x)                      # [B, 128, 4, 4]
        x = self.block3(x)                      # [B, 128, 2, 2]
        return self.classifier(x)


class LogPonsMultiLayerCNN(PonsMultiLayerCNN):
    """PonsMultiLayerCNN with the front end on a log-frequency axis (our LogPons idea)."""

    LOG_FREQ = True


class DeepMelHPSPonsCNN(DeepMelPonsCNN):
    """
    DeepMelPonsCNN + one mixture-robust harmonic channel.

    Separate harmonic-shifted channels (DeepHarmonicCNN) lost accuracy under
    PolyMix: in a mixture the "energy at 2f" channel often shows another
    instrument. This channel is a soft minimum over the log-mel values at
    f, 2f, 3f, 4f (frequency-exact, ExactHarmonicStack): it is high only where
    a whole harmonic series starts, so chance overlaps do not switch it on —
    a smooth harmonic-product-spectrum.

    Input channels: mel + harmonic soft-min + Δ + Δ².
    """

    IN_CHANNELS = 4
    HPS_HARMONICS = (1, 2, 3, 4)
    HPS_TEMP      = 1.0     # soft-min temperature in log-mel units

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__(num_classes)
        self.hps_stack = ExactHarmonicStack(self.HPS_HARMONICS)

    def _harmonic_stack(self, x: torch.Tensor) -> torch.Tensor:
        h   = self.hps_stack(x)                                          # [B, H, F, T]
        hps = -self.HPS_TEMP * torch.logsumexp(-h / self.HPS_TEMP, dim=1, keepdim=True)
        return torch.cat([x, hps], dim=1)


def _use_max_pool(model: nn.Module) -> None:
    """Swap block4's AdaptiveAvgPool2d for AdaptiveMaxPool2d (same output size).

    Averaging over the whole 3-s window dilutes an instrument that plays only
    briefly; max keeps its strongest activation (Han et al. use global max).
    """
    idx = next(i for i, m in enumerate(model.block4) if isinstance(m, nn.AdaptiveAvgPool2d))
    model.block4[idx] = nn.AdaptiveMaxPool2d(model.block4[idx].output_size)


class DeepExactOctaveMaxCNN(DeepExactOctaveCNN):
    """DeepExactOctaveCNN with max instead of average pooling in block4."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__(num_classes)
        _use_max_pool(self)


class DeepMelLogPonsMaxCNN(DeepMelLogPonsCNN):
    """DeepMelLogPonsCNN with max instead of average pooling in block4."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__(num_classes)
        _use_max_pool(self)


class DeepMelMaxCNN(DeepMelCNN):
    """Ablation: DeepMelCNN with max instead of average pooling in block4."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__(num_classes)
        _use_max_pool(self)


class _SE(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al. 2018)."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).view(x.shape[0], x.shape[1], 1, 1)


class SEDeepHarmonicCNN(nn.Module):
    """DeepHarmonicCNN with Squeeze-Excitation channel attention after each block."""

    NUM_HARMONICS = 4
    BINS_PER_OCT  = 12
    IN_CHANNELS   = 6

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), _SE(32),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), _SE(64),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), _SE(128),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Dropout2d(0.25), _SE(256),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes),
        )

    @staticmethod
    def _delta(x: torch.Tensor) -> torch.Tensor:
        xp = F.pad(x, (1, 1, 0, 0), mode='replicate')
        return (xp[:, :, :, 2:] - xp[:, :, :, :-2]) * 0.5

    def _harmonic_stack(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        sentinel = torch.full((B, C, 1, T), -20.7, device=x.device, dtype=x.dtype)
        channels = [x]
        for h in range(1, self.NUM_HARMONICS):
            shift = h * self.BINS_PER_OCT
            pad   = sentinel.expand(B, C, shift, T)
            channels.append(torch.cat([x[:, :, shift:, :], pad], dim=2))
        return torch.cat(channels, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        harmonic = self._harmonic_stack(x)
        delta    = self._delta(x)
        delta2   = self._delta(delta)
        x = torch.cat([harmonic, delta, delta2], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)


class Han2017CNN(nn.Module):
    """
    Reproduction of Han, Kim, Lee (arXiv 1605.09507).

    Architecture (Table I):
      4× conv block, each with 2 Conv(3×3) + LReLU(0.33) + MaxPool(3×3, s=2) + Dropout(0.25)
      Filter counts: 32 → 64 → 128 → 256
      GlobalMaxPool → FC(1024) → LReLU → Dropout(0.5) → FC(11)

    Input: [B, 1, 128, 43]  (1-second window: 128 mel bins × 43 time frames)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        lrelu = lambda: nn.LeakyReLU(negative_slope=0.33, inplace=True)

        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1),
                lrelu(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                lrelu(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout2d(0.25),
            )

        self.block1 = _block(1,   32)
        self.block2 = _block(32,  64)
        self.block3 = _block(64,  128)
        self.block4 = _block(128, 256)

        self.global_max = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 1024),
            lrelu(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 128, 43]
        x = self.block1(x)   # [B, 32,  64, 22]
        x = self.block2(x)   # [B, 64,  32, 11]
        x = self.block3(x)   # [B, 128, 16,  6]
        x = self.block4(x)   # [B, 256,  8,  3]
        x = self.global_max(x)       # [B, 256, 1, 1]
        return self.classifier(x)    # [B, 11]


class HarmonicHan2017CNN(nn.Module):
    """
    Han2017CNN with harmonic stacking as input (4 channels instead of 1).

    Same backbone as Han2017CNN but block1 receives 4-channel harmonic stack
    instead of raw mono mel — giving the network an explicit view of overtones
    without changing parameter count significantly.

    Input: [B, 1, 128, 43]  (same as Han; harmonic stack applied inside forward)
    """

    NUM_HARMONICS = 4
    BINS_PER_OCT  = 12

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        lrelu = lambda: nn.LeakyReLU(negative_slope=0.33, inplace=True)

        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1),
                lrelu(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                lrelu(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout2d(0.25),
            )

        self.block1 = _block(self.NUM_HARMONICS, 32)
        self.block2 = _block(32,  64)
        self.block3 = _block(64,  128)
        self.block4 = _block(128, 256)

        self.global_max = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 1024),
            lrelu(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def _harmonic_stack(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        sentinel = torch.full((B, C, 1, T), -20.7, device=x.device, dtype=x.dtype)
        channels = [x]
        for h in range(1, self.NUM_HARMONICS):
            shift = h * self.BINS_PER_OCT
            pad   = sentinel.expand(B, C, shift, T)
            channels.append(torch.cat([x[:, :, shift:, :], pad], dim=2))
        return torch.cat(channels, dim=1)   # [B, 4, F, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 128, 43]
        x = self._harmonic_stack(x)  # [B, 4, 128, 43]
        x = self.block1(x)           # [B, 32,  64, 22]
        x = self.block2(x)           # [B, 64,  32, 11]
        x = self.block3(x)           # [B, 128, 16,  6]
        x = self.block4(x)           # [B, 256,  8,  3]
        x = self.global_max(x)       # [B, 256,  1,  1]
        return self.classifier(x)    # [B, 11]
