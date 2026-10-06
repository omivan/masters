"""
Models for multi-label instrument identification.

Input:  [batch, 1, 128, 130]   (channels, n_mels, time_frames)
Output: [batch, NUM_CLASSES]   raw logits (apply sigmoid for probabilities)

Architectures:
  InstrumentCNN         — baseline: CNN + Global Average Pooling + MLP
  InstrumentCNNBiLSTM   — CNN + BiLSTM temporal modelling
  HarmonicCNN           — domain-aware CNN that explicitly sees harmonic structure
  HarmonicSeriesCNN     — true harmonic series (f,2f,3f,4f) with correct mel-bin offsets
  HarmonicProductCNN    — HarmonicSeriesCNN + Harmonic Product Spectrum channel
  CQTHarmonicCNN        — HarmonicCNN on CQT input (exact harmonic alignment)
  HPSSCombinedCNN       — harmonic stacking on HPSS harmonic + raw percussive channel
  DeltaHarmonicCNN      — HarmonicCNN + Δmel + Δ²mel channels for attack/decay
  LearnableHarmonicCNN  — HarmonicCNN with backprop-learned harmonic shift positions
  OddEvenHarmonicCNN    — dual-branch: separate paths for odd and even harmonics
  HarmonicSECNN         — HarmonicCNN + Squeeze-and-Excitation channel attention
  AttackSustainCNN      — dual-branch: separate paths for attack and sustain phases
  ContrastiveCNN        — two-stage: self-supervised pre-training + fine-tuning
  ProtoNet              — metric learning with per-class prototype embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import NUM_CLASSES


class InstrumentCNN(nn.Module):
    """Baseline CNN with Global Average Pooling — no temporal modelling."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # [16, 64, 65]

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # [32, 32, 32]

            # Block 3 — extra depth, GAP to fixed size
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # [64, 4, 4]

            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 64 * 4 * 4 = 1024
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.cnn(x))


class AttackSustainCNN(nn.Module):
    """
    Dual-branch CNN: separate processing paths for attack and sustain phases.

    Musical instruments are most distinguishable in two complementary regions:
      • Attack  (first ~460 ms = 20 frames): transient onset — bow scrape,
        reed buzz, hammer strike.  Percussive and plucked instruments are
        easiest to separate here.
      • Sustain (remaining ~2.5 s = 110 frames): steady-state timbre and
        harmonic envelope.  Wind and bowed instruments shine here.

    Both branches share the same CNN topology but have independent weights,
    letting them specialise.  A learned 2-way gate dynamically weights how
    much each branch contributes to the final prediction for each sample.
    """

    ATTACK_FRAMES = 20   # ≈ 464 ms  (20 × 512 / 22050)

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def _branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16), nn.ReLU(True), nn.MaxPool2d(2, 2),

                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2, 2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(True),
                nn.AdaptiveAvgPool2d((4, 4)),   # fixed output regardless of T
                nn.Dropout2d(0.25),
                nn.Flatten(),                   # [B, 64*4*4 = 1024]
            )

        self.attack_branch  = _branch()
        self.sustain_branch = _branch()

        feat_dim = 64 * 4 * 4  # 1024 per branch

        # Learned gate: for each sample decide how much to trust attack vs sustain
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, 2),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 128, 130]
        atk = x[:, :, :, :self.ATTACK_FRAMES]   # [B, 1, 128,  20]
        sus = x[:, :, :, self.ATTACK_FRAMES:]   # [B, 1, 128, 110]

        h_a = self.attack_branch(atk)    # [B, 1024]
        h_s = self.sustain_branch(sus)   # [B, 1024]

        # Gate: [B, 2] — how much weight to give each branch
        w   = self.gate(torch.cat([h_a, h_s], dim=1))
        h   = w[:, 0:1] * h_a + w[:, 1:2] * h_s   # [B, 1024] weighted sum

        return self.classifier(h)        # [B, num_classes]


class ContrastiveCNN(nn.Module):
    """
    Two-stage contrastive pre-training + supervised fine-tuning.

    Stage 1 — self-supervised (no labels needed):
        Two augmented views of the same mel spectrogram are fed through the
        encoder.  The NT-Xent loss pulls their embeddings together while
        pushing embeddings of different samples apart.  The model learns a
        rich representation of timbral similarity without any instrument labels.

    Stage 2 — supervised fine-tuning:
        The projection head is discarded.  A lightweight classifier head is
        attached to the frozen/unfrozen encoder and trained with BCEWithLogitsLoss.

    Use set_mode('pretrain') / set_mode('finetune') to switch between stages.
    """

    PRETRAIN = "pretrain"
    FINETUNE = "finetune"

    def __init__(self, num_classes: int = NUM_CLASSES,
                 enc_dim: int = 256, proj_dim: int = 64):
        super().__init__()
        self.enc_dim = enc_dim
        self._mode   = self.PRETRAIN

        # ── Shared CNN backbone ───────────────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(True), nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.25),
        )

        # ── Shared linear encoder → enc_dim representation ───────────────────
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, enc_dim),
            nn.ReLU(True),
        )

        # ── Stage 1: non-linear projection head (discarded after pre-training)
        self.projector = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.ReLU(True),
            nn.Linear(enc_dim, proj_dim),
        )

        # ── Stage 2: classifier head ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(enc_dim, num_classes),
        )

    def set_mode(self, mode: str):
        assert mode in (self.PRETRAIN, self.FINETUNE)
        self._mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.cnn(x))           # [B, enc_dim]
        if self._mode == self.PRETRAIN:
            return F.normalize(self.projector(h), dim=1)  # [B, proj_dim] L2-normed
        return self.classifier(h)                # [B, num_classes] raw logits


class ProtoNet(nn.Module):
    """
    Prototype Network for multi-label instrument identification.

    Instead of a fixed classifier head, this model learns an embedding space
    where same-instrument sounds cluster together.  Classification is done at
    inference by comparing a query embedding to per-class prototypes (the mean
    embedding of all training examples for that class).

    Training loss: multi-label prototypical loss —
      for each class k, compute a batch prototype from positive examples,
      then apply BCE on cosine similarities of all examples to that prototype.

    Inference:
      1. compute_prototypes() — one forward pass over training set
      2. predict()            — cosine_sim(query, prototype_k) > threshold
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()

        self.embed_dim = embed_dim

        # Same CNN backbone as InstrumentCNN (comparable parameter count)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.25),
        )

        # Projection head: flattened CNN features → L2-normalised embedding
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embedding of shape [B, embed_dim]."""
        feat = self.encoder(x)
        emb  = self.projection(feat)
        return F.normalize(emb, dim=1)


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

    NUM_HARMONICS   = 4   # fundamental + 3 overtones
    BINS_PER_OCT    = 12  # ≈ mel bins per octave for n_mels=128, sr=22050

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Receives NUM_HARMONICS channels after stacking
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
          ch=0 → x[f]                 (fundamental view)
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
            shift = h * self.BINS_PER_OCT
            # Take bins [shift:] and pad the top with sentinel values
            pad   = sentinel.expand(B, C, shift, T)
            shifted = torch.cat([x[:, :, shift:, :], pad], dim=2)
            channels.append(shifted)
        return torch.cat(channels, dim=1)  # [B, NUM_HARMONICS, F, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._harmonic_stack(x)   # [B, 4, 128, 130]
        x = self.block1(x)            # [B, 32,  64,  65]
        x = self.block2(x)            # [B, 64,  32,  32]
        x = self.block3(x)            # [B, 128,  4,   4]
        return self.classifier(x)     # [B, 11]


class HarmonicSeriesCNN(nn.Module):
    """
    True harmonic series CNN.

    HarmonicCNN uses octave-only shifts (f, 2f, 4f, 8f) with BINS_PER_OCT=12,
    which does not align to actual mel-bin octaves (mel-octave ≈ 31 bins for
    n_mels=128, sr=22050).

    This model uses the true harmonic series f, 2f, 3f, 4f with correct
    mel-bin offsets computed as round(log2(n) * BINS_PER_OCT):

      n=1  → shift   0  (fundamental)
      n=2  → shift  31  (octave above)
      n=3  → shift  50  (octave + perfect 5th)
      n=4  → shift  63  (two octaves)

    Musical motivation:
      • clarinet (cla)  — cylindrical bore suppresses even harmonics →
                          3f and 5f are disproportionately strong vs 2f and 4f
      • trumpet  (tru)  — all harmonics strong, 3f/4f especially bright
      • flute    (flu)  — weak overtones overall, strong fundamental
      • cello    (cel)  — rich 2f/3f, dark timbre

    Seeing 3f explicitly (shift 50) vs the previous shift-of-24 (≈ tritone)
    gives the first conv direct access to the most musically relevant overtones.
    """

    # mel-bin offsets for harmonics 1x, 2x, 3x, 4x
    # derived as round(log10(n) * 2595 / (mel_max / n_mels))
    # mel_max ≈ 3177 for sr=22050, n_mels=128  →  mels/bin ≈ 24.8
    HARMONIC_SHIFTS = [0, 31, 50, 63]

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        n_ch = len(self.HARMONIC_SHIFTS)   # 4

        self.block1 = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=3, padding=1),
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

    def _harmonic_stack(self, x: torch.Tensor) -> torch.Tensor:
        """Stack mel spectrogram shifted to each harmonic frequency."""
        B, C, F, T = x.shape
        sentinel = torch.full((B, C, 1, T), -20.7, device=x.device, dtype=x.dtype)
        channels = []
        for shift in self.HARMONIC_SHIFTS:
            if shift == 0:
                channels.append(x)
            else:
                pad     = sentinel.expand(B, C, shift, T)
                shifted = torch.cat([x[:, :, shift:, :], pad], dim=2)
                channels.append(shifted)
        return torch.cat(channels, dim=1)   # [B, 4, F, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._harmonic_stack(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


class HarmonicProductCNN(nn.Module):
    """
    Harmonic Product Spectrum CNN.

    Extends HarmonicSeriesCNN with a 5th channel: the Harmonic Product Spectrum.

    HPS intuition: for a truly harmonic sound at fundamental f0, energy exists
    simultaneously at f0, 2f0, 3f0, 4f0.  Summing the log-mel channels
    (= log of the amplitude product) gives a strong positive response only
    where ALL harmonics are present at once — a direct "harmonicity" detector.

    This is the channel that explicitly tells the model:
      "there is a pitched instrument playing here"
    vs regions dominated by noise or inharmonic content.

    Input channels (5):
      ch 0  : mel at f          (fundamental)
      ch 1  : mel at 2f (+31)
      ch 2  : mel at 3f (+50)
      ch 3  : mel at 4f (+63)
      ch 4  : ch0+ch1+ch2+ch3  (HPS — log-domain product of all four)
    """

    HARMONIC_SHIFTS = [0, 31, 50, 63]

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        n_ch = len(self.HARMONIC_SHIFTS) + 1   # 4 harmonic + 1 HPS = 5

        self.block1 = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=3, padding=1),
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

    def _stack_with_hps(self, x: torch.Tensor) -> torch.Tensor:
        """Build [B, 5, F, T]: 4 harmonic channels + HPS channel."""
        B, C, F, T = x.shape
        sentinel = torch.full((B, C, 1, T), -20.7, device=x.device, dtype=x.dtype)
        channels = []
        for shift in self.HARMONIC_SHIFTS:
            if shift == 0:
                channels.append(x)
            else:
                pad     = sentinel.expand(B, C, shift, T)
                shifted = torch.cat([x[:, :, shift:, :], pad], dim=2)
                channels.append(shifted)
        stacked = torch.cat(channels, dim=1)            # [B, 4, F, T]
        hps     = stacked.sum(dim=1, keepdim=True)     # [B, 1, F, T]  log-domain product
        return torch.cat([stacked, hps], dim=1)         # [B, 5, F, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._stack_with_hps(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


class CQTHarmonicCNN(nn.Module):
    """
    Harmonic CNN on Constant-Q Transform input.

    Unlike mel, CQT has exactly CQT_BINS_PER_OCT=24 bins per octave by
    construction, so harmonic shifts are exact integers independent of pitch:

      2f → +24 bins  (1 octave)
      3f → +38 bins  (1 oct + P5,  since log2(3)×24 = 38.04 ≈ 38)
      4f → +48 bins  (2 octaves)

    Input: [B, 1, 168, 130]  (CQT_N_BINS × TIME_FRAMES)
    Stack: [B, 4, 168, 130]  after harmonic stacking

    Compared to HarmonicCNN (mel, BINS_PER_OCT=12 ≈ wrong):
      • shifts are musically exact at every pitch
      • 3f channel is truly the 3rd harmonic, not an approximation
      • higher frequency resolution (24 bins/oct vs ~31 mel bins/oct effective,
        but consistent across the spectrum)

    Same CNN backbone as HarmonicCNN — AdaptiveAvgPool handles the larger
    168-bin frequency axis without any architectural change.
    """

    # Exact harmonic shifts for CQT with 24 bins/octave
    # derived as round(log2(n) * CQT_BINS_PER_OCT)
    HARMONIC_SHIFTS = [0, 24, 38, 48]   # f, 2f, 3f, 4f

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        n_ch = len(self.HARMONIC_SHIFTS)   # 4

        self.block1 = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=3, padding=1),
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

    def _harmonic_stack(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        sentinel = torch.full((B, C, 1, T), -20.7, device=x.device, dtype=x.dtype)
        channels = []
        for shift in self.HARMONIC_SHIFTS:
            if shift == 0:
                channels.append(x)
            else:
                pad     = sentinel.expand(B, C, shift, T)
                shifted = torch.cat([x[:, :, shift:, :], pad], dim=2)
                channels.append(shifted)
        return torch.cat(channels, dim=1)   # [B, 4, F, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._harmonic_stack(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


class HPSSCombinedCNN(nn.Module):
    """
    HPSS-aware harmonic CNN.

    Input: [B, 2, 128, 130] — (harmonic_log_mel, percussive_log_mel).

    The harmonic component is processed through the same octave-shift stacking
    as HarmonicCNN (4 channels).  The percussive component is kept as-is (1 channel)
    and concatenated alongside — giving the model simultaneous access to:

      ch 0–3 : harmonic stack of the HPSS-filtered tonal component
               (clean overtone structure, background noise removed)
      ch 4   : percussive component
               (attack envelope, onset sharpness — distinguishes piano/gac
                from org/vio even when the tonal content looks similar)

    This lets the model reason about both the steady-state timbre AND the
    transient attack independently, without mixing them in the input.
    """

    NUM_HARMONICS = 4
    BINS_PER_OCT  = 12
    IN_CHANNELS   = 5   # 4 harmonic stack + 1 percussive

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
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _harmonic_stack(self, x: torch.Tensor) -> torch.Tensor:
        """Stack mel at 0/1/2/3 octave shifts. x: [B, 1, F, T] → [B, 4, F, T]."""
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
        # x: [B, 2, 128, 130]
        harm_mel = x[:, 0:1, :, :]   # [B, 1, 128, 130]  HPSS harmonic
        perc_mel = x[:, 1:2, :, :]   # [B, 1, 128, 130]  HPSS percussive

        harm_stack = self._harmonic_stack(harm_mel)          # [B, 4, 128, 130]
        x = torch.cat([harm_stack, perc_mel], dim=1)         # [B, 5, 128, 130]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


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
        """Identical to HarmonicCNN._harmonic_stack."""
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


class LearnableHarmonicCNN(nn.Module):
    """
    HarmonicCNN with backprop-learned harmonic shift positions.

    HarmonicCNN uses fixed shifts [0, 12, 24, 36] mel bins (octave spacing).
    This model keeps channel 0 at shift=0 (fundamental) and learns channels 1–3
    from an initialization of [12, 24, 36], letting the network discover which
    harmonic offsets are actually most useful for instrument discrimination.

    Differentiable shift via linear interpolation:
      For a fractional shift s = s_int + alpha (alpha ∈ [0, 1)):
        out_f = (1-alpha) * x[f + s_int] + alpha * x[f + s_int + 1]

      Gradient flows through alpha: d(out)/d(s) = x[ceil(s)] − x[floor(s)],
      which tells the network "shift up if higher-bin energy is more useful."
      The integer part is non-differentiable but updates discretely as s crosses
      bin boundaries during training.

    After training, inspect .learned_shifts to see where the model converged.
    """

    NUM_HARMONICS = 4
    MAX_SHIFT     = 64  # hard clamp — allows up to ~2 octaves above init

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ch 0: fundamental, fixed at shift=0
        # ch 1–3: learnable, init to 1 / 2 / 3 octaves (same as HarmonicCNN)
        self.learned_shifts = nn.Parameter(torch.tensor([12.0, 24.0, 36.0]))

        self.block1 = nn.Sequential(
            nn.Conv2d(self.NUM_HARMONICS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes),
        )

    def _shift_channel(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Differentiable fractional frequency shift along the mel-bin axis.

        x: [B, 1, F, T] — input mel spectrogram
        s: scalar Parameter — shift in mel bins (may be fractional)
        Returns [B, 1, F, T] with bins [s:] interpolated to position [0:].
        """
        B, C, F, T = x.shape
        s_clamped = s.clamp(0.0, self.MAX_SHIFT)
        s_int  = int(s_clamped.detach().item())   # integer part — used for slicing
        alpha  = s_clamped - s_clamped.detach().floor()  # fractional part — differentiable

        F_out = F - s_int - 1
        if F_out <= 0:
            return torch.full_like(x, -20.7)

        # Linear interpolation between adjacent frequency slices
        lo = x[:, :, s_int    :s_int     + F_out, :]   # [B, 1, F_out, T]
        hi = x[:, :, s_int + 1:s_int + 1 + F_out, :]   # [B, 1, F_out, T]
        interpolated = (1.0 - alpha) * lo + alpha * hi   # [B, 1, F_out, T]

        # Pad high-frequency end with log-zero sentinel to restore shape [B,1,F,T]
        pad = torch.full((B, C, s_int + 1, T), -20.7, device=x.device, dtype=x.dtype)
        return torch.cat([interpolated, pad], dim=2)

    def _harmonic_stack(self, x: torch.Tensor) -> torch.Tensor:
        channels = [x]  # ch 0: fundamental, shift = 0 (exact, no interpolation)
        for s in self.learned_shifts:
            channels.append(self._shift_channel(x, s))
        return torch.cat(channels, dim=1)   # [B, 4, F, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._harmonic_stack(x)   # [B, 4, 128, 130]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


class HarmonicSECNN(nn.Module):
    """
    HarmonicCNN with Squeeze-and-Excitation attention over harmonic channels.

    After harmonic stacking, a lightweight SE block learns per-channel weights:
      1. Squeeze  — global average pool over (freq, time) → [B, 4] descriptor
      2. Excitation — two FC layers learn which harmonic channels matter for
                      this particular input:
                        [B, 4] → ReLU → [B, 4] → Sigmoid → channel weights ∈ (0,1)
      3. Scale    — multiply each harmonic channel by its learned weight

    Musical intuition:
      • Flute / voi : weak overtones → model should upweight ch0 (fundamental)
      • Organ       : all harmonics equally strong → flat weights
      • Clarinet    : odd harmonics dominant → ch0 and ch2 (1f, 3f) upweighted
      • Trumpet     : 3f/4f especially bright → ch2/ch3 upweighted

    The SE block adds only 40 extra parameters (two 4×4 linear layers) on top
    of the full HarmonicCNN (621k params), so the comparison is essentially
    fair — any improvement comes purely from the attention mechanism.
    """

    NUM_HARMONICS = 4
    BINS_PER_OCT  = 12

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # SE excitation: 4 → 4 → 4 (no bottleneck — only 4 channels to begin with)
        self.harmonic_se = nn.Sequential(
            nn.Linear(self.NUM_HARMONICS, self.NUM_HARMONICS),
            nn.ReLU(inplace=True),
            nn.Linear(self.NUM_HARMONICS, self.NUM_HARMONICS),
            nn.Sigmoid(),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(self.NUM_HARMONICS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes),
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
        x = self._harmonic_stack(x)          # [B, 4, F, T]

        # Squeeze: global average pool → [B, 4]
        gap = x.mean(dim=[-2, -1])
        # Excitation: per-channel sigmoid weights → [B, 4, 1, 1]
        weights = self.harmonic_se(gap).unsqueeze(-1).unsqueeze(-1)
        x = x * weights                      # [B, 4, F, T]  channel-scaled

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


class OddEvenHarmonicCNN(nn.Module):
    """
    Dual-branch CNN that separates odd and even harmonic series.

    Acoustic motivation:
      Cylindrical-bore instruments (clarinet) with one closed end act as
      quarter-wave resonators, selectively suppressing even harmonics:
        clarinet:  strong 1f, 3f, 5f  /  weak 2f, 4f, 6f
        all others: all harmonics present

    By routing odd and even harmonics through independent branches, the network
    can learn detectors that fire specifically on the odd-only or mixed patterns,
    making the clarinet/non-clarinet distinction trivially representable.

    Mel-bin offsets for the harmonic series (n_mels=128, sr=22050):
      1f →  +0 bins   (fundamental — odd)
      2f → +31 bins   (even)
      3f → +50 bins   (odd)
      4f → +63 bins   (even)

    Each branch uses a 3-block CNN producing a 1024-dim feature vector.
    Both vectors are concatenated (2048-dim) before the shared classifier,
    matching HarmonicCNN's classifier input size.
    """

    ODD_SHIFTS  = [0,  50]   # mel-bin offsets for f (1st) and 3f (3rd)
    EVEN_SHIFTS = [31, 63]   # mel-bin offsets for 2f (2nd) and 4f (4th)

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def _branch(n_ch: int) -> nn.Sequential:
            # Identical topology to HarmonicCNN but with 64 (not 128) in block3
            # → 64*4*4 = 1024 features per branch, 2048 total (same as HarmonicCNN)
            return nn.Sequential(
                nn.Conv2d(n_ch, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)), nn.Dropout2d(0.25),
                nn.Flatten(),   # 64 * 4 * 4 = 1024
            )

        self.odd_branch  = _branch(len(self.ODD_SHIFTS))    # 2-ch input
        self.even_branch = _branch(len(self.EVEN_SHIFTS))   # 2-ch input

        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2, 256),   # 2048 → 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _stack(x: torch.Tensor, shifts: list) -> torch.Tensor:
        """Stack mel spectrogram shifted to each harmonic. [B,1,F,T] → [B,n,F,T]."""
        B, C, F, T = x.shape
        sentinel = torch.full((B, C, 1, T), -20.7, device=x.device, dtype=x.dtype)
        channels = []
        for s in shifts:
            if s == 0:
                channels.append(x)
            else:
                pad = sentinel.expand(B, C, s, T)
                channels.append(torch.cat([x[:, :, s:, :], pad], dim=2))
        return torch.cat(channels, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        odd_stack  = self._stack(x, self.ODD_SHIFTS)    # [B, 2, F, T]
        even_stack = self._stack(x, self.EVEN_SHIFTS)   # [B, 2, F, T]

        h_odd  = self.odd_branch(odd_stack)    # [B, 1024]
        h_even = self.even_branch(even_stack)  # [B, 1024]

        return self.classifier(torch.cat([h_odd, h_even], dim=1))  # [B, num_classes]


class InstrumentCNNBiLSTM(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, lstm_hidden: int = 64):
        super().__init__()

        # ── CNN block ────────────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # [16, 64, 65]

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # [32, 32, 32]

            nn.Dropout2d(0.25),
        )

        # After CNN: [batch, 32, freq', time']
        # freq' = 128 // 4 = 32,  time' = 130 // 4 = 32
        self._cnn_freq = 32
        self._cnn_time = 32
        self._lstm_input_size = 32 * self._cnn_freq  # 1024

        # ── BiLSTM ───────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self._lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.3)

        # ── Classifier ───────────────────────────────────────────────────────
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)  # 128 → K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 128, 130]
        x = self.cnn(x)                          # [B, 32, 32, 32]

        B, C, F, T = x.shape
        # Reshape to sequence: [B, T, C*F]
        x = x.permute(0, 3, 1, 2)               # [B, T, C, F]
        x = x.reshape(B, T, C * F)              # [B, T, 1024]

        x, _ = self.lstm(x)                      # [B, T, 128]
        # Pool over time with mean
        x = x.mean(dim=1)                        # [B, 128]
        x = self.dropout(x)
        return self.fc(x)                         # [B, K]  — raw logits
