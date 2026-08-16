"""
Models for multi-label instrument identification.

Input:  [batch, 1, 128, 130]   (channels, n_mels, time_frames)
Output: [batch, NUM_CLASSES]   raw logits (apply sigmoid for probabilities)

Architectures:
  InstrumentCNN         — baseline: CNN + Global Average Pooling + MLP
  InstrumentCNNBiLSTM   — CNN + BiLSTM temporal modelling
  HarmonicCNN           — domain-aware CNN that explicitly sees harmonic structure
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
