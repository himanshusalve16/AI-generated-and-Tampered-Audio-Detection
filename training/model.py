"""
Model definitions for audio deepfake detection (training).

Contains two architectures:
  1. AudioResNet  — ResNet-18 adapted for 1-channel mel-spectrogram images (224×224).
  2. AudioLSTM    — Bidirectional LSTM for temporal mel-spectrogram sequences.

Both output 2 logits: index 0 → Real, index 1 → AI Generated.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


# ---------------------------------------------------------------------------
# 1. ResNet-18 for spectrogram images
# ---------------------------------------------------------------------------

class AudioResNet(nn.Module):
    """
    ResNet-18 adapted for single-channel mel-spectrogram input and binary
    classification (Real vs AI Generated).

    Modifications from standard ImageNet ResNet-18:
      - conv1: 1 input channel instead of 3
      - fc:    2 output classes instead of 1000
    """

    def __init__(self) -> None:
        super().__init__()
        base = resnet18(weights=None)

        # Accept 1-channel input (mel spectrogram) instead of 3-channel RGB
        base.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Replace classifier head with 2 classes
        num_features = base.fc.in_features
        base.fc = nn.Linear(num_features, 2)

        self.model = base

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# 2. Bidirectional LSTM for temporal sequences
# ---------------------------------------------------------------------------

class AudioLSTM(nn.Module):
    """
    Bidirectional LSTM for classifying temporal mel-spectrogram sequences.

    Input shape:  (batch, seq_len, input_dim)   e.g. (B, 94, 128)
    Output shape: (batch, num_classes)           e.g. (B, 2)

    Architecture:
      - Multi-layer bidirectional LSTM
      - Dropout between LSTM layers
      - Final hidden states (forward + backward) concatenated
      - Passed through a small FC head for classification
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Bidirectional → output is 2 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, 2 * hidden_dim)
        # Take the last time step's output
        last = lstm_out[:, -1, :]           # (batch, 2 * hidden_dim)
        logits = self.classifier(last)      # (batch, num_classes)
        return logits
