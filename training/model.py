"""
ResNet-18 model definition for audio deepfake detection (training).

The architecture is identical to the one used in the backend (model_loader.py)
so that state_dict saved here can be loaded directly for inference. Modifications
from the standard ImageNet ResNet-18:
  - First convolution: 1 input channel (mel spectrogram) instead of 3 (RGB).
  - Final fully connected layer: 2 output classes (Real, AI Generated) instead of 1000.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


class AudioResNet(nn.Module):
    """
    ResNet-18 adapted for single-channel mel-spectrogram input and binary
    classification (Real vs AI Generated).
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
