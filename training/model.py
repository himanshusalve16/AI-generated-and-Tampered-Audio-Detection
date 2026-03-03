from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


class AudioResNet(nn.Module):
  """
  ResNet-18 adapted for:
  - 1-channel input (mel-spectrogram "images").
  - Binary classification (2 output classes).
  """

  def __init__(self) -> None:
    super().__init__()
    base = resnet18(weights=None)

    base.conv1 = nn.Conv2d(
      in_channels=1,
      out_channels=64,
      kernel_size=7,
      stride=2,
      padding=3,
      bias=False,
    )

    num_features = base.fc.in_features
    base.fc = nn.Linear(num_features, 2)

    self.model = base

  def forward(self, x):
    return self.model(x)

