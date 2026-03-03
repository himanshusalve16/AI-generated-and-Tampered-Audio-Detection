from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet18


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / ".." / "models" / "audio_model.pth").resolve()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL: nn.Module | None = None


def _build_model() -> nn.Module:
  """
  Create a ResNet-18 model adapted for:
  - Single-channel (1-channel) input.
  - Binary classification (2 output classes).
  """
  model = resnet18(weights=None)

  # Replace first conv layer so it accepts 1-channel input instead of RGB (3-channel).
  model.conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False,
  )

  # Replace final fully connected layer with 2 output units (Real vs AI-Generated).
  num_features = model.fc.in_features
  model.fc = nn.Linear(num_features, 2)

  return model


def load_model() -> nn.Module:
  """
  Load the trained model weights from disk and keep a single shared instance.

  This should be called once at application startup so that the model is not
  reloaded on every request.
  """
  global _MODEL
  if _MODEL is not None:
    return _MODEL

  model = _build_model()

  if not MODEL_PATH.exists():
    # In a real deployment we might want to fail hard here. For now, raise a clear error.
    raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

  state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
  model.load_state_dict(state_dict)
  model.to(DEVICE)
  model.eval()

  _MODEL = model
  return _MODEL


@torch.inference_mode()
def predict(tensor: torch.Tensor) -> Tuple[str, float]:
  """
  Run a forward pass through the model and convert logits into:
  - A human-readable label ("Real" or "AI-Generated").
  - A confidence score (float in [0, 1]).

  The input tensor is expected to have shape (1, 224, 224).
  """
  if tensor.ndim != 3 or tensor.shape[0] != 1:
    raise ValueError(f"Expected tensor shape (1, 224, 224), got {tuple(tensor.shape)}")

  model = load_model()

  # Add batch dimension: (1, 1, 224, 224).
  batch = tensor.unsqueeze(0).to(DEVICE)

  logits = model(batch)
  probs = torch.softmax(logits, dim=1).squeeze(0)  # (2,)

  confidence, idx = torch.max(probs, dim=0)
  confidence_value = float(confidence.item())

  label = "Real" if int(idx.item()) == 0 else "AI-Generated"

  return label, confidence_value

