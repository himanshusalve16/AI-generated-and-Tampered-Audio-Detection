"""
ResNet-18 model loader for audio deepfake detection.

Provides a modified ResNet-18 CNN that accepts single-channel mel spectrograms
(shape 1x224x224) and outputs two-class logits (Real vs AI Generated). Weights
are loaded from disk once at server startup; inference is then performed in
memory with optional GPU acceleration.

Class index convention:
  0 -> Real
  1 -> AI Generated
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet18

# -----------------------------------------------------------------------------
# Device and singleton model
# -----------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Single global model instance, set by load_model() at startup. Used by predict().
_MODEL: nn.Module | None = None

# Class labels for output (index 0 and 1)
CLASS_LABELS = ("Real", "AI Generated")


# -----------------------------------------------------------------------------
# Model architecture: ResNet-18 for 1-channel input, 2 classes
# -----------------------------------------------------------------------------

def _build_model() -> nn.Module:
    """
    Construct a ResNet-18 with architecture changes for our task:

    1. First convolution: accept 1 channel (mel spectrogram) instead of 3 (RGB).
    2. Final fully connected layer: output 2 logits (Real, AI Generated) instead of 1000.
    """
    model = resnet18(weights=None)

    # Replace the first conv layer (in_channels=3 -> 1)
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    # Replace the classifier head
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    return model


# -----------------------------------------------------------------------------
# Load model from checkpoint (call once at startup)
# -----------------------------------------------------------------------------

def load_model(path: str) -> nn.Module:
    """
    Load the trained model weights from a .pth file and cache the model
    in memory for subsequent inference.

    Args:
        path: Absolute or relative path to the state_dict file (e.g. audio_model.pth).

    Returns:
        The loaded model in eval mode, on the appropriate device (CPU/GPU).

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model weights not found at {path}")

    model = _build_model()
    state_dict = torch.load(path_obj, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    _MODEL = model
    return _MODEL


# -----------------------------------------------------------------------------
# Inference: tensor -> label and confidence
# -----------------------------------------------------------------------------

@torch.inference_mode()
def predict(tensor: torch.Tensor) -> Tuple[str, float]:
    """
    Run a single-sample forward pass and return the predicted class label
    and confidence (probability of the predicted class).

    Steps:
      1. Add batch dimension (1, 1, 224, 224).
      2. Move to device and run forward pass.
      3. Apply softmax to logits to get probabilities.
      4. Take argmax for class index and max probability for confidence.
      5. Map class index to string label and return.

    Args:
        tensor: Preprocessed mel spectrogram of shape (1, 224, 224), on CPU.

    Returns:
        (label, confidence) where label is "Real" or "AI Generated" and
        confidence is a float in [0, 1].

    Raises:
        RuntimeError: If the model has not been loaded (load_model not called).
        ValueError: If the tensor shape is incorrect.
    """
    if _MODEL is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")

    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor (1, 224, 224), got ndim={tensor.ndim}")
    if tensor.shape[0] != 1 or tensor.shape[1] != 224 or tensor.shape[2] != 224:
        raise ValueError(f"Expected tensor shape (1, 224, 224), got {tuple(tensor.shape)}")

    # Add batch dimension and move to device
    batch = tensor.unsqueeze(0).to(DEVICE)

    logits = _MODEL(batch)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    confidence, idx = torch.max(probs, dim=0)
    confidence_val = float(confidence.item())
    class_idx = int(idx.item())

    label = CLASS_LABELS[class_idx]
    return label, confidence_val
