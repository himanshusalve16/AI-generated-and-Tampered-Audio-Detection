"""
Dual-model loader and ensemble inference for audio deepfake detection.

Manages two models:
  1. ResNet-18  — classifies mel-spectrogram images (1, 224, 224)
  2. LSTM       — classifies temporal mel sequences  (1, T, N_MELS)

At startup both .pth files are loaded into memory. The predict() function
runs both models, applies softmax, and computes a weighted ensemble.

Class index convention:
  0 → Real
  1 → AI Generated
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18

from config import (
    CLASS_LABELS,
    LSTM_HIDDEN_DIM,
    LSTM_INPUT_DIM,
    LSTM_NUM_CLASSES,
    LSTM_NUM_LAYERS,
    LSTM_WEIGHT,
    RESNET_NUM_CLASSES,
    RESNET_WEIGHT,
)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Singleton model instances
# ---------------------------------------------------------------------------

_RESNET_MODEL: Optional[nn.Module] = None
_LSTM_MODEL: Optional[nn.Module] = None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EnsemblePrediction:
    """All prediction outputs needed by the API response."""
    resnet_prediction: str
    resnet_confidence: float
    lstm_prediction: str
    lstm_confidence: float
    ensemble_prediction: str
    ensemble_confidence: float


# ---------------------------------------------------------------------------
# Architecture builders
# ---------------------------------------------------------------------------

def _build_resnet() -> nn.Module:
    """ResNet-18: 1-channel input, 2-class output."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, RESNET_NUM_CLASSES)
    return model


class _AudioLSTM(nn.Module):
    """Bidirectional LSTM matching training/model.py AudioLSTM."""

    def __init__(
        self,
        input_dim: int = LSTM_INPUT_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        num_layers: int = LSTM_NUM_LAYERS,
        num_classes: int = LSTM_NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        return self.classifier(last)


# ---------------------------------------------------------------------------
# State-dict cleaning (handles "model." prefix from training wrapper)
# ---------------------------------------------------------------------------

def _clean_state_dict(raw: dict) -> dict:
    """Strip optional 'model.' prefix from keys."""
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    cleaned = {}
    for k, v in raw.items():
        cleaned[k[len("model."):] if k.startswith("model.") else k] = v
    return cleaned


# ---------------------------------------------------------------------------
# Load helpers (called once at startup)
# ---------------------------------------------------------------------------

def load_resnet(path: str) -> nn.Module:
    """Load and cache the ResNet model from a .pth checkpoint."""
    global _RESNET_MODEL
    if _RESNET_MODEL is not None:
        return _RESNET_MODEL

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ResNet weights not found at {path}")

    checkpoint = torch.load(p, map_location=DEVICE)
    state_dict = _clean_state_dict(checkpoint)

    model = _build_resnet()
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    _RESNET_MODEL = model
    return _RESNET_MODEL


def load_lstm(path: str) -> nn.Module:
    """Load and cache the LSTM model from a .pth checkpoint."""
    global _LSTM_MODEL
    if _LSTM_MODEL is not None:
        return _LSTM_MODEL

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"LSTM weights not found at {path}")

    checkpoint = torch.load(p, map_location=DEVICE)
    state_dict = _clean_state_dict(checkpoint)

    model = _AudioLSTM()
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    _LSTM_MODEL = model
    return _LSTM_MODEL


def models_loaded() -> dict:
    """Return which models are currently loaded."""
    return {
        "resnet": _RESNET_MODEL is not None,
        "lstm": _LSTM_MODEL is not None,
    }


# ---------------------------------------------------------------------------
# Inference: dual-model ensemble
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict(
    resnet_tensor: torch.Tensor,
    lstm_tensor: torch.Tensor,
) -> EnsemblePrediction:
    """
    Run both models and compute the weighted ensemble prediction.

    Args:
        resnet_tensor: shape (1, 224, 224) — from preprocess_audio().
        lstm_tensor:   shape (1, T, N_MELS) — from preprocess_audio().

    Returns:
        EnsemblePrediction with individual and ensemble results.

    Raises:
        RuntimeError: If neither model is loaded.
    """
    resnet_probs = None
    lstm_probs = None

    # ----- ResNet inference -----
    if _RESNET_MODEL is not None:
        batch = resnet_tensor.unsqueeze(0).to(DEVICE)  # (1, 1, 224, 224)
        logits = _RESNET_MODEL(batch)
        resnet_probs = torch.softmax(logits, dim=1).squeeze(0)  # (2,)

    # ----- LSTM inference -----
    if _LSTM_MODEL is not None:
        batch = lstm_tensor.to(DEVICE)  # already (1, T, N_MELS)
        logits = _LSTM_MODEL(batch)
        lstm_probs = torch.softmax(logits, dim=1).squeeze(0)  # (2,)

    # ----- Require at least one model -----
    if resnet_probs is None and lstm_probs is None:
        raise RuntimeError(
            "No models loaded. Call load_resnet() and/or load_lstm() at startup."
        )

    # ----- Individual predictions -----
    def _label_and_conf(probs):
        conf, idx = torch.max(probs, dim=0)
        return CLASS_LABELS[int(idx.item())], round(float(conf.item()), 4)

    if resnet_probs is not None:
        resnet_label, resnet_conf = _label_and_conf(resnet_probs)
    else:
        resnet_label, resnet_conf = "N/A", 0.0

    if lstm_probs is not None:
        lstm_label, lstm_conf = _label_and_conf(lstm_probs)
    else:
        lstm_label, lstm_conf = "N/A", 0.0

    # ----- Ensemble (weighted average of probability vectors) -----
    if resnet_probs is not None and lstm_probs is not None:
        ensemble_probs = RESNET_WEIGHT * resnet_probs + LSTM_WEIGHT * lstm_probs
    elif resnet_probs is not None:
        ensemble_probs = resnet_probs
    else:
        ensemble_probs = lstm_probs

    ensemble_label, ensemble_conf = _label_and_conf(ensemble_probs)

    return EnsemblePrediction(
        resnet_prediction=resnet_label,
        resnet_confidence=resnet_conf,
        lstm_prediction=lstm_label,
        lstm_confidence=lstm_conf,
        ensemble_prediction=ensemble_label,
        ensemble_confidence=ensemble_conf,
    )
