"""
Audio preprocessing module for the Audio Deepfake Detection pipeline.

Returns three outputs from a single audio file:
  1. ResNet tensor   — shape (1, 224, 224) mel-spectrogram image
  2. LSTM tensor     — shape (1, T, N_MELS) sequential mel-spectrogram
  3. Spectrogram PNG — base64-encoded PNG string for frontend display

Steps performed:
  1. Load audio with librosa
  2. Resample to 16 000 Hz mono
  3. Trim leading/trailing silence
  4. Pad or crop to exactly AUDIO_DURATION_SEC seconds
  5. Compute mel spectrogram → log dB → normalize
  6. Resize to 224×224 for ResNet
  7. Keep original (T, N_MELS) sequence for LSTM
  8. Render spectrogram to a base64 PNG for the frontend
"""

import base64
import io
from dataclasses import dataclass
from typing import BinaryIO, Union

import librosa
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no GUI window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    AUDIO_DURATION_SEC,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    NUM_SAMPLES,
    POWER,
    RESNET_INPUT_SIZE,
    SAMPLE_RATE,
    TRIM_TOP_DB,
)

# -----------------------------------------------------------------------------
# Data class for preprocessed outputs
# -----------------------------------------------------------------------------

@dataclass
class PreprocessedAudio:
    """Container for all three outputs of the preprocessing pipeline."""
    resnet_tensor: torch.Tensor      # (1, 224, 224)
    lstm_tensor: torch.Tensor        # (1, T, N_MELS)
    spectrogram_base64: str          # base64-encoded PNG image


# -----------------------------------------------------------------------------
# Internal helper functions
# -----------------------------------------------------------------------------

def _trim_silence(y: np.ndarray) -> np.ndarray:
    """Trim leading and trailing silence from the waveform."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    return y_trimmed


def _pad_or_crop(y: np.ndarray, target_length: int) -> np.ndarray:
    """
    Ensure the waveform has exactly *target_length* samples.
    Shorter → zero-pad at the end.  Longer → take a centered segment.
    """
    n = len(y)
    if n < target_length:
        y = np.pad(y, (0, target_length - n), mode="constant", constant_values=0.0)
    elif n > target_length:
        start = (n - target_length) // 2
        y = y[start : start + target_length]
    return y


def _compute_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """Compute mel spectrogram from waveform (power scale)."""
    return librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=POWER,
    )


def _to_log_scale(mel_spec: np.ndarray) -> np.ndarray:
    """Convert power spectrogram to log scale (decibels)."""
    return librosa.power_to_db(mel_spec, ref=np.max)


def _normalize(mel_db: np.ndarray) -> np.ndarray:
    """Normalize to zero mean and unit variance (per spectrogram)."""
    mean = float(np.mean(mel_db))
    std = float(np.std(mel_db))
    if std <= 0:
        std = 1.0
    return (mel_db - mean) / std


def _resize_tensor(tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Resize a (1, H, W) tensor to (1, h, w) with bilinear interpolation."""
    return F.interpolate(
        tensor.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze(0)


def _spectrogram_to_base64(mel_db: np.ndarray) -> str:
    """
    Render the log-scale mel spectrogram as a PNG and return a
    base64-encoded data-URI string ready for <img src="...">.
    """
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    img = librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram", fontsize=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# -----------------------------------------------------------------------------
# Core pipeline: waveform → PreprocessedAudio
# -----------------------------------------------------------------------------

def _waveform_to_preprocessed(y: np.ndarray) -> PreprocessedAudio:
    """
    Convert a mono waveform (already at SAMPLE_RATE) into all
    three outputs needed by the dual-model backend.
    """
    y = _trim_silence(y)
    y = _pad_or_crop(y, NUM_SAMPLES)

    # Mel spectrogram (shape: N_MELS × T)
    mel_spec = _compute_mel_spectrogram(y)
    mel_db = _to_log_scale(mel_spec)
    mel_norm = _normalize(mel_db)

    # --- ResNet tensor: (1, 224, 224) ---
    resnet_tensor = torch.from_numpy(mel_norm).float().unsqueeze(0)  # (1, N_MELS, T)
    resnet_tensor = _resize_tensor(resnet_tensor, RESNET_INPUT_SIZE, RESNET_INPUT_SIZE)

    # --- LSTM tensor: (1, T, N_MELS) ---
    # mel_norm is (N_MELS, T) → transpose to (T, N_MELS), add batch dim
    lstm_seq = mel_norm.T  # (T, N_MELS)
    lstm_tensor = torch.from_numpy(lstm_seq).float().unsqueeze(0)  # (1, T, N_MELS)

    # --- Spectrogram image for frontend ---
    spec_b64 = _spectrogram_to_base64(mel_db)

    return PreprocessedAudio(
        resnet_tensor=resnet_tensor,
        lstm_tensor=lstm_tensor,
        spectrogram_base64=spec_b64,
    )


# -----------------------------------------------------------------------------
# Public API — accepts file-like, bytes, or path
# -----------------------------------------------------------------------------

def preprocess_audio(file: Union[BinaryIO, bytes, str]) -> PreprocessedAudio:
    """
    Preprocess an audio source and return a PreprocessedAudio bundle.

    Args:
        file: A binary file-like object, raw bytes, or a file path string.

    Returns:
        PreprocessedAudio with resnet_tensor, lstm_tensor, and
        spectrogram_base64.

    Raises:
        ValueError: If the audio cannot be loaded.
    """
    if isinstance(file, bytes):
        file = io.BytesIO(file)
    elif isinstance(file, str):
        y, _ = librosa.load(file, sr=SAMPLE_RATE, mono=True)
        return _waveform_to_preprocessed(y)

    # File-like object
    raw = file.read() if hasattr(file, "read") else file
    if isinstance(raw, bytes):
        file = io.BytesIO(raw)
    y, _ = librosa.load(file, sr=SAMPLE_RATE, mono=True)
    return _waveform_to_preprocessed(y)
