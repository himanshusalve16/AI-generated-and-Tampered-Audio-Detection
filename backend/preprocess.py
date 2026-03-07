"""
Audio preprocessing module for the Audio Deepfake Detection pipeline.

Converts raw audio (file or bytes) into a normalized mel-spectrogram tensor
of shape (1, 224, 224) suitable for the ResNet-18 model. Preprocessing steps
are identical to the training pipeline so that inference and training see
the same feature distribution.

Steps performed:
  1. Load audio using librosa (supports WAV, MP3, FLAC, etc.)
  2. Resample to 16000 Hz (single standard for all inputs)
  3. Trim leading and trailing silence
  4. Pad or crop to exactly 3 seconds
  5. Compute mel spectrogram
  6. Convert to log scale (decibels)
  7. Normalize (zero mean, unit variance per spectrogram)
  8. Resize to 224x224 (spatial size expected by ResNet-18)
  9. Convert to PyTorch tensor
"""

import io
from typing import BinaryIO, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Preprocessing constants (must match training/dataset.py)
# -----------------------------------------------------------------------------

# Target sample rate in Hz. 16 kHz is common for speech and keeps computation manageable.
TARGET_SR = 16000

# Fixed duration in seconds. All clips are normalized to this length.
TARGET_DURATION_SEC = 3.0

# Number of samples corresponding to TARGET_DURATION_SEC at TARGET_SR.
TARGET_NUM_SAMPLES = int(TARGET_SR * TARGET_DURATION_SEC)

# Output spatial size (height and width) for the spectrogram image. ResNet-18 typically expects 224x224.
TARGET_SIZE = 224

# Mel spectrogram parameters (same as in training for consistency)
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
POWER = 2.0

# Silence trimming: frames with amplitude below this dB relative to max are trimmed.
TRIM_TOP_DB = 20


# -----------------------------------------------------------------------------
# Core preprocessing: waveform -> tensor
# -----------------------------------------------------------------------------

def _trim_silence(y: np.ndarray) -> np.ndarray:
    """Trim leading and trailing silence from the waveform."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    return y_trimmed


def _pad_or_crop_to_fixed_length(y: np.ndarray, target_length: int) -> np.ndarray:
    """
    Ensure the waveform has exactly target_length samples.
    If shorter: zero-pad at the end. If longer: take a centered segment.
    """
    n = len(y)
    if n < target_length:
        pad_len = target_length - n
        y = np.pad(y, (0, pad_len), mode="constant", constant_values=0.0)
    elif n > target_length:
        start = (n - target_length) // 2
        y = y[start : start + target_length]
    return y


def _compute_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """Compute mel spectrogram from waveform (power scale)."""
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=POWER,
    )
    return mel_spec


def _to_log_scale(mel_spec: np.ndarray) -> np.ndarray:
    """Convert power spectrogram to log scale (decibels)."""
    return librosa.power_to_db(mel_spec, ref=np.max)


def _normalize_spectrogram(mel_db: np.ndarray) -> np.ndarray:
    """Normalize to zero mean and unit variance (per spectrogram)."""
    mean = float(np.mean(mel_db))
    std = float(np.std(mel_db))
    if std <= 0:
        std = 1.0
    return (mel_db - mean) / std


def _resize_to_target(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Resize a single-channel tensor (1, H, W) to (1, target_h, target_w)
    using bilinear interpolation.
    """
    # F.interpolate expects (N, C, H, W)
    tensor_4d = tensor.unsqueeze(0)
    resized = F.interpolate(
        tensor_4d,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def _waveform_to_tensor(y: np.ndarray) -> torch.Tensor:
    """
    Convert a mono waveform (already at TARGET_SR) to a tensor of shape (1, 224, 224).

    This function is the core of preprocessing and is shared conceptually with
    the training dataset so that train and inference use identical transforms.
    """
    y = _trim_silence(y)
    y = _pad_or_crop_to_fixed_length(y, TARGET_NUM_SAMPLES)

    mel_spec = _compute_mel_spectrogram(y)
    mel_db = _to_log_scale(mel_spec)
    mel_norm = _normalize_spectrogram(mel_db)

    # To tensor: (n_mels, time) -> (1, n_mels, time)
    tensor = torch.from_numpy(mel_norm).float().unsqueeze(0)
    tensor = _resize_to_target(tensor, TARGET_SIZE, TARGET_SIZE)
    return tensor  # (1, 224, 224)


# -----------------------------------------------------------------------------
# Public API: accept file-like, bytes, or file path
# -----------------------------------------------------------------------------

def preprocess_audio(file: Union[BinaryIO, bytes, str]) -> torch.Tensor:
    """
    Preprocess an audio source into a normalized mel-spectrogram tensor.

    Args:
        file: Either a binary file-like object (e.g. from FastAPI UploadFile.read()),
              raw bytes of an audio file, or a path (str) to an audio file on disk.

    Returns:
        Tensor of shape (1, 224, 224), dtype float32, on CPU. No batch dimension;
        the model loader will add the batch dimension for inference.

    Raises:
        ValueError: If the audio cannot be loaded or is invalid.
    """
    if isinstance(file, bytes):
        file = io.BytesIO(file)
    elif isinstance(file, str):
        y, _ = librosa.load(file, sr=TARGET_SR, mono=True)
        return _waveform_to_tensor(y)

    # File-like object: read content then load
    raw = file.read() if hasattr(file, "read") else file
    if isinstance(raw, bytes):
        file = io.BytesIO(raw)
    y, _ = librosa.load(file, sr=TARGET_SR, mono=True)
    return _waveform_to_tensor(y)
