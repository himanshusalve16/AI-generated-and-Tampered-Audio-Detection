"""
PyTorch Dataset for real vs AI-generated audio classification.

Supports two return modes controlled by the *mode* parameter:
  - "resnet" → returns (tensor_1x224x224, label)   — mel-spectrogram image
  - "lstm"   → returns (tensor_TxN_MELS, label)     — temporal mel sequence

Directory structure expected:
  root_dir/
    real/   → label 0 (Real)
    fake/   → label 1 (AI Generated / Fake)

Preprocessing mirrors backend/preprocess.py exactly so that training and
inference see the same feature distribution.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Preprocessing constants (must match backend/config.py exactly)
# ---------------------------------------------------------------------------

TARGET_SR = 16_000
TARGET_DURATION_SEC = 3.0
TARGET_NUM_SAMPLES = int(TARGET_SR * TARGET_DURATION_SEC)
TARGET_SIZE = 224
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
POWER = 2.0
TRIM_TOP_DB = 20


# ---------------------------------------------------------------------------
# Shared waveform helpers
# ---------------------------------------------------------------------------

def _trim_silence(y: np.ndarray) -> np.ndarray:
    y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    return y_trimmed


def _pad_or_crop(y: np.ndarray, target_length: int) -> np.ndarray:
    n = len(y)
    if n < target_length:
        y = np.pad(y, (0, target_length - n), mode="constant", constant_values=0.0)
    elif n > target_length:
        start = (n - target_length) // 2
        y = y[start : start + target_length]
    return y


def _compute_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=POWER,
    )


def _to_log_scale(mel_spec: np.ndarray) -> np.ndarray:
    return librosa.power_to_db(mel_spec, ref=np.max)


def _normalize(mel_db: np.ndarray) -> np.ndarray:
    mean = float(np.mean(mel_db))
    std = float(np.std(mel_db))
    if std <= 0:
        std = 1.0
    return (mel_db - mean) / std


def _resize_tensor(tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return F.interpolate(
        tensor.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False,
    ).squeeze(0)


# ---------------------------------------------------------------------------
# Mode-specific waveform → tensor converters
# ---------------------------------------------------------------------------

def waveform_to_resnet_tensor(y: np.ndarray) -> torch.Tensor:
    """
    Convert a mono waveform at TARGET_SR to a (1, 224, 224) tensor.
    Used by ResNet training and kept identical to backend preprocessing.
    """
    y = _trim_silence(y)
    y = _pad_or_crop(y, TARGET_NUM_SAMPLES)
    mel_spec = _compute_mel_spectrogram(y)
    mel_db = _to_log_scale(mel_spec)
    mel_norm = _normalize(mel_db)
    tensor = torch.from_numpy(mel_norm).float().unsqueeze(0)  # (1, N_MELS, T)
    tensor = _resize_tensor(tensor, TARGET_SIZE, TARGET_SIZE)
    return tensor  # (1, 224, 224)


def waveform_to_lstm_tensor(y: np.ndarray) -> torch.Tensor:
    """
    Convert a mono waveform at TARGET_SR to a (T, N_MELS) tensor.
    The LSTM sees the mel-spectrogram as a time-series of N_MELS features.
    """
    y = _trim_silence(y)
    y = _pad_or_crop(y, TARGET_NUM_SAMPLES)
    mel_spec = _compute_mel_spectrogram(y)
    mel_db = _to_log_scale(mel_spec)
    mel_norm = _normalize(mel_db)
    # mel_norm: (N_MELS, T) → transpose to (T, N_MELS) for the LSTM
    return torch.from_numpy(mel_norm.T).float()  # (T, N_MELS)


# Backward compatibility alias
waveform_to_tensor = waveform_to_resnet_tensor


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    """
    Dataset that scans root_dir/real and root_dir/fake, returning
    (tensor, label) pairs.

    Args:
        root_dir:  Path containing real/ and fake/ subdirectories.
        mode:      "resnet" → (1, 224, 224) spectrogram image tensor.
                   "lstm"   → (T, N_MELS) temporal sequence tensor.
        real_subdir / fake_subdir: names of class subdirectories.
        extensions: optional whitelist of file extensions.
    """

    VALID_MODES = {"resnet", "lstm"}

    def __init__(
        self,
        root_dir: str | Path,
        mode: str = "resnet",
        real_subdir: str = "real",
        fake_subdir: str = "fake",
        extensions: Optional[List[str]] = None,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.root_dir = Path(root_dir)
        self.mode = mode
        self.samples: List[Tuple[Path, int]] = []
        self.real_dir = self.root_dir / real_subdir
        self.fake_dir = self.root_dir / fake_subdir

        if not self.real_dir.exists():
            raise FileNotFoundError(f"Real audio directory not found: {self.real_dir}")
        if not self.fake_dir.exists():
            raise FileNotFoundError(f"Fake audio directory not found: {self.fake_dir}")

        ext_set = {e.lower() for e in (extensions or [])}

        for path in sorted(self.real_dir.iterdir()):
            if path.is_file() and (not ext_set or path.suffix.lower() in ext_set):
                self.samples.append((path, 0))

        for path in sorted(self.fake_dir.iterdir()):
            if path.is_file() and (not ext_set or path.suffix.lower() in ext_set):
                self.samples.append((path, 1))

        if len(self.samples) == 0:
            raise ValueError(
                f"No audio files found under {self.real_dir} and {self.fake_dir}. "
                "Check directory structure and file extensions."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        y, _ = librosa.load(path.as_posix(), sr=TARGET_SR, mono=True)

        if self.mode == "lstm":
            tensor = waveform_to_lstm_tensor(y)
        else:
            tensor = waveform_to_resnet_tensor(y)

        return tensor, label
