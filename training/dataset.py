"""
PyTorch Dataset for real vs AI-generated (or tampered) audio classification.

Loads audio files from a directory structure:
  dataset/
    real/   -> label 0 (Real)
    fake/   -> label 1 (AI Generated / Fake)

Preprocessing is identical to the backend (preprocess.py) so that the model
sees the same feature distribution during training and inference. Each
sample is converted to a mel-spectrogram tensor of shape (1, 224, 224).
"""

from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
# Preprocessing constants (must match backend/preprocess.py exactly)
# -----------------------------------------------------------------------------

TARGET_SR = 16000
TARGET_DURATION_SEC = 3.0
TARGET_NUM_SAMPLES = int(TARGET_SR * TARGET_DURATION_SEC)
TARGET_SIZE = 224
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
POWER = 2.0
TRIM_TOP_DB = 20


# -----------------------------------------------------------------------------
# Waveform -> tensor (same pipeline as backend)
# -----------------------------------------------------------------------------

def _trim_silence(y: np.ndarray) -> np.ndarray:
    y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    return y_trimmed


def _pad_or_crop_to_fixed_length(y: np.ndarray, target_length: int) -> np.ndarray:
    n = len(y)
    if n < target_length:
        pad_len = target_length - n
        y = np.pad(y, (0, pad_len), mode="constant", constant_values=0.0)
    elif n > target_length:
        start = (n - target_length) // 2
        y = y[start : start + target_length]
    return y


def _compute_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=POWER,
    )


def _to_log_scale(mel_spec: np.ndarray) -> np.ndarray:
    return librosa.power_to_db(mel_spec, ref=np.max)


def _normalize_spectrogram(mel_db: np.ndarray) -> np.ndarray:
    mean = float(np.mean(mel_db))
    std = float(np.std(mel_db))
    if std <= 0:
        std = 1.0
    return (mel_db - mean) / std


def _resize_to_target(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    tensor_4d = tensor.unsqueeze(0)
    resized = F.interpolate(
        tensor_4d,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def waveform_to_tensor(y: np.ndarray) -> torch.Tensor:
    """
    Convert a mono waveform at TARGET_SR to a (1, 224, 224) mel-spectrogram tensor.
    Kept in sync with backend/preprocess._waveform_to_tensor for identical preprocessing.
    """
    y = _trim_silence(y)
    y = _pad_or_crop_to_fixed_length(y, TARGET_NUM_SAMPLES)
    mel_spec = _compute_mel_spectrogram(y)
    mel_db = _to_log_scale(mel_spec)
    mel_norm = _normalize_spectrogram(mel_db)
    tensor = torch.from_numpy(mel_norm).float().unsqueeze(0)
    tensor = _resize_to_target(tensor, TARGET_SIZE, TARGET_SIZE)
    return tensor  # (1, 224, 224)


# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------

class AudioDataset(Dataset):
    """
    Dataset that scans dataset/real and dataset/fake and returns
    (tensor, label) pairs with preprocessing applied on the fly.
    """

    def __init__(
        self,
        root_dir: str | Path,
        real_subdir: str = "real",
        fake_subdir: str = "fake",
        extensions: Optional[List[str]] = None,
    ):
        """
        Args:
            root_dir: Path to the dataset root (containing real/ and fake/).
            real_subdir: Name of subfolder for real audio (label 0).
            fake_subdir: Name of subfolder for fake/AI-generated audio (label 1).
            extensions: If set, only include files with these extensions (e.g. [".wav", ".mp3"]).
                       If None, include all files (librosa will attempt to load them).
        """
        self.root_dir = Path(root_dir)
        self.samples: List[Tuple[Path, int]] = []
        self.real_dir = self.root_dir / real_subdir
        self.fake_dir = self.root_dir / fake_subdir

        if not self.real_dir.exists():
            raise FileNotFoundError(f"Real audio directory not found: {self.real_dir}")
        if not self.fake_dir.exists():
            raise FileNotFoundError(f"Fake audio directory not found: {self.fake_dir}")

        if extensions is None:
            extensions = []  # no filter
        ext_set = {e.lower() for e in extensions}

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
        tensor = waveform_to_tensor(y)
        return tensor, label
