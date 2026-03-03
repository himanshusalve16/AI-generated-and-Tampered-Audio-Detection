from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


TARGET_SR = 16_000
TARGET_DURATION_SECONDS = 3.0
TARGET_NUM_SAMPLES = int(TARGET_SR * TARGET_DURATION_SECONDS)
TARGET_SIZE = 224


def _load_and_preprocess(path: Path) -> torch.Tensor:
  """
  Load a single audio file from disk and apply the same preprocessing
  steps that are used in the FastAPI backend.
  """
  y, sr = librosa.load(path.as_posix(), sr=TARGET_SR, mono=True)
  y, _ = librosa.effects.trim(y, top_db=20)

  if len(y) < TARGET_NUM_SAMPLES:
    pad_width = TARGET_NUM_SAMPLES - len(y)
    y = np.pad(y, (0, pad_width), mode="constant")
  elif len(y) > TARGET_NUM_SAMPLES:
    start = (len(y) - TARGET_NUM_SAMPLES) // 2
    y = y[start : start + TARGET_NUM_SAMPLES]

  mel_spec = librosa.feature.melspectrogram(
    y=y,
    sr=TARGET_SR,
    n_fft=1024,
    hop_length=512,
    n_mels=128,
    power=2.0,
  )

  mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

  mean = mel_spec_db.mean()
  std = mel_spec_db.std() if mel_spec_db.std() > 0 else 1.0
  mel_spec_norm = (mel_spec_db - mean) / std

  tensor = torch.from_numpy(mel_spec_norm).float().unsqueeze(0)  # (1, n_mels, time)
  tensor = tensor.unsqueeze(0)  # (1, 1, n_mels, time)

  tensor_resized = F.interpolate(
    tensor,
    size=(TARGET_SIZE, TARGET_SIZE),
    mode="bilinear",
    align_corners=False,
  )

  return tensor_resized.squeeze(0)  # (1, 224, 224)


class AudioDataset(Dataset):
  """
  Simple dataset that assumes the following folder structure:

  dataset/
    real/
      *.wav, *.mp3, ...
    fake/
      *.wav, *.mp3, ...
  """

  def __init__(self, root_dir: str | Path):
    self.root_dir = Path(root_dir)
    self.samples: List[Tuple[Path, int]] = []

    real_dir = self.root_dir / "real"
    fake_dir = self.root_dir / "fake"

    for path in sorted(real_dir.glob("*")):
      if path.is_file():
        self.samples.append((path, 0))  # label 0 → Real

    for path in sorted(fake_dir.glob("*")):
      if path.is_file():
        self.samples.append((path, 1))  # label 1 → AI-Generated/Fake

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
    path, label = self.samples[idx]
    tensor = _load_and_preprocess(path)
    return tensor, label

