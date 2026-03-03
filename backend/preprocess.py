from __future__ import annotations

import io
from typing import BinaryIO

import librosa
import numpy as np
import torch
import torch.nn.functional as F


TARGET_SR = 16_000
TARGET_DURATION_SECONDS = 3.0
TARGET_NUM_SAMPLES = int(TARGET_SR * TARGET_DURATION_SECONDS)
TARGET_SIZE = 224


def preprocess_audio(file: BinaryIO) -> torch.Tensor:
  """
  Preprocess an uploaded audio file into a normalized mel-spectrogram tensor.

  Steps:
  1. Load audio using librosa from a binary file-like object.
  2. Resample to 16 kHz.
  3. Trim leading and trailing silence.
  4. Pad or cut the signal to exactly 3 seconds.
  5. Compute mel-spectrogram.
  6. Convert to log scale (dB).
  7. Normalize values (per-spectrogram standardization).
  8. Resize to 224x224 using bilinear interpolation.
  9. Return a PyTorch tensor of shape (1, 224, 224).
  """

  # Read the entire file into memory and create a buffer so librosa can load it.
  raw_bytes = file.read()
  audio_buffer = io.BytesIO(raw_bytes)

  # 1 & 2. Load and resample audio to TARGET_SR in one step.
  # librosa will convert the audio to mono by default (mono=True).
  y, sr = librosa.load(audio_buffer, sr=TARGET_SR, mono=True)

  # 3. Trim leading and trailing silence to reduce irrelevant padding.
  y, _ = librosa.effects.trim(y, top_db=20)

  # 4. Pad or cut the waveform to a fixed length (3 seconds at 16 kHz).
  if len(y) < TARGET_NUM_SAMPLES:
    # Pad with zeros at the end if the clip is too short.
    pad_width = TARGET_NUM_SAMPLES - len(y)
    y = np.pad(y, (0, pad_width), mode="constant")
  elif len(y) > TARGET_NUM_SAMPLES:
    # If the clip is too long, take a centered 3-second segment.
    start = (len(y) - TARGET_NUM_SAMPLES) // 2
    y = y[start : start + TARGET_NUM_SAMPLES]

  # 5. Compute mel-spectrogram.
  mel_spec = librosa.feature.melspectrogram(
    y=y,
    sr=TARGET_SR,
    n_fft=1024,
    hop_length=512,
    n_mels=128,
    power=2.0,
  )

  # 6. Convert power spectrogram to decibel (log) scale.
  mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

  # 7. Normalize the spectrogram (per-sample mean/std) for more stable training/inference.
  mean = mel_spec_db.mean()
  std = mel_spec_db.std() if mel_spec_db.std() > 0 else 1.0
  mel_spec_norm = (mel_spec_db - mean) / std

  # 8. Convert to a PyTorch tensor and resize to (1, 224, 224).
  # Current shape is (n_mels, time_steps). We treat this as a single-channel image.
  tensor = torch.from_numpy(mel_spec_norm).float().unsqueeze(0)  # (1, n_mels, time)

  # Use bilinear interpolation to resize the "image" to 224x224.
  # Interpolate expects a 4D tensor: (batch, channels, height, width).
  tensor = tensor.unsqueeze(0)  # (1, 1, n_mels, time)
  tensor_resized = F.interpolate(
    tensor,
    size=(TARGET_SIZE, TARGET_SIZE),
    mode="bilinear",
    align_corners=False,
  )

  # 9. Remove the batch dimension; final shape is (1, 224, 224).
  tensor_resized = tensor_resized.squeeze(0)

  return tensor_resized

