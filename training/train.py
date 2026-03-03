from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from dataset import AudioDataset
from model import AudioResNet


def train(
  dataset_root: str | Path = "../dataset",
  models_dir: str | Path = "../models",
  batch_size: int = 8,
  num_epochs: int = 10,
  learning_rate: float = 1e-4,
  val_split: float = 0.2,
) -> None:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  dataset = AudioDataset(dataset_root)

  val_size = max(1, int(len(dataset) * val_split))
  train_size = len(dataset) - val_size
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

  model = AudioResNet().to(device)
  optimizer = Adam(model.parameters(), lr=learning_rate)
  criterion = CrossEntropyLoss()

  models_path = Path(models_dir)
  models_path.mkdir(parents=True, exist_ok=True)
  best_model_path = models_path / "audio_model.pth"

  best_val_acc = 0.0

  for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()

      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * inputs.size(0)
      _, preds = torch.max(outputs, dim=1)
      correct += (preds == labels).sum().item()
      total += labels.size(0)

    train_loss = running_loss / total if total > 0 else 0.0
    train_acc = correct / total if total > 0 else 0.0

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
      for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, dim=1)
        val_correct += (preds == labels).sum().item()
        val_total += labels.size(0)

    val_loss = val_loss / val_total if val_total > 0 else 0.0
    val_acc = val_correct / val_total if val_total > 0 else 0.0

    print(
      f"Epoch {epoch:02d}/{num_epochs:02d} "
      f"- Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} "
      f"- Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
    )

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save(model.state_dict(), best_model_path)
      print(f"  -> New best model saved to {best_model_path} (val_acc={best_val_acc:.3f})")


if __name__ == "__main__":
  train()

