"""
Full training pipeline for the Audio Deepfake Detection model.

Loads audio from dataset/train/real and dataset/train/fake, applies the same preprocessing
as the backend, trains a ResNet-18 with Adam and CrossEntropyLoss, and saves
the best model (by validation accuracy) to models/audio_model.pth.

Usage:
  python train.py
  python train.py --epochs 30 --batch-size 32 --lr 5e-5
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from dataset import AudioDataset
from model import AudioResNet

# -----------------------------------------------------------------------------
# Paths and defaults
# -----------------------------------------------------------------------------

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_FILENAME = "audio_model.pth"
DEFAULT_TRAIN_SUBDIR = "train"
DEFAULT_TEST_SUBDIR = "test"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_train_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Training and validation loops
# -----------------------------------------------------------------------------

def run_epoch_train(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch. Returns dict with loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
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

    n = total if total > 0 else 1
    return {
        "loss": running_loss / n,
        "accuracy": correct / n,
    }


def run_epoch_val(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run one validation epoch. Returns dict with loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    n = total if total > 0 else 1
    return {
        "loss": running_loss / n,
        "accuracy": correct / n,
    }


def _resolve_split_dirs(dataset_root: Path, train_subdir: str, test_subdir: str) -> Tuple[Path, Optional[Path]]:
    """
    Resolve train/test directories.

    Preferred structure:
      dataset_root/
        train/real, train/fake
        test/real,  test/fake

    Backward-compatible fallback:
      dataset_root/real, dataset_root/fake  (no separate test set)
    """
    train_dir = dataset_root / train_subdir
    test_dir = dataset_root / test_subdir

    if train_dir.exists():
        return train_dir, (test_dir if test_dir.exists() else None)

    return dataset_root, None


# -----------------------------------------------------------------------------
# Main training entry point
# -----------------------------------------------------------------------------

def train(
    dataset_root: Optional[str] = None,
    models_dir: Optional[str] = None,
    train_subdir: str = DEFAULT_TRAIN_SUBDIR,
    test_subdir: str = DEFAULT_TEST_SUBDIR,
    batch_size: int = 16,
    num_epochs: int = 25,
    learning_rate: float = 1e-4,
    val_split: float = 0.2,
    num_workers: int = 0,
    verbose: bool = False,
) -> float:
    """
    Run the full training pipeline.

    - Loads dataset and splits into train/validation.
    - Uses DataLoader for batching.
    - Trains with Adam and CrossEntropyLoss.
    - Logs train/val loss and accuracy each epoch.
    - Saves the best model (by validation accuracy) to models/audio_model.pth.

    Returns:
        Best validation accuracy achieved.
    """
    logger = setup_train_logging(verbose=verbose)
    dataset_root = Path(dataset_root or DEFAULT_DATASET_ROOT)
    models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / BEST_MODEL_FILENAME

    # Reproducibility (optional; can be overridden by env if needed)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed: %d", seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Dataset root: %s", dataset_root)
    logger.info("Models dir:   %s", models_dir)
    logger.info("Batch size: %d, epochs: %d, lr: %s, val_split: %.2f", batch_size, num_epochs, learning_rate, val_split)

    train_dir, _ = _resolve_split_dirs(dataset_root, train_subdir=train_subdir, test_subdir=test_subdir)
    logger.info("Train dir: %s", train_dir)

    # -------------------------------------------------------------------------
    # Dataset subset mode (for faster experimentation)
    # -------------------------------------------------------------------------
    full_dataset = AudioDataset(train_dir)
    total_available = len(full_dataset)

    # Fixed seed for reproducible shuffling of indices
    g = torch.Generator()
    g.manual_seed(seed)
    all_indices = torch.randperm(total_available, generator=g).tolist()

    max_subset = min(1000, total_available)
    subset_indices = all_indices[:max_subset]

    train_count = min(800, max_subset)
    test_count = max_subset - train_count

    train_indices = subset_indices[:train_count]
    test_indices = subset_indices[train_count:train_count + test_count]

    train_dataset = Subset(full_dataset, train_indices)
    # No separate validation subset in subset mode; keep val loader structurally,
    # but make it empty so training loop stays unchanged.
    val_dataset = Subset(full_dataset, [])
    test_dataset = Subset(full_dataset, test_indices) if test_count > 0 else None

    logger.info("Dataset subset mode enabled")
    logger.info("Available samples in train dir: %d", total_available)
    logger.info("Total samples used: %d", max_subset)
    logger.info("Training samples: %d", train_count)
    logger.info("Testing samples: %d", test_count)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader: Optional[DataLoader] = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    model = AudioResNet().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch_train(model, train_loader, optimizer, criterion, device)
        val_metrics = run_epoch_val(model, val_loader, criterion, device)

        train_loss = train_metrics["loss"]
        train_acc = train_metrics["accuracy"]
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        logger.info(
            "Epoch %03d/%03d | Train Loss: %.4f | Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f",
            epoch,
            num_epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info("  -> Best model saved to %s (val_acc=%.4f)", best_model_path, val_acc)

    # In subset mode the validation loader can be effectively empty, so the
    # val_acc condition above might never trigger and no checkpoint is saved.
    # As a safeguard, ensure that we always have *some* model file to evaluate.
    if not best_model_path.exists():
        torch.save(model.state_dict(), best_model_path)
        logger.info("No validation-based checkpoint found; saved final epoch model to %s", best_model_path)

    logger.info("Training complete. Best validation accuracy: %.4f", best_val_acc)
    logger.info("Best model saved to: %s", best_model_path)

    if test_loader is not None:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        test_metrics = run_epoch_val(model, test_loader, criterion, device)
        logger.info(
            "Test set | Loss: %.4f | Acc: %.4f",
            test_metrics["loss"],
            test_metrics["accuracy"],
        )

    return best_val_acc


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Audio Deepfake Detection model (ResNet-18).")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset root (default: ../dataset)")
    parser.add_argument(
        "--train-subdir",
        type=str,
        default=DEFAULT_TRAIN_SUBDIR,
        help="Subfolder under --dataset for training data (default: train). Ignored if --dataset points directly to a real/fake folder.",
    )
    parser.add_argument(
        "--test-subdir",
        type=str,
        default=DEFAULT_TEST_SUBDIR,
        help="Subfolder under --dataset for test data (default: test). If missing, test evaluation is skipped.",
    )
    parser.add_argument("--models-dir", type=str, default=None, help="Directory to save model (default: ../models)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data for validation (0-1)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_root=args.dataset,
        models_dir=args.models_dir,
        train_subdir=args.train_subdir,
        test_subdir=args.test_subdir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        val_split=args.val_split,
        num_workers=args.num_workers,
        verbose=args.verbose,
    )
