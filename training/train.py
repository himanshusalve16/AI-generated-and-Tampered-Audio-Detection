"""
Training pipeline for the Audio Deepfake Detection models.

Trains either a ResNet-18 (spectrogram image classifier) or a bidirectional
LSTM (temporal sequence classifier) depending on the --model flag.

Usage:
  python train.py --model resnet
  python train.py --model lstm
  python train.py --model resnet --epochs 30 --batch-size 32 --lr 5e-5
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from dataset import AudioDataset
from model import AudioLSTM, AudioResNet

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUTS_DIR = TRAINING_DIR / "outputs"

# Output filenames for each architecture
MODEL_FILENAMES = {
    "resnet": "resnet_audio_model.pth",
    "lstm": "lstm_audio_model.pth",
}

DEFAULT_TRAIN_SUBDIR = "train"
DEFAULT_TEST_SUBDIR = "test"

# Class label names (index 0 = Real, index 1 = Fake/AI Generated)
CLASS_NAMES = ["Real", "AI Generated"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------

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


def _resolve_split_dirs(
    dataset_root: Path, train_subdir: str, test_subdir: str
) -> Tuple[Path, Optional[Path]]:
    """
    Resolve train/test directories.

    Preferred structure:
      dataset_root/train/real, train/fake
      dataset_root/test/real,  test/fake

    Backward-compatible fallback:
      dataset_root/real, dataset_root/fake (no separate test set)
    """
    train_dir = dataset_root / train_subdir
    test_dir = dataset_root / test_subdir

    if train_dir.exists():
        return train_dir, (test_dir if test_dir.exists() else None)

    return dataset_root, None


# ---------------------------------------------------------------------------
# Plotting and evaluation helpers
# ---------------------------------------------------------------------------

def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path,
) -> None:
    """Plot training and validation loss curves and save to file."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, "o-", label="Train Loss", color="#2563eb", linewidth=2)
    plt.plot(epochs, val_losses, "s-", label="Validation Loss", color="#dc2626", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_accuracy_curve(
    train_accuracies: List[float],
    val_accuracies: List[float],
    save_path: Path,
) -> None:
    """Plot training and validation accuracy curves and save to file."""
    epochs = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracies, "o-", label="Train Accuracy", color="#2563eb", linewidth=2)
    plt.plot(epochs, val_accuracies, "s-", label="Validation Accuracy", color="#16a34a", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Path,
) -> None:
    """Compute and plot the confusion matrix, save to file."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def generate_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Path,
    logger: logging.Logger,
) -> None:
    """Generate sklearn classification report, print metrics, and save to file."""
    report = classification_report(y_true, y_pred, target_names=class_names)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    logger.info("--- Test Set Metrics ---")
    logger.info("Accuracy : %.4f", acc)
    logger.info("Precision: %.4f", prec)
    logger.info("Recall   : %.4f", rec)
    logger.info("F1-score : %.4f", f1)
    logger.info("\n%s", report)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== Classification Report ===\n\n")
        f.write(report)
        f.write(f"\nAccuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    """Run model on a DataLoader and return (y_true, y_pred) lists."""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_model(arch: str) -> torch.nn.Module:
    """Instantiate the requested architecture."""
    if arch == "resnet":
        return AudioResNet()
    elif arch == "lstm":
        return AudioLSTM(
            input_dim=128,
            hidden_dim=128,
            num_layers=2,
            num_classes=2,
            dropout=0.3,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    architecture: str = "resnet",
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
    Run the full training pipeline for a given architecture.

    Args:
        architecture: "resnet" or "lstm".
        (other args same as before)

    Returns:
        Best validation accuracy achieved.
    """
    logger = setup_train_logging(verbose=verbose)
    dataset_root_path = Path(dataset_root or DEFAULT_DATASET_ROOT)
    models_dir_path = Path(models_dir or DEFAULT_MODELS_DIR)
    models_dir_path.mkdir(parents=True, exist_ok=True)

    if architecture not in MODEL_FILENAMES:
        raise ValueError(f"architecture must be 'resnet' or 'lstm', got '{architecture}'")

    best_model_path = models_dir_path / MODEL_FILENAMES[architecture]

    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed: %d", seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Architecture: %s", architecture.upper())
    logger.info("Device: %s", device)
    logger.info("Dataset root: %s", dataset_root_path)
    logger.info("Models dir:   %s", models_dir_path)
    logger.info(
        "Batch size: %d, epochs: %d, lr: %s, val_split: %.2f",
        batch_size, num_epochs, learning_rate, val_split,
    )

    train_dir, test_dir = _resolve_split_dirs(
        dataset_root_path, train_subdir=train_subdir, test_subdir=test_subdir
    )
    logger.info("Train dir: %s", train_dir)

    # -----------------------------------------------------------------------
    # 1. Sample 1000 from train dir → 800 train + 200 validation
    # -----------------------------------------------------------------------
    full_train_dataset = AudioDataset(train_dir, mode=architecture)
    total_train_samples = len(full_train_dataset)

    SUBSET_SIZE = 1000
    TRAIN_SIZE = 800
    VAL_SIZE = SUBSET_SIZE - TRAIN_SIZE  # 200

    # Reproducible shuffle of all indices
    g = torch.Generator().manual_seed(seed)
    all_train_indices = torch.randperm(total_train_samples, generator=g).tolist()

    # Take first 1000 indices from the shuffled list
    subset_indices = all_train_indices[:SUBSET_SIZE]

    # First 800 → training, remaining 200 → validation (no overlap)
    train_indices = subset_indices[:TRAIN_SIZE]
    val_indices = subset_indices[TRAIN_SIZE:SUBSET_SIZE]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    # -----------------------------------------------------------------------
    # 2. Sample 300 from test dir (separate, NOT used during training)
    # -----------------------------------------------------------------------
    TEST_SUBSET_SIZE = 300
    test_dataset = None
    test_count = 0
    if test_dir is not None:
        try:
            full_test_dataset = AudioDataset(test_dir, mode=architecture)
            total_test_samples = len(full_test_dataset)
            g_test = torch.Generator().manual_seed(seed)
            all_test_indices = torch.randperm(total_test_samples, generator=g_test).tolist()
            test_subset_indices = all_test_indices[:min(TEST_SUBSET_SIZE, total_test_samples)]
            test_dataset = Subset(full_test_dataset, test_subset_indices)
            test_count = len(test_dataset)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Could not load test set from %s: %s", test_dir, e)

    # -----------------------------------------------------------------------
    # 3. Print dataset sizes
    # -----------------------------------------------------------------------
    logger.info("Dataset subset mode enabled")
    logger.info("Total train samples (original): %d", total_train_samples)
    logger.info("Sampled subset: %d", SUBSET_SIZE)
    logger.info("Training samples: %d", TRAIN_SIZE)
    logger.info("Validation samples: %d", VAL_SIZE)
    logger.info("Test samples used: %d", test_count)

    # -----------------------------------------------------------------------
    # 4. Create DataLoaders
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Model, optimizer, criterion
    # -----------------------------------------------------------------------
    model = _build_model(architecture).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    best_val_acc = 0.0

    # -----------------------------------------------------------------------
    # 5. Metric tracking lists
    # -----------------------------------------------------------------------
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch_train(model, train_loader, optimizer, criterion, device)
        val_metrics = run_epoch_val(model, val_loader, criterion, device)

        train_loss = train_metrics["loss"]
        train_acc = train_metrics["accuracy"]
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        logger.info(
            "Epoch %03d/%03d | Train Loss: %.4f | Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f",
            epoch, num_epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info("  -> Best model saved to %s (val_acc=%.4f)", best_model_path, val_acc)

    # Safeguard: always save at least the final epoch
    if not best_model_path.exists():
        torch.save(model.state_dict(), best_model_path)
        logger.info("No validation-based checkpoint found; saved final epoch model to %s", best_model_path)

    logger.info("Training complete. Best validation accuracy: %.4f", best_val_acc)
    logger.info("Best model saved to: %s", best_model_path)

    # -----------------------------------------------------------------------
    # 6. Generate plots and evaluation
    # -----------------------------------------------------------------------
    outputs_dir = DEFAULT_OUTPUTS_DIR
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving evaluation outputs to: %s", outputs_dir)

    # A) Loss curve
    plot_loss_curve(train_losses, val_losses, outputs_dir / "loss_curve.png")
    logger.info("Saved: loss_curve.png")

    # B) Accuracy curve
    plot_accuracy_curve(train_accuracies, val_accuracies, outputs_dir / "accuracy_curve.png")
    logger.info("Saved: accuracy_curve.png")

    # C) Test set evaluation: confusion matrix + classification report
    if test_loader is not None:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)

        test_metrics = run_epoch_val(model, test_loader, criterion, device)
        logger.info(
            "Test set | Loss: %.4f | Acc: %.4f",
            test_metrics["loss"],
            test_metrics["accuracy"],
        )

        # Collect predictions for confusion matrix and report
        y_true, y_pred = collect_predictions(model, test_loader, device)

        # Confusion matrix
        plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, outputs_dir / "confusion_matrix.png")
        logger.info("Saved: confusion_matrix.png")

        # Classification report
        generate_classification_report(
            y_true, y_pred, CLASS_NAMES,
            outputs_dir / "classification_report.txt",
            logger,
        )
        logger.info("Saved: classification_report.txt")
    else:
        logger.warning("No test set found — skipping confusion matrix and classification report.")

    logger.info("All evaluation outputs saved to: %s", outputs_dir)
    return best_val_acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Audio Deepfake Detection model (ResNet-18 or LSTM)."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet", "lstm"],
        default="resnet",
        help="Architecture to train: 'resnet' or 'lstm' (default: resnet)",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset root (default: ../dataset)")
    parser.add_argument(
        "--train-subdir", type=str, default=DEFAULT_TRAIN_SUBDIR,
        help="Subfolder under --dataset for training data (default: train).",
    )
    parser.add_argument(
        "--test-subdir", type=str, default=DEFAULT_TEST_SUBDIR,
        help="Subfolder under --dataset for test data (default: test).",
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
        architecture=args.model,
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
