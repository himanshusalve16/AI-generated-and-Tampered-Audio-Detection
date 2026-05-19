"""
Evaluate a trained model on the test set WITHOUT retraining.

Loads an existing .pth checkpoint and generates:
  - confusion_matrix.png
  - classification_report.txt

Usage:
  python evaluate.py --model resnet
  python evaluate.py --model lstm
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from torch.utils.data import DataLoader, Subset

from dataset import AudioDataset
from model import AudioLSTM, AudioResNet

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = TRAINING_DIR / "outputs"

MODEL_FILENAMES = {
    "resnet": "resnet_audio_model.pth",
    "lstm": "lstm_audio_model.pth",
}
CLASS_NAMES = ["Real", "AI Generated"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(__name__)


def build_model(arch: str) -> torch.nn.Module:
    if arch == "resnet":
        return AudioResNet()
    elif arch == "lstm":
        return AudioLSTM(input_dim=128, hidden_dim=128, num_layers=2,
                         num_classes=2, dropout=0.3)
    raise ValueError(f"Unknown architecture: {arch}")


def collect_predictions(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[List[int], List[int]]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger = setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate a trained model (no retraining).")
    parser.add_argument("--model", choices=["resnet", "lstm"], required=True)
    parser.add_argument("--test-samples", type=int, default=300,
                        help="Number of test samples to evaluate on (default: 300)")
    args = parser.parse_args()

    arch = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

    # --- Load model ---
    model_path = DEFAULT_MODELS_DIR / MODEL_FILENAMES[arch]
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        sys.exit(1)

    logger.info("Loading %s model from %s", arch.upper(), model_path)
    model = build_model(arch)

    checkpoint = torch.load(model_path, map_location=device)
    # Handle wrapped state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # Try loading directly first
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        # AudioResNet wraps resnet18 inside self.model — keys may need prefix adjustment
        adjusted = {}
        for k, v in checkpoint.items():
            if k.startswith("model."):
                adjusted[k] = v                 # already has prefix
            else:
                adjusted["model." + k] = v      # add prefix for AudioResNet wrapper
        try:
            model.load_state_dict(adjusted)
        except RuntimeError:
            # Last resort: strip prefix
            stripped = {k[len("model."):] if k.startswith("model.") else k: v
                        for k, v in checkpoint.items()}
            model.load_state_dict(stripped)

    model.to(device).eval()
    logger.info("Model loaded successfully.")

    # --- Load test data ---
    test_dir = DEFAULT_DATASET_ROOT / "test"
    if not test_dir.exists():
        logger.error("Test directory not found: %s", test_dir)
        sys.exit(1)

    full_test = AudioDataset(test_dir, mode=arch)
    total = len(full_test)

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=g).tolist()
    subset_indices = indices[:min(args.test_samples, total)]
    test_dataset = Subset(full_test, subset_indices)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    logger.info("Test samples: %d / %d", len(test_dataset), total)

    # --- Evaluate ---
    criterion = CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total_count = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total_count += labels.size(0)

    test_loss = running_loss / max(total_count, 1)
    test_acc = correct / max(total_count, 1)
    logger.info("Test Loss: %.4f | Test Acc: %.4f", test_loss, test_acc)

    # --- Collect predictions ---
    y_true, y_pred = collect_predictions(model, test_loader, device)

    # --- Save outputs ---
    out_dir = OUTPUTS_DIR / arch
    out_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {arch.upper()}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_dir / "confusion_matrix.png")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    logger.info("--- %s Test Metrics ---", arch.upper())
    logger.info("Accuracy : %.4f", acc)
    logger.info("Precision: %.4f", prec)
    logger.info("Recall   : %.4f", rec)
    logger.info("F1-score : %.4f", f1)
    logger.info("\n%s", report)

    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== {arch.upper()} Classification Report ===\n\n")
        f.write(report)
        f.write(f"\nAccuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")
    logger.info("Saved: %s", report_path)

    logger.info("All outputs saved to: %s", out_dir)


if __name__ == "__main__":
    main()
