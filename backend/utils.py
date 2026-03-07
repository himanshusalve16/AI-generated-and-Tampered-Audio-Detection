"""
Shared utilities for the Audio Deepfake Detection backend.
Provides structured logging, request validation, constants, and error handling helpers.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

# -----------------------------------------------------------------------------
# Constants used across the backend
# -----------------------------------------------------------------------------

# API and server
API_TITLE = "Audio Deepfake Detection API"
API_DESCRIPTION = (
    "Detect real vs AI-generated or tampered audio using "
    "spectral feature analysis and a ResNet-18 deep learning model."
)
API_VERSION = "1.0.0"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# CORS: frontend runs on Vite default port
CORS_ORIGINS = ["http://localhost:5173"]

# File upload validation
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "audio/ogg",
    "audio/x-m4a",
    "audio/webm",
    "application/octet-stream",  # some clients send this for binary uploads
}

# Model and paths
MODEL_FILENAME = "audio_model.pth"
# Base directory is the project root (parent of backend/)
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / MODEL_FILENAME


# -----------------------------------------------------------------------------
# Structured logging setup
# -----------------------------------------------------------------------------

def setup_logging(
    level: int = logging.INFO,
    log_format: Optional[str] = None,
    stream: Any = None,
) -> logging.Logger:
    """
    Configure structured logging for the application.
    Returns the root logger so callers can get child loggers via getLogger(__name__).
    """
    if log_format is None:
        log_format = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    if stream is None:
        stream = sys.stdout

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=stream,
        force=True,
    )

    # Reduce noise from third-party libs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)

    return logging.getLogger()


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name (typically __name__)."""
    return logging.getLogger(name)


# -----------------------------------------------------------------------------
# File and request validation
# -----------------------------------------------------------------------------

def validate_audio_upload(
    filename: Optional[str],
    content_type: Optional[str],
    file_size: int,
) -> Tuple[bool, Optional[str]]:
    """
    Validate an uploaded file as acceptable for audio processing.

    Returns:
        (is_valid, error_message). If is_valid is True, error_message is None.
    """
    if not filename or not filename.strip():
        return False, "Missing or empty filename."

    if file_size <= 0:
        return False, "Uploaded file is empty."

    if file_size > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        return False, f"File too large. Maximum size is {max_mb:.0f} MB."

    ext = Path(filename).suffix.lower()
    if ext and ext not in ALLOWED_AUDIO_EXTENSIONS:
        return False, (
            f"Unsupported file extension '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}."
        )

    if content_type and content_type.strip():
        # Content-Type can include charset, e.g. "audio/wav; charset=binary"
        base_type = content_type.split(";")[0].strip().lower()
        if base_type not in ALLOWED_CONTENT_TYPES:
            # Be lenient: if extension is allowed, accept anyway (browsers sometimes send wrong type)
            if not ext or ext not in ALLOWED_AUDIO_EXTENSIONS:
                return False, f"Unsupported content type: {content_type}"

    return True, None


def ensure_models_directory() -> Path:
    """Ensure the models directory exists; create if necessary. Returns the path."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def get_model_path(custom_path: Optional[str] = None) -> Path:
    """Return the path to the saved model weights (.pth file)."""
    if custom_path:
        return Path(custom_path)
    return DEFAULT_MODEL_PATH
