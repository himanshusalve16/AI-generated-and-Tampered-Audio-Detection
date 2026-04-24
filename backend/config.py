"""
Centralized configuration for the Audio Deepfake Detection backend.

All model paths, feature dimensions, ensemble weights, and preprocessing
constants live here so that every module imports from one place.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Directory layout
# -----------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

# -----------------------------------------------------------------------------
# Model file paths
# -----------------------------------------------------------------------------

RESNET_MODEL_FILENAME = "resnet_audio_model.pth"
LSTM_MODEL_FILENAME = "lstm_audio_model.pth"

RESNET_MODEL_PATH = MODELS_DIR / RESNET_MODEL_FILENAME
LSTM_MODEL_PATH = MODELS_DIR / LSTM_MODEL_FILENAME

# Legacy single-model path (backward compat)
LEGACY_MODEL_PATH = MODELS_DIR / "audio_model.pth"

# -----------------------------------------------------------------------------
# Audio preprocessing constants (shared with training/dataset.py)
# -----------------------------------------------------------------------------

# Target sample rate in Hz — 16 kHz is standard for speech.
SAMPLE_RATE = 16_000

# Fixed clip duration in seconds.
AUDIO_DURATION_SEC = 3.0

# Number of waveform samples for the fixed duration.
NUM_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION_SEC)  # 48 000

# Mel spectrogram parameters
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
POWER = 2.0

# Silence trimming threshold (dB below peak)
TRIM_TOP_DB = 20

# ResNet spatial input size
RESNET_INPUT_SIZE = 224  # 224×224

# -----------------------------------------------------------------------------
# LSTM feature dimensions
# -----------------------------------------------------------------------------

# The LSTM receives the mel-spectrogram *before* resize.
# With 48 000 samples, N_FFT=1024, HOP_LENGTH=512:
#   time_steps = 1 + floor(48000 / 512) = 94
LSTM_INPUT_DIM = N_MELS        # 128 features per time step
LSTM_HIDDEN_DIM = 128          # hidden state size
LSTM_NUM_LAYERS = 2            # stacked LSTM layers
LSTM_NUM_CLASSES = 2           # Real / AI Generated

# ResNet classes
RESNET_NUM_CLASSES = 2

# -----------------------------------------------------------------------------
# Ensemble weights  (must sum to 1.0)
# Adjust these to trust one model more than the other.
# -----------------------------------------------------------------------------

RESNET_WEIGHT = 0.5
LSTM_WEIGHT = 0.5

# -----------------------------------------------------------------------------
# Class labels  (index 0 → Real, index 1 → AI Generated)
# -----------------------------------------------------------------------------

CLASS_LABELS = ("Real", "AI Generated")
