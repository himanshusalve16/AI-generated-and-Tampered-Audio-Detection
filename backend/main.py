"""
FastAPI application for AI-Generated and Tampered Audio Detection.

Provides a REST API for uploading audio and receiving predictions from a
ResNet-18 + LSTM ensemble. Both models are loaded once at server startup.

Endpoints:
  GET  /health  - Health check (includes model status)
  POST /predict - Upload audio, receive ensemble prediction JSON
"""

import io
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model_loader import load_resnet, load_lstm, models_loaded, predict as ensemble_predict
from preprocess import preprocess_audio
from utils import (
    API_DESCRIPTION,
    API_TITLE,
    CORS_ORIGINS,
    get_logger,
    get_resnet_model_path,
    get_lstm_model_path,
    setup_logging,
    validate_audio_upload,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

setup_logging()
logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Application lifecycle: load both models at startup
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained ResNet and LSTM models when the server starts.
    Missing model files are warned but do not prevent the server from running.
    """
    logger.info("Starting up: loading model weights...")

    # --- ResNet ---
    resnet_path = get_resnet_model_path()
    if resnet_path.exists():
        try:
            load_resnet(str(resnet_path))
            logger.info("ResNet model loaded from %s", resnet_path)
        except Exception as e:
            logger.exception("Failed to load ResNet model: %s", e)
    else:
        logger.warning("ResNet model not found at %s — ResNet predictions will be unavailable.", resnet_path)

    # --- LSTM ---
    lstm_path = get_lstm_model_path()
    if lstm_path.exists():
        try:
            load_lstm(str(lstm_path))
            logger.info("LSTM model loaded from %s", lstm_path)
        except Exception as e:
            logger.exception("Failed to load LSTM model: %s", e)
    else:
        logger.warning("LSTM model not found at %s — LSTM predictions will be unavailable.", lstm_path)

    status = models_loaded()
    if not status["resnet"] and not status["lstm"]:
        logger.warning(
            "No models loaded. /predict will return 503 until at least one model is trained."
        )

    yield
    logger.info("Shutting down.")


# -----------------------------------------------------------------------------
# FastAPI app and middleware
# -----------------------------------------------------------------------------

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Health check endpoint for load balancers and monitoring.
    Also reports which models are currently loaded.
    """
    return {
        "status": "ok",
        "models": models_loaded(),
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)")):
    """
    Accept an uploaded audio file, preprocess it, run the ResNet + LSTM
    ensemble, and return individual and ensemble predictions.

    Pipeline:
      1. Validate uploaded file (size, type, extension)
      2. Load audio into memory
      3. Preprocess: resample, trim, pad, mel spectrogram
      4. Run ResNet inference on spectrogram image
      5. Run LSTM inference on temporal sequence
      6. Compute weighted ensemble
      7. Return JSON with all predictions + spectrogram image

    Response format:
      {
        "resnet_prediction": "Real",
        "resnet_confidence": 0.91,
        "lstm_prediction": "AI Generated",
        "lstm_confidence": 0.87,
        "ensemble_prediction": "AI Generated",
        "ensemble_confidence": 0.89,
        "spectrogram_image": "data:image/png;base64,..."
      }
    """
    # ----- Step 1: Validate -----
    filename = file.filename or ""
    content_type = file.content_type or ""

    contents = await file.read()
    file_size = len(contents)

    is_valid, validation_error = validate_audio_upload(filename, content_type, file_size)
    if not is_valid:
        logger.warning("Predict request rejected: %s", validation_error)
        raise HTTPException(status_code=400, detail=validation_error)

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    file_like = io.BytesIO(contents)

    try:
        # ----- Step 2 & 3: Preprocess -----
        logger.debug("Preprocessing audio from %s (%d bytes)", filename, file_size)
        preprocessed = preprocess_audio(file_like)

        # ----- Step 4, 5, 6: Ensemble inference -----
        status = models_loaded()
        if not status["resnet"] and not status["lstm"]:
            raise RuntimeError(
                "No models are loaded. Train at least one model and restart the server."
            )

        result = ensemble_predict(
            resnet_tensor=preprocessed.resnet_tensor,
            lstm_tensor=preprocessed.lstm_tensor,
        )

        # ----- Step 7: Build response -----
        return {
            "resnet_prediction": result.resnet_prediction,
            "resnet_confidence": result.resnet_confidence,
            "lstm_prediction": result.lstm_prediction,
            "lstm_confidence": result.lstm_confidence,
            "ensemble_prediction": result.ensemble_prediction,
            "ensemble_confidence": result.ensemble_confidence,
            "spectrogram_image": preprocessed.spectrogram_base64,
        }

    except RuntimeError as e:
        err_msg = str(e).lower()
        if "not loaded" in err_msg or "no models" in err_msg:
            logger.error("Predict called but models are not loaded.")
            raise HTTPException(
                status_code=503,
                detail="No models loaded. Please train models and restart the server.",
            )
        logger.exception("Runtime error during prediction: %s", e)
        raise HTTPException(status_code=422, detail=f"Processing error: {str(e)}")

    except ValueError as e:
        logger.warning("Validation error during preprocessing: %s", e)
        raise HTTPException(status_code=422, detail=f"Invalid audio or processing error: {str(e)}")

    except Exception as e:
        logger.exception("Unexpected error during prediction: %s\n%s", e, traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your request. Please try again.",
        )
