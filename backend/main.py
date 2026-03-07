"""
FastAPI application for AI-Generated and Tampered Audio Detection.

Provides a REST API for uploading audio and receiving predictions from a
ResNet-18 model trained on mel-spectrogram features. The model is loaded
once at server startup for efficient inference.

Endpoints:
  GET  /health  - Health check
  POST /predict - Upload audio file, receive prediction (Real / AI Generated) and confidence
"""

import io
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model_loader import load_model, predict as model_predict
from preprocess import preprocess_audio
from utils import (
    API_DESCRIPTION,
    API_TITLE,
    CORS_ORIGINS,
    get_logger,
    get_model_path,
    setup_logging,
    validate_audio_upload,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

setup_logging()
logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Application lifecycle: load model once at startup
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained model once when the server starts.
    This avoids loading weights on every request and keeps inference fast.
    """
    logger.info("Starting up: loading model weights...")
    try:
        model_path = get_model_path()
        if not model_path.exists():
            logger.warning(
                "Model file not found at %s. Prediction endpoints will return 503 until a model is trained and saved.",
                model_path,
            )
        else:
            load_model(str(model_path))
            logger.info("Model loaded successfully from %s", model_path)
    except Exception as e:
        logger.exception("Failed to load model at startup: %s", e)
        # Do not raise: allow server to start so /health still works; /predict will return 503
    yield
    logger.info("Shutting down.")


# -----------------------------------------------------------------------------
# FastAPI app and middleware
# -----------------------------------------------------------------------------

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version="1.0.0",
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
# Response helpers (keep frontend contract: label + confidence)
# -----------------------------------------------------------------------------

def prediction_response(label: str, confidence: float) -> Dict[str, Any]:
    """
    Build the JSON response for a successful prediction.
    Frontend expects 'label' and 'confidence' (see App.jsx).
    """
    return {
        "label": label,
        "confidence": round(float(confidence), 4),
    }


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    """
    Health check endpoint for load balancers and monitoring.
    Returns a simple status payload.
    """
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)")):
    """
    Accept an uploaded audio file, preprocess it, run the deep learning model,
    and return the predicted class (Real or AI Generated) with a confidence score.

    Pipeline:
      1. Validate uploaded file (size, type, extension)
      2. Load audio into memory
      3. Preprocess: resample, trim, pad/crop, mel spectrogram, normalize, resize
      4. Convert to PyTorch tensor (1, 224, 224)
      5. Run model inference
      6. Apply softmax and take argmax
      7. Return label and confidence as JSON

    Response format (compatible with existing frontend):
      { "label": "Real" | "AI Generated", "confidence": 0.92 }
    """
    # ----- Step 1: Validate uploaded file -----
    filename = file.filename or ""
    content_type = file.content_type or ""

    # We need to read the file to get size; do a bounded read for validation
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
        # ----- Step 2 & 3: Load and preprocess audio -----
        logger.debug("Preprocessing audio from %s (%d bytes)", filename, file_size)
        tensor = preprocess_audio(file_like)

        # ----- Step 4, 5, 6: Run model inference -----
        label, confidence = model_predict(tensor)

        # ----- Step 7: Return prediction (frontend expects 'label' and 'confidence') -----
        return prediction_response(label, confidence)

    except RuntimeError as e:
        err_msg = str(e).lower()
        if "not loaded" in err_msg:
            logger.error("Predict called but model is not loaded.")
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train a model and place audio_model.pth in the models/ directory, then restart the server.",
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
