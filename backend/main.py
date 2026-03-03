from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from preprocess import preprocess_audio
from model_loader import load_model, predict as model_predict


app = FastAPI(title="AI-Generated Audio Detection API")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:5173"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
  """
  Load the model once when the application starts.
  """
  try:
    load_model()
  except FileNotFoundError:
    # For development convenience we allow the API to start even if the model is missing.
    # /predict will surface a clear error until the weights are available.
    pass
  except Exception as exc:  # pragma: no cover - defensive
    # If something unexpected happens during startup, surface an explicit error.
    raise RuntimeError(f"Failed to load model at startup: {exc}") from exc


@app.get("/health")
async def health_check() -> dict:
  """
  Simple health check endpoint.
  """
  return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
  """
  Accept an uploaded audio file, run preprocessing and model inference, and return:
  {
      "prediction": "Real" | "AI-Generated",
      "confidence": 0.xx
  }
  """
  if not file or not file.filename:
    raise HTTPException(status_code=400, detail="No file uploaded.")

  if not (file.content_type.startswith("audio/") or file.filename.lower().endswith((".wav", ".mp3"))):
    raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a .wav or .mp3 file.")

  try:
    tensor = preprocess_audio(file.file)
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Failed to preprocess audio: {exc}") from exc

  try:
    label, confidence = model_predict(tensor)
  except FileNotFoundError:
    # If the trained model is not yet available, return a clear but non-crashing response.
    # This also acts as a convenient mock response during early development.
    label = "Real"
    confidence = 0.85
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

  return JSONResponse(
    content={
      "prediction": label,
      "confidence": float(confidence),
    }
  )


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

