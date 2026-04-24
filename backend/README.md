# Backend — Audio Deepfake Detection API

FastAPI backend for the AI-Generated and Tampered Audio Detection system. Handles audio preprocessing, dual-model inference (ResNet-18 + LSTM), and serves ensemble predictions to the React frontend.

---

## Overview

The backend performs three jobs on every `/predict` request:

1. **Preprocess** the uploaded audio file into tensors for both models and a spectrogram image.
2. **Run inference** through the ResNet-18 (spatial patterns) and LSTM (temporal patterns).
3. **Compute ensemble** prediction using weighted averaging of softmax probabilities.

---

## File Structure

| File              | Purpose                                                        |
|-------------------|----------------------------------------------------------------|
| `config.py`       | Centralized constants: model paths, feature dims, ensemble weights, sample rate |
| `main.py`         | FastAPI app definition, `/health` and `/predict` endpoints, model loading lifecycle |
| `model_loader.py` | Architecture definitions (ResNet + LSTM), weight loading, dual inference + ensemble |
| `preprocess.py`   | Audio → ResNet tensor `(1, 224, 224)` + LSTM tensor `(1, T, 128)` + base64 spectrogram |
| `utils.py`        | Logging setup, upload validation, model path helpers           |
| `requirements.txt`| Python dependencies                                            |

---

## Endpoints

### `GET /health`

Returns server status and which models are loaded.

```json
{
  "status": "ok",
  "models": {
    "resnet": true,
    "lstm": true
  }
}
```

### `POST /predict`

Accepts an audio file (multipart form data, field name: `file`).

**Supported formats:** WAV, MP3, FLAC, OGG, M4A, WebM (up to 50 MB).

**Processing pipeline:**

1. Validate file extension, content type, and size.
2. Load audio with librosa at 16 kHz mono.
3. Trim silence, pad/crop to 3 seconds (48,000 samples).
4. Compute mel spectrogram (128 mel bands, 1024 FFT, 512 hop).
5. Generate three outputs:
   - ResNet tensor: log-dB normalized, resized to 224×224
   - LSTM tensor: log-dB normalized, shape (94, 128), no resize
   - Spectrogram PNG: rendered with matplotlib, returned as base64
6. Run ResNet-18 forward pass → softmax → prediction + confidence.
7. Run LSTM forward pass → softmax → prediction + confidence.
8. Compute weighted ensemble: `w_resnet × P_resnet + w_lstm × P_lstm`.

**Response:**

```json
{
  "resnet_prediction": "Real",
  "resnet_confidence": 0.91,
  "lstm_prediction": "AI Generated",
  "lstm_confidence": 0.87,
  "ensemble_prediction": "AI Generated",
  "ensemble_confidence": 0.89,
  "spectrogram_image": "data:image/png;base64,..."
}
```

**Error codes:**
- `400` — Bad request (invalid file)
- `422` — Processing error (corrupt audio)
- `503` — No models loaded
- `500` — Unexpected server error

---

## Model Loading

Models are loaded **once at startup** in the FastAPI `lifespan` handler:

1. Backend looks for `models/resnet_audio_model.pth` (falls back to `models/audio_model.pth` for backward compatibility).
2. Backend looks for `models/lstm_audio_model.pth`.
3. Each model is optional — the server starts even if one or both are missing.
4. If **neither** model is found, `/predict` returns `503`.

Model instances are cached as module-level singletons in `model_loader.py` to avoid reloading on every request.

---

## Configuration

All tunable constants are in `config.py`:

| Constant          | Default | Description                               |
|-------------------|---------|-------------------------------------------|
| `SAMPLE_RATE`     | 16000   | Audio resampling rate (Hz)                |
| `AUDIO_DURATION_SEC` | 3.0  | Fixed clip length (seconds)               |
| `N_MELS`          | 128     | Number of mel frequency bands             |
| `RESNET_INPUT_SIZE` | 224   | Spatial size for ResNet input              |
| `LSTM_HIDDEN_DIM` | 128     | LSTM hidden state dimension               |
| `LSTM_NUM_LAYERS` | 2       | Number of stacked LSTM layers             |
| `RESNET_WEIGHT`   | 0.5     | Ensemble weight for ResNet                |
| `LSTM_WEIGHT`     | 0.5     | Ensemble weight for LSTM                  |

---

## Running

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## CORS

The backend allows requests from `http://localhost:5173` (Vite dev server). To add more origins, edit `CORS_ORIGINS` in `utils.py`.
