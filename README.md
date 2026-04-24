# AI-Generated and Tampered Audio Detection


**Using Spectral Feature Analysis and Deep Learning**

A full-stack web application that detects AI-generated or tampered audio using a **ResNet-18 + LSTM ensemble** trained on mel-spectrogram features. Upload a short speech clip, and the system will classify it as **Real** or **AI Generated** with confidence scores from two independent deep learning models.

---

## Problem Statement

The rapid advancement of AI voice synthesis (text-to-speech, voice cloning, deepfake audio) has made it increasingly difficult to distinguish real human speech from machine-generated audio. This project addresses the challenge of **automatic audio authenticity verification** by combining two complementary deep learning approaches.

## Motivation

- AI-generated speech is being misused for fraud, misinformation, and identity theft.
- Single-model classifiers can be brittle — an ensemble of architecturally different models provides more robust detection.
- Mel-spectrogram analysis captures both spatial (frequency domain patterns) and temporal (time-series dynamics) anomalies left by audio generation algorithms.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                         │
│  Upload audio → Preview → Send to API → Display ensemble results │
│  Show spectrogram image, ResNet card, LSTM card, Ensemble card   │
└────────────────────────────┬─────────────────────────────────────┘
                             │  HTTP POST /predict (multipart file)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI)                            │
│                                                                  │
│  1. Validate uploaded audio file                                 │
│  2. Preprocess (preprocess.py):                                  │
│     • Resample to 16 kHz mono                                    │
│     • Trim silence, pad/crop to 3 seconds                        │
│     • Compute mel spectrogram (128 mel bands)                    │
│     • Generate:                                                  │
│       - ResNet tensor (1, 224, 224)                               │
│       - LSTM tensor (1, 94, 128)                                 │
│       - Base64 spectrogram PNG                                   │
│                                                                  │
│  3. Inference (model_loader.py):                                 │
│     • ResNet-18 → softmax probabilities                          │
│     • LSTM (BiLSTM) → softmax probabilities                      │
│     • Weighted ensemble (configurable weights)                   │
│                                                                  │
│  4. Return JSON response with all predictions                    │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User** selects an audio file (WAV, MP3, etc.) in the React frontend.
2. **Frontend** sends the file via `POST /predict` to the FastAPI backend.
3. **Backend** preprocesses the audio once and produces tensors for both models.
4. **ResNet-18** receives a 224×224 mel-spectrogram image and classifies spatial patterns.
5. **LSTM** receives a (94, 128) temporal mel sequence and classifies time-series dynamics.
6. **Ensemble** merges both probability vectors via weighted averaging.
7. **Response** JSON is sent back containing individual predictions, confidence scores, and the spectrogram image.
8. **Frontend** displays the spectrogram and three prediction cards (ResNet, LSTM, Ensemble).

---

## Features

- **Dual-model ensemble** — ResNet-18 (CNN) + Bidirectional LSTM (RNN) for robust detection.
- **Individual model predictions** — See what each model thinks independently.
- **Ensemble final prediction** — Weighted average of both models' softmax probabilities.
- **Spectrogram visualization** — The mel spectrogram used by ResNet is displayed on the dashboard.
- **Audio preview** — Listen to the uploaded clip directly in the browser.
- **Configurable ensemble weights** — Adjust model trust in `backend/config.py`.
- **Modern dark-themed UI** — Glassmorphism, GSAP animations, responsive layout.

---

## Tech Stack

| Layer       | Technology                              |
|-------------|----------------------------------------|
| **Frontend** | React 18, Tailwind CSS 3, GSAP, Lucide Icons, Vite |
| **Backend**  | FastAPI, Uvicorn, Python 3.10+          |
| **Deep Learning** | PyTorch, torchvision, librosa      |
| **Visualization** | matplotlib (server-side spectrogram rendering) |

---

## Folder Structure

```
DL Project/
├── backend/                  # FastAPI backend
│   ├── config.py             # Centralized constants (paths, dims, weights)
│   ├── main.py               # FastAPI app, endpoints, lifecycle
│   ├── model_loader.py       # Dual model loading + ensemble inference
│   ├── preprocess.py         # Audio → ResNet tensor + LSTM tensor + spectrogram
│   ├── utils.py              # Logging, validation, helpers
│   └── requirements.txt      # Python dependencies
│
├── frontend/                 # React frontend (Vite)
│   ├── src/
│   │   ├── App.jsx           # Main layout and API call
│   │   ├── components/
│   │   │   ├── UploadCard.jsx    # File upload + audio preview
│   │   │   └── ResultCard.jsx    # Spectrogram + 3 prediction cards
│   │   ├── App.css           # Additional styles
│   │   ├── index.css         # Tailwind base + body background
│   │   └── main.jsx          # React entry point
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.mjs
│   └── vite.config.mjs
│
├── training/                 # Model training pipeline
│   ├── train.py              # CLI training script (--model resnet|lstm)
│   ├── model.py              # AudioResNet + AudioLSTM definitions
│   └── dataset.py            # PyTorch Dataset with resnet/lstm modes
│
├── models/                   # Trained weight files
│   ├── resnet_audio_model.pth
│   └── lstm_audio_model.pth
│
├── dataset/                  # Training data (not committed)
│   └── train/
│       ├── real/             # Real audio files
│       └── fake/             # AI-generated audio files
│
└── README.md                 # This file
```

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Node.js 18+ and npm
- (Optional) NVIDIA GPU with CUDA for faster training

### 1. Clone the repository

```bash
git clone <repository-url>
cd "DL Project"
```

### 2. Backend setup

```bash
cd backend
pip install -r requirements.txt
```

### 3. Frontend setup

```bash
cd frontend
npm install
```

### 4. Train models

Place your audio dataset under `dataset/train/real/` and `dataset/train/fake/`, then:

```bash
cd training

# Train ResNet-18
python train.py --model resnet --epochs 25 --batch-size 16

# Train LSTM
python train.py --model lstm --epochs 25 --batch-size 16
```

Model weights will be saved to `models/resnet_audio_model.pth` and `models/lstm_audio_model.pth`.

### 5. Run the backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Run the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Model Details

### ResNet-18 (Spectrogram Image Classifier)

| Property       | Value                                     |
|----------------|-------------------------------------------|
| Architecture   | ResNet-18 (modified: 1-channel input, 2-class output) |
| Input          | Mel spectrogram image `(1, 224, 224)`     |
| Features       | 128 mel bands, 1024 FFT, 512 hop length   |
| Preprocessing  | Log-dB, z-score normalization, bilinear resize |
| Activations    | ReLU (in every residual block and after conv1) |
| Output         | 2 logits → softmax → [P(Real), P(Fake)]  |

### LSTM (Temporal Sequence Classifier)

| Property       | Value                                     |
|----------------|-------------------------------------------|
| Architecture   | 2-layer Bidirectional LSTM + FC classifier |
| Input          | Mel-spectrogram time series `(94, 128)`   |
| Features       | Same mel spectrogram (transposed, no resize) |
| Hidden dim     | 128 per direction (256 total)             |
| Gate activations | Sigmoid (input/forget/output gates), Tanh (cell state) |
| Classifier activation | ReLU (in FC head between Linear layers) |
| Dropout        | 0.3                                       |
| Output         | 2 logits → softmax → [P(Real), P(Fake)]  |

### Loss Function

Both models are trained with **CrossEntropyLoss** (`torch.nn.CrossEntropyLoss`).

CrossEntropyLoss combines `LogSoftmax` and `NLLLoss` in one step:

```
L = -log( exp(logits[y]) / Σ exp(logits[j]) )
```

Where `y` is the true class index (0 = Real, 1 = AI Generated). This is the standard loss for multi-class classification with raw logits.

### Activation Functions Summary

| Location | Activation | Purpose |
|----------|-----------|---------|
| ResNet BasicBlocks | **ReLU** | Non-linearity after each Conv+BN pair |
| ResNet Conv1 | **ReLU** | Non-linearity after first convolution |
| LSTM gates (internal) | **Sigmoid** | Controls information flow (input, forget, output gates) |
| LSTM cell state (internal) | **Tanh** | Squashes cell state and candidate values to [-1, 1] |
| LSTM classifier FC head | **ReLU** | Non-linearity between the two Linear layers |
| Inference (both models) | **Softmax** | Converts raw logits to probabilities for prediction |

> **Note:** Softmax is applied only at inference time (`model_loader.py`). During training, `CrossEntropyLoss` handles the softmax internally, so the models output raw logits.

### Ensemble Strategy

Both models produce softmax probability vectors `[P(Real), P(Fake)]`. The ensemble computes a weighted average:

```
ensemble_probs = w_resnet × resnet_probs + w_lstm × lstm_probs
```

Default weights: `w_resnet = 0.5`, `w_lstm = 0.5` (configurable in `backend/config.py`).

---

## API Documentation

### `GET /health`

Health check. Returns model loading status.

**Response:**
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

Upload an audio file and receive an ensemble prediction.

**Request:** Multipart form data with a `file` field containing the audio file.

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

**Error responses:**
- `400` — Invalid file (empty, wrong type, too large)
- `422` — Audio processing error
- `503` — No models loaded
- `500` — Internal server error

---

## Screenshots

> Screenshots will be added after the UI is finalized. The dashboard shows:
> - Audio upload card with preview player
> - Mel spectrogram visualization
> - Three prediction cards: ResNet-18, LSTM, and Ensemble
> - Confidence bars with gradient colors

---

## Future Improvements

- Add support for longer audio clips with sliding-window analysis.
- Implement attention mechanisms in the LSTM for interpretability.
- Add a third model (e.g., wav2vec2) for a stronger ensemble.
- Provide Grad-CAM visualizations on the spectrogram.
- Deploy to cloud with Docker containerization.
- Add user authentication and prediction history.
- Support real-time microphone input.
- Fine-tune ensemble weights using a validation set (learned fusion).
