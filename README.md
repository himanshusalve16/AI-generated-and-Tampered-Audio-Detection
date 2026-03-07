## AI-Generated and Tampered Audio Detection

**AI-Generated and Tampered Audio Detection using Spectral Feature Analysis and Deep Learning**

This project is a full‑stack system that detects whether an uploaded audio file is:

- **Real human speech**, or  
- **AI-generated / tampered speech**

It combines:

- **Frontend**: React (Vite) + Tailwind CSS + GSAP + Lucide React (modern animated UI)
- **Backend**: FastAPI (REST API + inference)
- **Audio Processing**: Librosa (mel‑spectrograms)
- **Deep Learning**: PyTorch + ResNet‑18 (binary classifier)
- **Training Module**: Separate training pipeline for reproducible experiments

---

## 1. High-Level Architecture

**Flow:**

1. User uploads `.wav` / `.mp3` in the **React frontend**.
2. Frontend sends the file via `POST /predict` to the **FastAPI backend**.
3. Backend:
   - Preprocesses audio into a **log-mel spectrogram** tensor.
   - Runs inference with a **ResNet‑18** binary classifier.
   - Returns JSON with **prediction** (`"Real"` / `"AI-Generated"`) and **confidence**.
4. Frontend displays the result with a dark, animated UI.

**Ports:**

- **Frontend**: `http://localhost:5173`
- **Backend**: `http://localhost:8000`

---

## 2. Project Structure

```text
DL Project/
  frontend/          # React (Vite) SPA UI (do not modify)
    src/
      main.jsx
      App.jsx
      index.css
      components/
        UploadCard.jsx
        ResultCard.jsx
    package.json
    vite.config.mjs
    tailwind.config.mjs
    postcss.config.mjs
    index.html

  backend/           # FastAPI API + inference
    main.py          # App, CORS, /health, /predict, lifespan, logging, validation
    preprocess.py    # Audio → mel spectrogram → (1, 224, 224) tensor
    model_loader.py  # ResNet-18 load + predict() with softmax
    utils.py         # Logging, file validation, constants, paths
    requirements.txt

  training/          # Training-only code (no API)
    dataset.py       # AudioDataset (real/fake), same preprocessing as backend
    model.py         # AudioResNet (ResNet-18, 1-channel, 2 classes)
    train.py         # Train/val loops, metrics, CLI, best model save

  dataset/           # (You create this)
    real/            # Real speech audio clips
    fake/            # AI-generated / tampered clips

  models/
    audio_model.pth  # (Created after training; used by backend at startup)
```

- **frontend/** – UI only (upload, preview, call `/predict`, display result)  
- **backend/** – API + inference only (modular, with logging and validation)  
- **training/** – training pipeline only (dataset, model, train script with CLI)  
- **models/** – saved weights only  

---

## 3. Frontend (React + Vite + Tailwind + GSAP)

### 3.1 Tech Stack

- **React 18** – functional components and hooks
- **Vite** – fast dev server and build
- **Tailwind CSS** – utility-first modern styling
- **GSAP** – smooth entrance and result animations
- **Lucide React** – icons
- **Axios** – HTTP client for API calls

### 3.2 Features

- Single page app:
  - **File input** for `.wav` / `.mp3`
  - **Audio preview** using `<audio>` element
  - **Analyze audio** button
  - **Loading state** while prediction is running
  - **Prediction display**: “Real” or “AI-Generated”
  - **Confidence score** display (percentage + bar)
- Modern dark UI:
  - Gradient background + glassmorphism center card
  - GSAP entrance animation for the main card
  - Animated button hover/click
  - Fade-in animation when prediction appears
  - Icons for upload, audio file, status, and confidence

### 3.3 Frontend API Contract

- **Endpoint**: `POST http://localhost:8000/predict`
- **Request**:
  - `Content-Type: multipart/form-data`
  - Field: `file` (audio file)
- **Response (backend returns)**:

```json
{
  "label": "Real",
  "confidence": 0.93
}
```

The frontend reads `label` and `confidence` from the response to display the prediction and confidence score.

### 3.4 Running the Frontend

From the project root:

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite (typically `http://localhost:5173`).

---

## 4. Backend (FastAPI + Librosa + PyTorch)

### 4.1 Tech Stack

- **FastAPI** – web framework with request validation and error handling
- **Uvicorn** – ASGI server
- **Librosa** – audio loading and mel-spectrogram extraction
- **NumPy** – array operations
- **PyTorch** + **Torchvision** – ResNet-18 and tensors

The backend uses a **modular architecture**: `utils.py` provides structured logging, file validation (size, extension, content type), and path constants; `main.py` wires endpoints and lifecycle; `preprocess.py` and `model_loader.py` handle audio and inference.

### 4.2 Installing Backend Dependencies

From the project root:

```bash
cd backend
pip install -r requirements.txt
```

### 4.3 Running the Backend

From `backend/`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API base URL: `http://localhost:8000`

### 4.4 CORS Configuration

`main.py` configures CORS so the React frontend can call the API:

- **Allowed origins**: `http://localhost:5173` (from `utils.CORS_ORIGINS`)
- **Methods**: GET, POST, OPTIONS
- **Headers**: all

---

## 5. Backend Endpoints

### 5.1 `GET /health`

- Health check for monitoring and load balancers.
- Response:

```json
{ "status": "ok" }
```

### 5.2 `POST /predict`

- **Parameter**: `file` (UploadFile) – audio file (e.g. WAV, MP3, FLAC).
- **Request validation** (in `utils.validate_audio_upload`):
  - Non-empty filename and file size > 0.
  - Max size: 50 MB.
  - Allowed extensions: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.webm`.
  - Allowed content types: common `audio/*` and `application/octet-stream`.
- **Processing steps**:
  1. Validate uploaded file (size, type, extension).
  2. Load audio and preprocess via `preprocess_audio()` → tensor `(1, 224, 224)`.
  3. Run `model_loader.predict(tensor)` → `(label, confidence)`.
  4. Return JSON (frontend expects `label` and `confidence`):

```json
{
  "label": "Real",
  "confidence": 0.85
}
```

- **If the model is not loaded** (e.g. `models/audio_model.pth` missing): the server still starts, but `/predict` returns **503** with a message to train and place the model, then restart.
- **Error handling**: validation errors (400), processing/inference errors (422), model-not-loaded (503), and unexpected errors (500). All errors include a `detail` message; structured logging records failures.

---

## 6. Audio Preprocessing (`backend/preprocess.py`)

Preprocessing is **modular**: each step is a small function so the pipeline is easy to follow and keep in sync with the training dataset.

### 6.1 Constants (must match `training/dataset.py`)

- `TARGET_SR = 16000`, `TARGET_DURATION_SEC = 3.0`, `TARGET_NUM_SAMPLES = 48000`
- `TARGET_SIZE = 224` (output height and width)
- `N_FFT = 1024`, `HOP_LENGTH = 512`, `N_MELS = 128`, `POWER = 2.0`, `TRIM_TOP_DB = 20`

### 6.2 Function: `preprocess_audio(file)`

- **Input**: `file` can be a binary file-like object, raw `bytes`, or a `str` path to an audio file.
- **Output**: `torch.Tensor` of shape `(1, 224, 224)`, float32, CPU.

**Steps (implemented as helpers):**

1. **Load & resample** – `librosa.load(..., sr=TARGET_SR, mono=True)` (16 kHz, mono).
2. **Trim silence** – `_trim_silence(y)` using `librosa.effects.trim(y, top_db=TRIM_TOP_DB)`.
3. **Pad / crop** – `_pad_or_crop_to_fixed_length(y, TARGET_NUM_SAMPLES)` (3 seconds).
4. **Mel spectrogram** – `_compute_mel_spectrogram(y)` with the constants above.
5. **Log scale** – `_to_log_scale(mel_spec)` → `librosa.power_to_db(..., ref=np.max)`.
6. **Normalize** – `_normalize_spectrogram(mel_db)` (zero mean, unit variance per spectrogram).
7. **To tensor and resize** – `_resize_to_target(..., 224, 224)` with `F.interpolate` (bilinear).
8. **Return** – tensor shape `(1, 224, 224)` for the ResNet-18 input.

---

## 7. Model Loading & Inference (`backend/model_loader.py`)

### 7.1 Model Architecture

- Base model: `resnet18(weights=None)` from `torchvision.models`.
- Modifications:
  - **First convolution**: 1-channel instead of 3-channel RGB.
  - **Final fully connected layer**: 2 outputs (binary classification).

This architecture is also used in the training module.

### 7.2 Weights and Device

- Weights file:

```text
models/audio_model.pth
```

- Device:
  - Uses GPU (`cuda`) if available, else CPU.

### 7.3 One-Time Model Loading

- `load_model()`:
  - Builds the modified ResNet‑18.
  - Loads `state_dict` from `audio_model.pth`.
  - Moves model to the correct device.
  - Sets `model.eval()`.
  - Caches the model in a module-level variable so it is **not reloaded per request**.
  - Called once on FastAPI startup.

### 7.4 Prediction Function

```python
def predict(tensor: torch.Tensor) -> Tuple[str, float]:
    # tensor shape: (1, 224, 224)
    # 1. Add batch dimension -> (1, 1, 224, 224), move to device
    # 2. Forward pass (logits)
    # 3. Softmax -> probabilities
    # 4. Argmax for class index, max prob for confidence
    # 5. Map index to CLASS_LABELS: 0 -> "Real", 1 -> "AI Generated"
```

- **Returns**: `(label, confidence)` where `label` is `"Real"` or `"AI Generated"` and `confidence` is in [0, 1].
- **Raises**: `RuntimeError` if the model was not loaded; `ValueError` if the tensor shape is not `(1, 224, 224)`.

---

## 8. Training Pipeline (`training/`)

Training code is **separate** from the backend and not imported by the API.

### 8.1 Dataset (`training/dataset.py`)

Expected folder structure:

```text
dataset/
  real/   # label 0 (Real)
  fake/   # label 1 (AI Generated / Fake)
```

**AudioDataset**:

- Constructor: `root_dir`, optional `real_subdir`/`fake_subdir`, and optional `extensions` filter.
- Scans both subdirs and builds a list of `(path, label)`.
- **Preprocessing** is identical to `backend/preprocess.py`: same constants and step-by-step helpers (`_trim_silence`, `_pad_or_crop_to_fixed_length`, mel spectrogram, log scale, normalize, resize). The shared helper is `waveform_to_tensor(y)`.
- `__getitem__(idx)` loads the file with librosa (sr=16000, mono), then `waveform_to_tensor(y)` → tensor `(1, 224, 224)` and integer label 0 or 1.
- Raises `FileNotFoundError` if `real/` or `fake/` is missing, and `ValueError` if no files are found.

### 8.2 Model (`training/model.py`)

- `AudioResNet`:
  - Wraps modified ResNet‑18:
    - Input: 1-channel
    - Output: 2 classes
  - `forward(x)` simply calls the underlying ResNet.

### 8.3 Training Script (`training/train.py`)

**Default configuration:**

- `dataset_root` = project root `dataset/`
- `models_dir` = project root `models/`
- `batch_size = 16`
- `num_epochs = 25`
- `learning_rate = 1e-4`
- `val_split = 0.2`
- `num_workers = 0`
- Random seed: `42` (for reproducibility)

**CLI options:**

```bash
python train.py [--dataset PATH] [--models-dir PATH] [--batch-size N] [--epochs N] [--lr FLOAT] [--val-split FLOAT] [--num-workers N] [--verbose]
```

**Training steps:**

1. Set up logging and random seed (numpy, torch, CUDA if available).
2. Load `AudioDataset`, split with `random_split` into train/validation.
3. Create `DataLoader`s (train shuffled, val not).
4. Instantiate `AudioResNet`, `Adam`, `CrossEntropyLoss`.
5. For each epoch:
   - **Train**: `run_epoch_train()` – forward, loss, backward, step; aggregate loss and accuracy.
   - **Validation**: `run_epoch_val()` – no grad, aggregate loss and accuracy.
   - Log: `Epoch xxx/xxx | Train Loss: ... | Train Acc: ... | Val Loss: ... | Val Acc: ...`
   - If validation accuracy improves: save `model.state_dict()` to `models/audio_model.pth` and log that the best model was saved.
6. Final log: best validation accuracy and path to saved model.

**Run training** (from project root or `training/`):

```bash
cd training
python train.py
# Or with options:
python train.py --epochs 30 --batch-size 32 --lr 5e-5
```

After training, place or keep `audio_model.pth` in `models/` and (re)start the backend so it loads the new weights at startup.

---

## 9. End-to-End Setup Instructions

1. **Clone / open the project folder** (already at `d:\DL Project`).
2. **(Optional but recommended) Create a virtual environment**:

```bash
cd "d:\DL Project"
python -m venv .venv
.venv\Scripts\activate    # On Windows PowerShell
```

3. **Prepare dataset**:

```text
d:\DL Project\dataset\real\   # real speech audio files
d:\DL Project\dataset\fake\   # AI / tampered audio files
```

4. **Install backend + training dependencies**:

```bash
cd backend
pip install -r requirements.txt
```

5. **(Optional) Train the model**:

```bash
cd ../training
python train.py
```

Confirm that `../models/audio_model.pth` is created.

6. **Run FastAPI backend**:

```bash
cd ../backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Check health:

```text
http://localhost:8000/health   -> { "status": "ok" }
```

7. **Install and run frontend**:

```bash
cd ../frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

8. **Use the system**:

- Upload a `.wav` or `.mp3` speech clip in the UI.
- Click **“Analyze audio”**.
- The frontend sends a request to `POST http://localhost:8000/predict`.
- The backend preprocesses the audio, runs ResNet‑18, and responds with:

```json
{
  "label": "Real",
  "confidence": 0.85
}
```

- The frontend displays the label and confidence with smooth animations.

---

## 10. How to Explain in Viva

- **Core idea**:  
  Convert raw audio into a **mel-spectrogram image** and use a **CNN (ResNet‑18)** to classify real vs AI-generated/tampered speech.

- **Spectral features**:
  - Mel-spectrogram captures **time–frequency** patterns of speech.
  - Log scaling and normalization stabilize inputs for the neural network.

- **Model**:
  - ResNet‑18 is a standard CNN architecture, modified for 1-channel input and 2 outputs.
  - Training uses **CrossEntropyLoss** and **Adam** optimizer.

- **Clean architecture**:
  - `frontend/` (UI), `backend/` (API + inference), `training/` (training), `models/` (weights).
  - Same preprocessing in training and inference.
  - Model is loaded once at backend startup for efficiency.

- **Full-stack integration**:
  - React frontend → FastAPI backend via REST.
  - CORS allows only the frontend origin.
  - Responses are simple JSON, easy to log and interpret.

This structure is minimal, modular, and ready to be explained step-by-step during viva or project demonstration.

