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
  frontend/          # React (Vite) SPA UI
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
    main.py          # API, CORS, endpoints, startup model load
    preprocess.py    # Audio preprocessing -> (1, 224, 224) tensor
    model_loader.py  # ResNet-18 model + predict()
    requirements.txt

  training/          # Training-only code (no API)
    dataset.py       # AudioDataset (dataset/real, dataset/fake)
    model.py         # AudioResNet (ResNet-18, 1-channel, 2 classes)
    train.py         # Training loop, validation, saves best model

  dataset/           # (You create this)
    real/            # Real speech audio clips
    fake/            # AI-generated / tampered clips

  models/
    audio_model.pth  # (Created after training; used by backend)
```

- **frontend/** – UI only  
- **backend/** – API + inference only  
- **training/** – training pipeline only  
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
- **Response (expected)**:

```json
{
  "prediction": "Real",
  "confidence": 0.93
}
```

The frontend uses this response to render the label and confidence.

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

- **FastAPI** – web framework for Python
- **Uvicorn** – ASGI server
- **Librosa** – audio loading and feature extraction
- **NumPy**
- **PyTorch** + **Torchvision** – model and tensors

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

`main.py` configures CORS to allow the React frontend:

- **Allowed origin**: `http://localhost:5173`
- **Methods**: all
- **Headers**: all

---

## 5. Backend Endpoints

### 5.1 `GET /health`

- Simple health check.
- Response:

```json
{ "status": "ok" }
```

### 5.2 `POST /predict`

- Accepts an uploaded file:
  - Parameter name: `file`
  - Type: `UploadFile`
  - Expected content: `.wav` / `.mp3` (`audio/*` content type)
- Processing steps:
  1. Validate file presence and type.
  2. Call `preprocess_audio(file.file)` → tensor of shape `(1, 224, 224)`.
  3. Call `model_loader.predict(tensor)` → `(label, confidence)`.
  4. Return JSON:

```json
{
  "prediction": "Real",
  "confidence": 0.85
}
```

- If model weights are missing (`models/audio_model.pth` not found), the backend returns a **mock-like** response:
  - `prediction`: `"Real"`
  - `confidence`: `0.85`

Errors during preprocessing or inference are wrapped as `HTTPException` with clear messages.

---

## 6. Audio Preprocessing (`backend/preprocess.py`)

### 6.1 Function: `preprocess_audio(file)`

Signature:

```python
def preprocess_audio(file) -> torch.Tensor:
    # returns tensor of shape (1, 224, 224)
```

**Steps:**

1. **Load & resample**
   - Read raw bytes from `file`.
   - Wrap in `io.BytesIO` and call `librosa.load(buffer, sr=16000, mono=True)` to:
     - Decode audio
     - Convert to mono
     - Resample to **16 kHz**
2. **Trim silence**
   - `librosa.effects.trim(y, top_db=20)` removes leading and trailing silence.
3. **Pad / cut to 3 seconds**
   - Target duration: 3 seconds at 16 kHz → 48,000 samples.
   - If shorter: zero-pad at the end.
   - If longer: take a centered 3-second window.
4. **Mel-spectrogram**
   - Use `librosa.feature.melspectrogram` with:
     - `n_fft=1024`, `hop_length=512`, `n_mels=128`, `power=2.0`.
5. **Log scale**
   - Convert to decibel scale with `librosa.power_to_db(mel, ref=np.max)`.
6. **Normalize**
   - Standardize per spectrogram: subtract mean and divide by standard deviation.
7. **Resize to 224×224**
   - Convert to torch tensor `(1, n_mels, time)`.
   - Add batch dimension → `(1, 1, n_mels, time)`.
   - Use `torch.nn.functional.interpolate` (bilinear) to `(1, 1, 224, 224)`.
8. **Return**
   - Remove batch dimension → final tensor shape `(1, 224, 224)`.

This is the exact input shape expected by the ResNet‑18 model.

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
def predict(tensor):
    # tensor shape: (1, 224, 224)
    # 1. Add batch dimension -> (1, 1, 224, 224)
    # 2. Forward pass
    # 3. Apply softmax
    # 4. Return label ("Real"/"AI-Generated") and confidence in [0, 1]
```

- Output:
  - **Label**: `"Real"` if class index 0, else `"AI-Generated"`.
  - **Confidence**: maximum softmax probability (float).

---

## 8. Training Pipeline (`training/`)

Training code is **separate** from the backend and not imported by the API.

### 8.1 Dataset (`training/dataset.py`)

Expected folder structure:

```text
dataset/
  real/   # label 0
  fake/   # label 1
```

**AudioDataset**:

- Scans `real/` and `fake/` for files.
- For each file:
  - Assigns label `0` (real) or `1` (fake/AI).
  - Applies the **same preprocessing** as `preprocess_audio`:
    - Load with librosa (sr=16000, mono)
    - Trim silence
    - Pad/cut to 3 seconds
    - Mel-spectrogram → log → normalize → resize to 224×224
- `__getitem__` returns:
  - Tensor of shape `(1, 224, 224)`
  - Integer label (0 or 1)

### 8.2 Model (`training/model.py`)

- `AudioResNet`:
  - Wraps modified ResNet‑18:
    - Input: 1-channel
    - Output: 2 classes
  - `forward(x)` simply calls the underlying ResNet.

### 8.3 Training Script (`training/train.py`)

Default configuration:

- `dataset_root = "../dataset"`
- `models_dir = "../models"`
- `batch_size = 8`
- `num_epochs = 10`
- `learning_rate = 1e-4`
- `val_split = 0.2`

Training steps:

1. Select device (`cuda` if available).
2. Instantiate `AudioDataset`.
3. Split dataset into train/validation using `random_split`.
4. Create `DataLoader`s for train and validation.
5. Instantiate `AudioResNet`, `Adam` optimizer, `CrossEntropyLoss`.
6. For each epoch:
   - **Train loop**:
     - Forward pass, compute loss, backward, optimizer step.
     - Track training loss and accuracy.
   - **Validation loop**:
     - Evaluate on validation set.
     - Track validation loss and accuracy.
   - Print a concise summary per epoch:

```text
Epoch 01/10 - Train Loss: 0.6931 | Train Acc: 0.500 - Val Loss: 0.6900 | Val Acc: 0.550
```

7. If validation accuracy improves:
   - Save `model.state_dict()` to `../models/audio_model.pth`.
   - Print a message indicating a new best model has been saved.

Run training from the `training/` directory:

```bash
cd training
python train.py
```

After successful training, the backend will automatically pick up `models/audio_model.pth` for inference.

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
  "prediction": "Real",
  "confidence": 0.85
}
```

- The frontend shows the label and confidence with smooth animations.

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

