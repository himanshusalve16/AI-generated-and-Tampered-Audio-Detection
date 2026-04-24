# LSTM — Bidirectional LSTM for Audio Deepfake Detection


## AudioLSTM class details (`training/model.py`)

```python
class AudioLSTM(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2,
                 num_classes=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,        # 128 mel bands per time step
            hidden_size=hidden_dim,      # 128 hidden units per direction
            num_layers=num_layers,       # 2 stacked LSTM layers
            batch_first=True,
            bidirectional=True,          # forward + backward = 256 output
            dropout=dropout,             # 0.3 between layers
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),         # regularization
            nn.Linear(hidden_dim * 2, 64),  # 256 → 64
            nn.ReLU(),                   # non-linearity
            nn.Dropout(dropout),         # regularization
            nn.Linear(64, num_classes),  # 64 → 2 (Real vs AI Generated)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)       # (B, T, 256)
        last = lstm_out[:, -1, :]        # (B, 256) — last time step
        logits = self.classifier(last)   # (B, 2) — raw logits
        return logits
```

---

## Why LSTM for audio deepfake detection?

- Audio is a **temporal signal** — patterns unfold over time.
- AI-generated speech often has subtle **temporal inconsistencies** that are hard to spot in individual spectral frames but visible across a sequence.
- LSTM's gated memory mechanism can learn to detect **long-range temporal artifacts** like unnatural pitch transitions, repeated patterns, or missing micro-pauses.
- A **bidirectional** LSTM looks at the sequence from both directions, capturing context from both past and future frames.

---

## Architecture Overview

```
Input: (B, 94, 128)          ← 94 time steps × 128 mel bands
         │
    ┌────▼────┐
    │ LSTM L1 │  Forward → (B, 94, 128)
    │ BiDir   │  Backward → (B, 94, 128)
    │         │  Combined → (B, 94, 256)
    └────┬────┘
         │ Dropout (0.3)
    ┌────▼────┐
    │ LSTM L2 │  Forward → (B, 94, 128)
    │ BiDir   │  Backward → (B, 94, 128)
    │         │  Combined → (B, 94, 256)
    └────┬────┘
         │ Take last time step → (B, 256)
         │
    ┌────▼────┐
    │ Dropout │  (B, 256)
    │ Linear  │  256 → 64
    │ ReLU    │  (B, 64)
    │ Dropout │  (B, 64)
    │ Linear  │  64 → 2
    └────┬────┘
         │
    Output: (B, 2)              ← raw logits (Real vs AI Generated)
```

---

## Input Preparation

The LSTM receives the mel spectrogram **without** the 224×224 resize that ResNet uses:

1. Audio is loaded and resampled to 16 kHz mono.
2. Silence is trimmed, then padded/cropped to 3 seconds (48,000 samples).
3. Mel spectrogram is computed: shape `(128 mel bands, T time frames)`.
4. With `N_FFT=1024` and `HOP_LENGTH=512`: $T = 1 + \lfloor 48000 / 512 \rfloor = 94$.
5. Log-dB conversion and z-score normalization.
6. **Transpose** to `(T, 128)` = `(94, 128)` — each time step is a 128-dim feature vector.

This preserves the original temporal resolution, which is critical for the LSTM.

---

## LSTM Cell Mathematics

At each time step $t$, the LSTM computes:

| Gate | Formula | Activation | Purpose |
|------|---------|-----------|---------|
| Forget | $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$ | **Sigmoid** | What to discard from memory |
| Input | $i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$ | **Sigmoid** | What new info to store |
| Candidate | $\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$ | **Tanh** | New candidate values |
| Cell update | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ | — | Update memory |
| Output | $o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$ | **Sigmoid** | What to output |
| Hidden | $h_t = o_t \odot \tanh(c_t)$ | **Tanh** | Final output |

Where $\sigma$ = sigmoid, $\tanh$ = hyperbolic tangent, $\odot$ = element-wise multiply.

---

## Bidirectional Processing

Two separate LSTMs process the same sequence:

- **Forward**: $x_1, x_2, \dots, x_{94}$ → $\overrightarrow{h}_1, \dots, \overrightarrow{h}_{94}$
- **Backward**: $x_{94}, x_{93}, \dots, x_1$ → $\overleftarrow{h}_1, \dots, \overleftarrow{h}_{94}$

At each step, outputs are concatenated:

$$
h_t = [\overrightarrow{h}_t ; \overleftarrow{h}_t] \in \mathbb{R}^{256}
$$

This gives the model access to **both past and future context** at every time step.

---

## Activation Functions

| Location | Activation | Output Range | Purpose |
|----------|-----------|-------------|---------|
| Forget/Input/Output gates | **Sigmoid** | $(0, 1)$ | Soft switches controlling info flow |
| Cell state / candidate | **Tanh** | $(-1, 1)$ | Squash values symmetrically |
| Classifier FC head | **ReLU** | $[0, \infty)$ | Non-linearity between Linear layers |
| Inference | **Softmax** | $(0, 1)$, sums to 1 | Convert logits to probabilities |

---

## Loss Function

**CrossEntropyLoss** (`torch.nn.CrossEntropyLoss`), same as ResNet:

$$
L = -\log\left(\frac{e^{z_y}}{\sum_{j=0}^{1} e^{z_j}}\right)
$$

- Accepts raw logits (no softmax in model output during training).
- $y$ = true class: 0 (Real) or 1 (AI Generated).

---

## Model Parameters

| Component | Parameters | Details |
|-----------|-----------|---------|
| LSTM Layer 1 | ~330K | input_dim=128, hidden_dim=128, bidirectional |
| LSTM Layer 2 | ~530K | input_dim=256 (from BiDir L1), hidden_dim=128 |
| Classifier | ~16.5K + 130 | Linear(256→64) + Linear(64→2) |
| **Total** | **~400K** | Much smaller than ResNet (~11M) |

---

## Comparison: LSTM vs ResNet

| Aspect | ResNet-18 | LSTM |
|--------|-----------|------|
| Input | 2D spectrogram image (1, 224, 224) | 1D temporal sequence (94, 128) |
| Type | CNN (Convolutional Neural Network) | RNN (Recurrent Neural Network) |
| What it learns | Spatial texture patterns | Temporal dynamics |
| Activation (main) | ReLU | Sigmoid + Tanh (gates), ReLU (head) |
| Parameters | ~11M | ~400K |
| Strength | Frequency-domain artifacts | Temporal inconsistencies |

---

## Training

```bash
cd training
python train.py --model lstm --epochs 25 --batch-size 16 --lr 1e-4
```

Saves weights to: `models/lstm_audio_model.pth`

---

## Data Flow Summary

```
Audio File
  → Resample 16kHz
  → Trim silence
  → Pad/crop to 3 seconds (48,000 samples)
  → Mel spectrogram (128, 94)
  → Log-dB + Normalize
  → Transpose to (94, 128)
  → LSTM Layer 1 (BiDir) → (94, 256)
  → Dropout → LSTM Layer 2 (BiDir) → (94, 256)
  → Last time step → (256,)
  → Dropout → Linear → ReLU → Dropout → Linear → (2,) logits
  → Softmax (inference) → [P(Real), P(AI Generated)]
```
