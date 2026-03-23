# Mathematical Calculation

This document contains all the detailed mathematical calculations for the project, including audio preprocessing, ResNet-18 operations, pooling, and classification.

## 1. Audio Preprocessing Mathematics

### 1.1 Audio Loading and Resampling
- Input: raw audio file (WAV, MP3, etc.) with sample rate $f_s$ Hz.
- Resampling: convert to target sample rate $f_s' = 16000$ Hz.
- Mono conversion: $y_{\text{mono}}[n] = \frac{y_{\text{left}}[n] + y_{\text{right}}[n]}{2}$.
- Output: waveform $y[n]$, where $n = 0, 1, \dots, N-1$, $N = f_s' \times \text{duration}$.

### 1.2 Silence Trimming
- Algorithm: `librosa.effects.trim` with `top_db = 20`.
- Keep frames where:

$$
20 \log_{10}\left(\frac{|y[n]|}{\max|y|}\right) > -20.
$$

### 1.3 Fixed-Length Padding/Cropping
- Target $N_{\text{target}} = 16000 \times 3 = 48000$ samples.
- If $\text{len}(y) < N_{\text{target}}$:

$$
y_{\text{padded}}[n] = \begin{cases}
  y[n], & n < N \\ 
  0, & n \ge N
\end{cases}
$$

- If $\text{len}(y) > N_{\text{target}}$:

$$
\text{start} = \left\lfloor\frac{\text{len}(y) - N_{\text{target}}}{2}\right\rfloor,\qquad
y_{\text{cropped}}[n] = y[\text{start} + n].
$$

### 1.4 Mel Spectrogram Computation
- STFT parameters: $N_{\text{FFT}} = 1024$, $\text{HOP_LENGTH} = 512$, window = Hann.
- Hann window:

$$
w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N_{\text{FFT}} - 1}\right).
$$

- STFT:

$$
X[m, k] = \sum_{n=0}^{N_{\text{FFT}}-1} y[n + mH] \cdot w[n] \cdot e^{-j 2\pi k n / N_{\text{FFT}}}.
$$

- Power spectrogram:

$$
P[m, k] = |X[m, k]|^2.
$$

- Mel filterbank: $N_{\text{mels}} = 128$.
- Mel frequency conversion:

$$
m = 2595 \log_{10}\left(1 + \frac{f}{700} \right).
$$

- Mel spectrogram:

$$
S[m, t] = \sum_{k} P[t, k] \cdot F[k, m],
$$

where $F[k,m]$ is the mel filterbank matrix.

### 1.5 Log Scaling (Decibels)

$$
S_{\text{dB}}[m, t] = 10 \log_{10}\left(\frac{S[m, t]}{\max(S)}\right).
$$

### 1.6 Normalization
- Mean:

$$
\mu = \frac{1}{M T} \sum_{m=1}^{M} \sum_{t=1}^{T} S_{\text{dB}}[m, t].
$$

- Standard deviation:

$$
\sigma = \sqrt{\frac{1}{M T} \sum_{m=1}^{M} \sum_{t=1}^{T} \left(S_{\text{dB}}[m, t] - \mu\right)^2}.
$$

- Z-score normalization:

$$
S_{\text{norm}}[m, t] =
\begin{cases}
\frac{S_{\text{dB}}[m, t] - \mu}{\sigma}, & \sigma > 0 \\ 
S_{\text{dB}}[m, t] - \mu, & \sigma = 0
\end{cases}
$$

### 1.7 Spatial Resizing
- Input shape: $(128, T)$.
- Target shape: $(224, 224)$ via bilinear interpolation.
- Output shape after preprocessing: $(1, 224, 224)$.

## 2. ResNet-18 Model Mathematics

### 2.1 Input/Output
- Input tensor: $x \in \mathbb{R}^{1 \times 224 \times 224}$.
- Output logits: $\mathrm{logits} \in \mathbb{R}^{2}$ (Real vs AI).

### 2.2 Feature Extraction Layers

#### 2.2.1 Conv1 + BatchNorm + ReLU + MaxPool
- Conv1 weights: $W_1 \in \mathbb{R}^{64 \times 1 \times 7 \times 7}$, stride $=2$, padding $=3$.
- Output dims:

$$
\text{out	n} = \left\lfloor\frac{224 + 2\cdot3 - 7}{2} \right\rfloor + 1 = 112.
$$

- BatchNorm:

$$
\hat{f} = \gamma \frac{f - \mu_{\text{batch}}}{\sqrt{\sigma_{\text{batch}}^2 + \epsilon}} + \beta.
$$

- ReLU:

$$
a_1 = \max(0, \hat{f}).
$$

- MaxPool1 ($3\times3$, stride $=2$, padding $=1$): output dims $56 \times 56$.

#### 2.2.2 Conv2_x block (no downsample)
- Input dims: $64 \times 56 \times 56$.
- Two conv layers, each $3 \times 3$, 64 filters, stride $=1$, padding $=1$.
- Residual path:

$$
\begin{aligned}
  z_1 &= \text{ReLU}(\text{BN}(\text{Conv}(x))), \\
  z_2 &= \text{BN}(\text{Conv}(z_1)), \\
  y &= \text{ReLU}(z_2 + x).
\end{aligned}
$$

#### 2.2.3 Conv3_x block (downsample)
- Input dims: $64 \times 56 \times 56$.
- Main path first conv: 128 filters, stride $=2$ $\to$ $128 \times 28 \times 28$.
- Shortcut path: 1x1 conv, $64 \to 128$, stride $=2$.
- Output after addition and ReLU: $128 \times 28 \times 28$.

#### 2.2.4 Conv4_x block (downsample)
- Input dims: $128 \times 28 \times 28$.
- Main path output: $256 \times 14 \times 14$.
- Shortcut: 1x1 conv, $128 \to 256$, stride $=2$.

#### 2.2.5 Conv5_x block (downsample)
- Input dims: $256 \times 14 \times 14$.
- Main path output: $512 \times 7 \times 7$.
- Shortcut: 1x1 conv, $256 \to 512$, stride $=2$.

### 2.3 Global Average Pooling

$$
g[c] = \frac{1}{7 \times 7} \sum_{i=1}^{7} \sum_{j=1}^{7} a_{\text{res4}}[c, i, j].
$$

Output: $g \in \mathbb{R}^{512}$.

### 2.4 Fully Connected Output
- $W_{\text{fc}} \in \mathbb{R}^{2 \times 512}$, $b_{\text{fc}} \in \mathbb{R}^{2}$.

$$
\mathrm{logits} = W_{\text{fc}} g + b_{\text{fc}}.
$$

## 3. Pooling and Classification

### 3.1 Pooling Operations
- MaxPool1: Feature reduction via local maxima.
- GlobalAvgPool: channel-wise mean to 512 dimensions.

### 3.2 Softmax and Prediction
- Softmax probabilities:

$$
p_i = \frac{e^{\text{logits}_i}}{\sum_{j=0}^{1} e^{\text{logits}_j}},\qquad i \in \{0,1\}.
$$

- Prediction:

$$
\hat{y} = \arg\max_{i} p_i.
$$

- Confidence:

$$
\text{confidence} = \max_{i} p_i.
$$

- Class label mapping: $0 \to$ Real, $1 \to$ AI Generated.

## 4. Notes

- This math document matches the code in:
  - `backend/preprocess.py`
  - `backend/model_loader.py`
  - `training/dataset.py`
  - `training/model.py`
  - `training/train.py`

- Keep this file updated if preprocessing constants or model architecture change.
