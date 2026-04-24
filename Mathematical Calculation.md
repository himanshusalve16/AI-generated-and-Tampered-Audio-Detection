# Mathematical Calculation

This document contains all the detailed mathematical calculations for the project, including audio preprocessing, ResNet-18 operations, LSTM operations, pooling, classification, loss function, and ensemble logic.


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
- STFT parameters: $N_{\text{FFT}} = 1024$, $\text{HOP\_LENGTH} = 512$, window = Hann.
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

### 1.7 Spatial Resizing (ResNet path)
- Input shape: $(128, T)$.
- Target shape: $(224, 224)$ via bilinear interpolation.
- Output shape after preprocessing: $(1, 224, 224)$.

### 1.8 Temporal Sequence (LSTM path)
- Input shape: $(128, T)$ where $T = 1 + \lfloor 48000 / 512 \rfloor = 94$.
- Transpose to time-major: $(T, 128) = (94, 128)$.
- No resizing — preserves original temporal resolution.

## 2. ResNet-18 Model Mathematics

### 2.1 Input/Output
- Input tensor: $x \in \mathbb{R}^{1 \times 224 \times 224}$.
- Output logits: $\mathrm{logits} \in \mathbb{R}^{2}$ (Real vs AI).

### 2.2 Feature Extraction Layers

#### 2.2.1 Conv1 + BatchNorm + ReLU + MaxPool
- Conv1 weights: $W_1 \in \mathbb{R}^{64 \times 1 \times 7 \times 7}$, stride $=2$, padding $=3$.
- Output dims:

$$
\text{out} = \left\lfloor\frac{224 + 2\cdot3 - 7}{2} \right\rfloor + 1 = 112.
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
\mathrm{logits}_{\text{resnet}} = W_{\text{fc}} g + b_{\text{fc}}.
$$

## 3. LSTM Model Mathematics

### 3.1 Input/Output
- Input tensor: $x \in \mathbb{R}^{T \times D}$ where $T = 94$ (time steps), $D = 128$ (mel bands).
- Output logits: $\mathrm{logits} \in \mathbb{R}^{2}$ (Real vs AI).

### 3.2 LSTM Cell Equations

At each time step $t$, the LSTM cell computes:

**Forget gate** (what to discard from cell state):

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
$$

**Input gate** (what new info to store):

$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
$$

**Candidate cell state** (new candidate values):

$$
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
$$

**Cell state update** (combine old and new):

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

**Output gate** (what to output):

$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
$$

**Hidden state** (final output at time $t$):

$$
h_t = o_t \odot \tanh(c_t)
$$

Where:
- $\sigma$ is the **sigmoid** activation: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- $\tanh$ is the **hyperbolic tangent**: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
- $\odot$ is element-wise (Hadamard) product

### 3.3 Bidirectional LSTM

Two separate LSTM passes run on the same input:
- **Forward LSTM**: processes $x_1, x_2, \dots, x_T$ → produces $\overrightarrow{h}_1, \dots, \overrightarrow{h}_T$
- **Backward LSTM**: processes $x_T, x_{T-1}, \dots, x_1$ → produces $\overleftarrow{h}_1, \dots, \overleftarrow{h}_T$

At each time step, outputs are concatenated:

$$
h_t = [\overrightarrow{h}_t ; \overleftarrow{h}_t] \in \mathbb{R}^{2H}
$$

Where $H = 128$ (hidden dim), so $h_t \in \mathbb{R}^{256}$.

### 3.4 Multi-Layer Stacking

With $L = 2$ layers:
- Layer 1 input: $x_t \in \mathbb{R}^{128}$ (mel features)
- Layer 1 output: $h_t^{(1)} \in \mathbb{R}^{256}$ (bidirectional)
- Layer 2 input: $h_t^{(1)} \in \mathbb{R}^{256}$
- Layer 2 output: $h_t^{(2)} \in \mathbb{R}^{256}$
- Dropout ($p = 0.3$) applied between layers.

### 3.5 Classifier Head

Takes the last time step output $h_T^{(2)}$ and passes through:

$$
\begin{aligned}
  a_1 &= \text{Dropout}(h_T^{(2)}), & & a_1 \in \mathbb{R}^{256} \\
  a_2 &= \text{ReLU}(W_1 a_1 + b_1), & & a_2 \in \mathbb{R}^{64} \\
  a_3 &= \text{Dropout}(a_2), & & a_3 \in \mathbb{R}^{64} \\
  \mathrm{logits}_{\text{lstm}} &= W_2 a_3 + b_2, & & \mathrm{logits} \in \mathbb{R}^{2}
\end{aligned}
$$

Where $W_1 \in \mathbb{R}^{64 \times 256}$, $W_2 \in \mathbb{R}^{2 \times 64}$.

## 4. Loss Function

### 4.1 CrossEntropyLoss

Both models are trained with **CrossEntropyLoss**. For a single sample with true class $y$:

$$
L = -\log\left(\frac{e^{z_y}}{\sum_{j=0}^{1} e^{z_j}}\right) = -z_y + \log\left(\sum_{j=0}^{1} e^{z_j}\right)
$$

Where $z_0, z_1$ are the raw logits from the model.

For a mini-batch of $N$ samples:

$$
L_{\text{batch}} = \frac{1}{N} \sum_{n=1}^{N} L_n.
$$

### 4.2 Why CrossEntropyLoss?

- Standard choice for multi-class classification with mutually exclusive classes.
- Applies softmax internally, so the model outputs raw logits (no explicit softmax layer needed during training).
- Numerically stable (uses log-sum-exp trick).

## 5. Activation Functions

### 5.1 ReLU (Rectified Linear Unit)

Used in ResNet BasicBlocks and LSTM classifier head:

$$
\text{ReLU}(x) = \max(0, x)
$$

- Gradient: $\frac{\partial}{\partial x}\text{ReLU}(x) = \begin{cases} 1, & x > 0 \\ 0, & x \le 0 \end{cases}$
- Prevents vanishing gradients for positive activations.

### 5.2 Sigmoid

Used internally in LSTM gates (input, forget, output):

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

- Output range: $(0, 1)$ — acts as a "soft switch" controlling information flow.

### 5.3 Tanh (Hyperbolic Tangent)

Used internally in LSTM for cell state computation:

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

- Output range: $(-1, 1)$ — squashes values symmetrically around zero.

### 5.4 Softmax

Used at inference time to convert logits to probabilities:

$$
p_i = \frac{e^{z_i}}{\sum_{j=0}^{1} e^{z_j}}, \qquad i \in \{0, 1\}.
$$

- Not used during training (CrossEntropyLoss handles it internally).

## 6. Pooling and Classification

### 6.1 Pooling Operations (ResNet)
- MaxPool1: Feature reduction via local maxima.
- GlobalAvgPool: channel-wise mean to 512 dimensions.

### 6.2 Prediction

- Predicted class:

$$
\hat{y} = \arg\max_{i} p_i.
$$

- Confidence:

$$
\text{confidence} = \max_{i} p_i.
$$

- Class label mapping: $0 \to$ Real, $1 \to$ AI Generated.

## 7. Ensemble Mathematics

### 7.1 Weighted Probability Averaging

Given individual model probabilities:
- ResNet: $p_{\text{resnet}} = [p_0^R, p_1^R]$
- LSTM: $p_{\text{lstm}} = [p_0^L, p_1^L]$

Ensemble probability:

$$
p_{\text{ensemble}} = w_R \cdot p_{\text{resnet}} + w_L \cdot p_{\text{lstm}}
$$

Default weights: $w_R = 0.5$, $w_L = 0.5$ (configurable).

### 7.2 Ensemble Prediction

$$
\hat{y}_{\text{ensemble}} = \arg\max_{i} \; p_{\text{ensemble}, i}
$$

$$
\text{confidence}_{\text{ensemble}} = \max_{i} \; p_{\text{ensemble}, i}
$$

## 8. Notes

- This math document matches the code in:
  - `backend/preprocess.py`
  - `backend/model_loader.py`
  - `backend/config.py`
  - `training/dataset.py`
  - `training/model.py`
  - `training/train.py`

- Keep this file updated if preprocessing constants, model architecture, or loss function change.
