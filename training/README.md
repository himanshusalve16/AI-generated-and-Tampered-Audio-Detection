# Training — Audio Deepfake Detection Models

Training pipeline for the ResNet-18 and LSTM models used in the audio deepfake detection system. Both models are trained on mel-spectrogram features extracted from real and AI-generated audio clips.

---

## Dataset Format

The training script expects this directory structure:

```
dataset/
└── train/
    ├── real/          # Real human speech audio files
    │   ├── clip_001.wav
    │   ├── clip_002.mp3
    │   └── ...
    └── fake/          # AI-generated or tampered audio files
        ├── clip_001.wav
        ├── clip_002.mp3
        └── ...
```

- **Label 0** = Real (from the `real/` subdirectory)
- **Label 1** = AI Generated / Fake (from the `fake/` subdirectory)
- Supported formats: Any format librosa can load (WAV, MP3, FLAC, OGG, etc.)
- **Optional:** A `test/` directory with the same `real/` and `fake/` structure for evaluation.

---

## Preprocessing

Both models share the same initial preprocessing (identical to the backend `preprocess.py`):

1. **Load** audio with librosa at the target sample rate.
2. **Resample** to 16,000 Hz, mono channel.
3. **Trim** leading and trailing silence (threshold: 20 dB below peak).
4. **Pad or crop** to exactly 3 seconds (48,000 samples).
5. **Compute** mel spectrogram with 128 mel bands, 1024 FFT window, 512 hop length.
6. **Convert** to log-decibel scale.
7. **Normalize** to zero mean and unit variance (per spectrogram).

The two models then diverge:

| Model   | What happens next                                    | Final tensor shape |
|---------|------------------------------------------------------|--------------------|
| ResNet  | Resize spectrogram to 224×224 via bilinear interpolation | `(1, 224, 224)` |
| LSTM    | Transpose to time-major order (no resize)            | `(94, 128)`       |

---

## File Structure

| File         | Purpose                                                     |
|--------------|-------------------------------------------------------------|
| `train.py`   | CLI training script with `--model resnet\|lstm` flag         |
| `model.py`   | `AudioResNet` and `AudioLSTM` class definitions             |
| `dataset.py` | `AudioDataset` PyTorch Dataset with `mode="resnet"\|"lstm"` |

---

## How Training Works

1. The `AudioDataset` scans `dataset/train/real/` and `dataset/train/fake/`.
2. Up to 1,000 samples are randomly selected (for fast iteration).
3. 800 are used for training, 200 for testing.
4. The model trains with **Adam** optimizer and **CrossEntropyLoss**.
5. At the end of training, the model is evaluated on the test split.
6. The best model (or the final epoch model) is saved.

---

## How to Run Training

### Train ResNet-18

```bash
cd training
python train.py --model resnet --epochs 25 --batch-size 16 --lr 1e-4
```

Saves weights to: `models/resnet_audio_model.pth`

### Train LSTM

```bash
cd training
python train.py --model lstm --epochs 25 --batch-size 16 --lr 1e-4
```

Saves weights to: `models/lstm_audio_model.pth`

### Full CLI options

```
python train.py --help

  --model {resnet,lstm}   Architecture to train (default: resnet)
  --dataset PATH          Path to dataset root (default: ../dataset)
  --train-subdir NAME     Training subfolder (default: train)
  --test-subdir NAME      Test subfolder (default: test)
  --models-dir PATH       Output directory for .pth files (default: ../models)
  --batch-size N          Batch size (default: 16)
  --epochs N              Number of epochs (default: 25)
  --lr FLOAT              Learning rate (default: 1e-4)
  --val-split FLOAT       Validation fraction (default: 0.2)
  --num-workers N         DataLoader workers (default: 0)
  --verbose               Enable debug logging
```

---

## Where Models Are Saved

```
models/
├── resnet_audio_model.pth    # ResNet-18 state_dict
└── lstm_audio_model.pth      # LSTM state_dict
```

These are standard PyTorch `state_dict` files loaded by `backend/model_loader.py` at server startup.

---

## Difference Between ResNet and LSTM Training

| Aspect        | ResNet-18                           | LSTM                                |
|---------------|-------------------------------------|--------------------------------------|
| Input         | 2D spectrogram image (1, 224, 224)  | 1D temporal sequence (94, 128)       |
| Architecture  | Deep CNN with residual connections  | 2-layer bidirectional LSTM + FC head |
| What it learns| Spatial texture patterns in spectrograms | Temporal dynamics in mel frames   |
| Parameters    | ~11M                                | ~400K                                |
| Strength      | Catches frequency-domain artifacts  | Catches temporal inconsistencies     |

Together, they complement each other in the ensemble for more robust detection.

---

## Tips

- **More data is better.** The default subset mode caps at 1,000 samples for fast experimentation. For production, remove the cap or increase it.
- **Use a GPU.** Training is significantly faster with CUDA. The scripts auto-detect GPU availability.
- **Consistent preprocessing.** The `dataset.py` preprocessing is intentionally identical to `backend/preprocess.py`. Do not modify one without the other.
