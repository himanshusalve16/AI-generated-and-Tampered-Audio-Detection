# ResNet-18 (Audio Deepfake Detection)

## AudioResNet class details (`training/model.py`)

```python
class AudioResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = resnet18(weights=None)

        # 1-channel input instead of 3
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 2 output classes instead of 1000
        num_features = base.fc.in_features
        base.fc = nn.Linear(num_features, 2)

        self.model = base

    def forward(self, x):
        return self.model(x)
```

- `base` is the full standard ResNet-18 instance.
- `conv1` is replaced to accept 1 input channel (mel spectrogram instead of 3-channel RGB).
- `fc` is replaced to output 2 scores (Real vs AI Generated).

---

## Standard ResNet-18 architecture (from torchvision)

ResNet-18 consists of:

1. `Conv1` (7x7, stride 2, pad 3) + BatchNorm + **ReLU**
2. `MaxPool` (3x3, stride 2)
3. Layer1 (2 x `BasicBlock` with 64 channels)
4. Layer2 (2 x `BasicBlock` with 128 channels, first block stride 2)
5. Layer3 (2 x `BasicBlock` with 256 channels, first block stride 2)
6. Layer4 (2 x `BasicBlock` with 512 channels, first block stride 2)
7. `AdaptiveAvgPool2d((1, 1))`
8. `Flatten` + `Linear` (num_features -> num_classes)

### BasicBlock details
- Each `BasicBlock`:
  1. Conv (3x3, padding 1) -> BatchNorm -> **ReLU**
  2. Conv (3x3, padding 1) -> BatchNorm
  3. Skip connection: add input (or projected input when channel/size changes)
  4. **ReLU**

- Shortcut projection happens when:
  - channel count changes (e.g., 64->128)
  - spatial size changes (stride 2)

---

## Loss Function

**CrossEntropyLoss** (`torch.nn.CrossEntropyLoss`) is used during training.

- Combines `LogSoftmax` + `NLLLoss` internally.
- Accepts raw logits (no softmax needed in the model output during training).
- Formula:

$$
L = -\log\left(\frac{e^{z_y}}{\sum_{j=0}^{1} e^{z_j}}\right)
$$

Where $z_y$ is the logit for the true class $y$ (0 = Real, 1 = AI Generated).

---

## Activation Functions in ResNet-18

| Location | Activation | Purpose |
|----------|-----------|---------|
| After Conv1 + BatchNorm | **ReLU** | Non-linearity after the first convolution |
| Inside every BasicBlock (after first Conv+BN) | **ReLU** | Non-linearity within residual path |
| After residual addition (skip + main) | **ReLU** | Non-linearity after skip connection merge |
| At inference time (model_loader.py) | **Softmax** | Converts 2 raw logits to probabilities |

**ReLU** is the only trainable activation in ResNet-18:

$$
\text{ReLU}(x) = \max(0, x)
$$

**Softmax** is applied only at inference time (not during training, since CrossEntropyLoss handles it):

$$
p_i = \frac{e^{z_i}}{\sum_{j=0}^{1} e^{z_j}}
$$

---

## Data flow in inference / training

1. `x` : (B,1,224,224)
2. `conv1` + BN + **ReLU**: (B,64,112,112)
3. `maxpool`: (B,64,56,56)
4. `layer1` (2 BasicBlocks with **ReLU**): (B,64,56,56)
5. `layer2` (2 BasicBlocks with **ReLU**): (B,128,28,28)
6. `layer3` (2 BasicBlocks with **ReLU**): (B,256,14,14)
7. `layer4` (2 BasicBlocks with **ReLU**): (B,512,7,7)
8. `avgpool`: (B,512,1,1) → flatten (B,512)
9. `fc`: (B,2) ← raw logits
10. **Softmax** (inference only): (B,2) ← probabilities

---

## Why only minimal code in model.py?

- The code doesn't manually define each convolutional/residual stage because torchvision provides the complete ResNet structure.
- `AudioResNet` customizes only the first and last layers while leveraging the solid ResNet implementation.
- All internal **ReLU** activations and **BatchNorm** layers are built into torchvision's `resnet18()`.
