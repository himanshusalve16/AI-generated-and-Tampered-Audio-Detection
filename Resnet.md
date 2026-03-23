# ResNet-18 (Audio Deepfake Detection)

This file explains everything related to the ResNet model used in this project:
- architecture
- how data flows
- how the adaptation is done for 1-channel audio
- how classification outputs are produced

---

## 1. Overview

The model is `ResNet-18` from torchvision, adapted for audio spectrogram input.

### 1.1 Why ResNet?
- Residual blocks help training deeper networks by allowing identity shortcuts.
- ResNet-18 is light enough for practical experiments and strong for audio pattern detection.

### 1.2 Input type
- Input is a mel-spectrogram converted to a tensor of shape:
  - `(batch_size, 1, 224, 224)`
- ONE channel (grayscale), not RGB.

### 1.3 Output type
- Output is 2 logits for binary classification:
  - class 0 = Real
  - class 1 = AI Generated

---

## 2. Code location
- `training/model.py` defines the `AudioResNet` class.
- `backend/model_loader.py` also builds the same architecture for inference.

---

## 3. AudioResNet class details (`training/model.py`)

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
- `conv1` is replaced to accept 1 input channel.
- `fc` is replaced to output 2 scores.

---

## 4. Standard ResNet-18 architecture (from torchvision)

ResNet-18 consists of:

1. `Conv1` (7x7, stride 2, pad 3) + BatchNorm + ReLU
2. `MaxPool` (3x3, stride 2)
3. Layer1 (2 x `BasicBlock` with 64 channels)
4. Layer2 (2 x `BasicBlock` with 128 channels, first block stride 2)
5. Layer3 (2 x `BasicBlock` with 256 channels, first block stride 2)
6. Layer4 (2 x `BasicBlock` with 512 channels, first block stride 2)
7. `AdaptiveAvgPool2d((1, 1))`
8. `Flatten` + `Linear` (num_features -> num_classes)

### 4.1 BasicBlock details
- Each `BasicBlock`:
  1. Conv (3x3, padding 1) -> BatchNorm -> ReLU
  2. Conv (3x3, padding 1) -> BatchNorm
  3. Skip connection: add input (or projected input when channel/size changes)
  4. ReLU

- Shortcut projection happens when:
  - channel count changes (e.g., 64->128)
  - spatial size changes (stride 2)

---

## 5. Data flow in inference / training

1. `x` : (B,1,224,224)
2. `conv1`: (B,64,112,112)
3. `maxpool`: (B,64,56,56)
4. `layer1`: (B,64,56,56)
5. `layer2`: (B,128,28,28)
6. `layer3`: (B,256,14,14)
7. `layer4`: (B,512,7,7)
8. `avgpool`: (B,512,1,1) → flatten (B,512)
9. `fc`: (B,2)

---

## 6. OT: Why only minimal code in model.py?

- The code doesn't manually define each convolutional/residual stage because torchvision provides the complete ResNet structure.
- `AudioResNet` customizes only the first and last layers while leveraging the solid ResNet implementation.

---

## 7. Training / inference tie-in

- `training/train.py` instantiates `AudioResNet`, trains with cross entropy.
- `backend/model_loader.py` builds equivalent ResNet and loads the saved weights.
- Because architecture is identical, weights are compatible.

---

## 8. Activation and class probability

After `fc`, use softmax to build probabilities:

$$
p_i = \frac{e^{logit_i}}{e^{logit_0} + e^{logit_1}}\quad (i=0,1)$$

- predicted label = `argmax p_i`
- confidence = `max p_i`

---

## 9. Customization notes

To adjust sensitivity or complexity:
- Change ResNet depth (e.g., ResNet-34, ResNet-50) in both train and backend.
- Add dropout after global pooling (optional, if overfitting).
- Keep `conv1` and `fc` architecture edits synchronized across training + inference files.
