# Chapter 4: Designing Models

| | |
|---|---|
| **Level** | Core |
| **You will learn** | How an image network is built, how to choose its settings, how to check its shapes and size, and what makes an efficient design |
| **Prerequisites** | Chapter 1 |
| **Ready-made code** | `cv_toolkit/models.py` (`build_mlp`, `ConvBlock`, `ConvNet`, `InvertedResidualBlock`, `MobileNetLikeClassifier`) |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

**In this chapter**
[4.1 The idea](#41-the-idea) · [4.2 Layers](#42-the-layers-and-what-each-does) · [4.3 The CNN block](#43-the-cnn-block) · [4.4 Building a CNN](#44-building-a-cnn-step-by-step) · [4.5 Choosing settings](#45-choosing-the-settings) · [4.6 Checking the model](#46-checking-the-model) · [4.7 Efficient design](#47-efficient-design-depthwise-and-skip-connections) · [4.8 Scratch or pretrained](#48-train-from-scratch-or-use-a-pretrained-model) · [4.9 Common problems](#49-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 4.1 The idea

An image network has two halves:

1. **Feature extractor:** a stack of convolution blocks that turns pixels into a smaller, richer picture ("here is an edge, here is a texture, here is something wheel-shaped").
2. **Decision head:** a few ordinary layers that read those features and output one score per class.

Early blocks see small details, later blocks see larger structures. Along the way the picture gets **smaller and deeper**: fewer pixels, more channels.

---

## 4.2 The layers and what each does

| Layer | In plain words | Key setting |
|---|---|---|
| `nn.Conv2d` | Slides small pattern detectors (filters) over the image | `out_channels` (how many detectors), `kernel_size` (how big each looks), `stride`, `padding` |
| `nn.BatchNorm2d` | Re-centres the numbers between layers so training stays stable and can use a bigger learning rate | channels (must match the conv before it) |
| `nn.ReLU` | Lets the network learn curves, not just straight lines | none |
| `nn.MaxPool2d(2)` | Keeps the strongest signal from each 2x2 area: halves the picture size | window size |
| `nn.Flatten` | Unrolls the final feature grid into one long list | none |
| `nn.Linear` | Weighs all features to produce scores | in and out sizes |
| `nn.Dropout(p)` | Randomly switches off a share `p` of units during training so the network cannot lean on any single one | `p`, 0.2 to 0.5 |
| `nn.AdaptiveAvgPool2d(1)` | Averages each channel's grid to one number | none |

---

## 4.3 The CNN block

The standard building brick, repeated:

```python
import torch
import torch.nn as nn

def conv_block(in_ch, out_ch, kernel=3, pool=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel, padding=(kernel - 1) // 2),   # "same" padding: keeps the size
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))                                  # halves height and width
    return nn.Sequential(*layers)
```

**Order:** convolution, then normalisation, then activation, then pooling. Padding of `(kernel-1)//2` keeps height and width unchanged for odd kernels, so the only thing that shrinks the picture is pooling. That makes the shape arithmetic easy.

---

## 4.4 Building a CNN, step by step

**Step 1. Fix the input.** Decide the image size and channels (for example 3 x 64 x 64). The model must be built for the size you feed it.

**Step 2. Stack blocks, doubling the channels as the picture shrinks:**

```python
features = nn.Sequential(
    conv_block(3, 32),     # 64x64 -> 32x32, 32 channels
    conv_block(32, 64),    # -> 16x16, 64 channels
    conv_block(64, 128),   # -> 8x8, 128 channels
)
```

**Step 3. Work out the flattened size** (the input size of the first `Linear` layer). Do not do it by hand: let the code measure it.

```python
with torch.no_grad():
    flat = features.eval()(torch.zeros(1, 3, 64, 64)).numel()   # here 128 * 8 * 8 = 8192
```

**Step 4. Add the head.**

```python
model = nn.Sequential(
    features,
    nn.Flatten(),
    nn.Linear(flat, 512), nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes),        # one output per class, raw scores, no softmax
)
```

**Step 5. Check the output shape and the size** (4.6) before any training.

> [!WARNING]
> **Watch out: the final layer outputs raw scores (logits).** Do not add `Softmax` when training with `CrossEntropyLoss`: it already does that internally, and doing it twice degrades learning.

> [!IMPORTANT]
> **🟣 Outdated vs current: computing the flatten size lazily inside `forward`.** Some tutorials create the first `Linear` layer the first time data passes through. That means the model is *incomplete* until you run a dummy batch, so an optimizer built too early misses those weights, and loading a saved file fails until the dummy pass happens. Compute the size once when you build the model, as above.

---

## 4.5 Choosing the settings

These are the choices you actually make. Treat the "typical start" as a starting point to improve on, not a rule.

| Setting | Typical start | Why it matters | Signs to change it |
|---|---|---|---|
| **Number of conv blocks** | 3 (small images, 32 to 64 px) to 4 or 5 (224 px) | Each pooling halves the picture; more blocks see larger structures | Underfits (training accuracy low): add capacity. Image becomes 1x1 or zero: too many blocks for the size |
| **Channels per block** | 32, 64, 128 (double each block) | More channels, more patterns detected, more parameters | Underfit: widen. Overfit or too big: narrow |
| **Kernel size** | 3 | Two stacked 3x3 layers see as much as one 5x5 with fewer parameters | Rarely worth changing. 5 or 7 mostly for the first layer on large images |
| **Hidden size of the head** | 128 to 512 | Capacity of the decision part | Head dominates the parameter count: shrink it or use average pooling (below) |
| **Dropout** | 0.3 to 0.5 in the head | Fights overfitting | Training accuracy far above validation: raise it. Both low: lower it |
| **BatchNorm** | Yes, after each conv | Faster, steadier training | Almost always keep |
| **Input size** | 32 to 64 for small custom CNNs; 224 for pretrained | Bigger shows more detail but costs speed and memory (cost grows with the square of the side) | Fine details matter (small defects): larger. Slow: smaller |

**Where do the parameters go?** In a small CNN most parameters sit in the *first Linear layer after Flatten*, because it connects every flattened feature to every hidden unit. In the example above that layer alone has `8192 x 512` (about 4 million) weights, far more than all the convolutions together. Two ways to shrink it: pool more, or replace Flatten with global average pooling:

```python
head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes))
```

This turns each of the 128 channels into one number, so the head has `128 x num_classes` weights. It is also independent of the input size, so the same model accepts different image sizes. Modern networks use this.

---

## 4.6 Checking the model

Do these three checks every time you write or change a network:

```python
x = torch.randn(4, 3, 64, 64)               # a fake batch of 4 images
print(model(x).shape)                        # expect torch.Size([4, num_classes])

n = sum(p.numel() for p in model.parameters())
print(f"{n:,} parameters, about {n * 4 / 1e6:.1f} MB as 32-bit floats")
```

| Check | What a wrong answer tells you |
|---|---|
| Output shape is `[batch, num_classes]` | Wrong flatten size or wrong last layer |
| Parameter count is plausible | A surprise (far more than expected) usually means a huge `Linear` after Flatten |
| It trains on one tiny batch to near-zero loss (Chapter 7) | If it cannot even memorise 16 images, there is a bug in the model or the loop |

> [!TIP]
> **🟢 Extra: the parameter budget.** Know your limit before you design: a phone app or an edge device may allow a few million parameters, a server far more. Write the budget down first, then design to it.

---

## 4.7 Efficient design: depthwise and skip connections

Two ideas make modern networks small and trainable.

### Depthwise separable convolution

A normal 3x3 convolution mixes *space and channels at once*. Split the job in two:

1. **Depthwise 3x3:** each channel is filtered on its own (`groups=channels`).
2. **Pointwise 1x1:** a 1x1 convolution mixes channels.

For `C` channels in and out, the cost drops from about `9 x C x C` to `9 x C + C x C` weights: about 8 times fewer at 64 channels, approaching 9 times as channels grow (for a 3x3 kernel).

```python
depthwise = nn.Conv2d(C, C, 3, padding=1, groups=C)    # groups=C is what makes it depthwise
pointwise = nn.Conv2d(C, C, 1)
```

> [!WARNING]
> **Watch out: a layer named "depthwise" is only depthwise if `groups` equals the channel count.** Without `groups`, it is an ordinary convolution with many times more parameters, and your "efficient" network is not. Always compare parameter counts against what the design promises.

### Inverted residual block (MobileNet style)

```
input ──► 1x1 expand ──► 3x3 depthwise ──► 1x1 squeeze ──► (+) ──► output
  │                                                         ▲
  └─────────────────── skip connection ─────────────────────┘
```

- **Expand** the channels (for example 6x) so the cheap depthwise filter has room to work.
- **Depthwise 3x3** looks at each channel's pattern.
- **Squeeze** back down to a small width.
- **Skip connection** adds the input to the output, so each block only has to learn a *correction*. This keeps very deep networks trainable.

**The skip needs matching shapes.** When the number of channels or the stride changes, pass the input through a 1x1 convolution first so the two can be added.

**Global average pooling plus one `Linear`** then forms the head, as in 3.5.

> [!NOTE]
> **🔵 Advanced alternative: don't hand-build, reuse.**
>
> **What changes:** take a proven design from a library instead of writing blocks yourself, for example `torchvision.models.mobilenet_v3_large` or `timm.create_model("efficientnet_b0")`.
>
> **Why teams do it:** published networks have been tuned on huge compute budgets, and come with pretrained weights (Chapter 5). Hand-built blocks are for learning how they work, or for unusual inputs where no ready-made network fits.

---

## 4.8 Train from scratch or use a pretrained model?

| Situation | Choose |
|---|---|
| Natural photos, up to roughly 100,000 images | **Pretrained** (Chapter 5). It is almost always better |
| Unusual input (1-channel scientific data, spectrograms, tiny 28x28 images) | From scratch, or pretrained with the first layer adapted |
| Huge dataset of your own and a very different domain | From scratch can work |
| You are learning how networks work | From scratch, on a small dataset |

---

## 4.9 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Wrong flatten size | `mat1 and mat2 shapes cannot be multiplied` | Measure with a dummy batch (4.4, step 3) |
> | Too many pooling layers for the image size | Feature grid reaches 0 or 1 pixel; error or zero size | Fewer blocks, or a larger input |
> | Softmax before `CrossEntropyLoss` | Loss stalls, learning is weak | Output raw scores |
> | Model far larger than expected | Parameter count is huge | Look at the `Linear` after Flatten; use global average pooling |
> | "Depthwise" layer with no `groups` | Parameter count far above the design's promise | Set `groups=channels` |
> | Skip connection shape error | Cannot add tensors of different shape | 1x1 projection on the skip when channels or stride change |
> | Model cannot even fit 16 images | Loss will not approach zero on a tiny batch | Bug hunt before tuning (Chapter 7) |
> | Built for 64 px, fed 224 px | Shape error or silently wrong behaviour | One image-size setting, shared by data and model |

---

## Key takeaways

1. A CNN is **blocks of Conv, BatchNorm, ReLU, Pool** that shrink the picture and deepen the channels, then a small **head** that outputs one raw score per class.
2. Let code **measure** the flatten size, and check the output shape and parameter count every time.
3. Most parameters in a small CNN sit in the first `Linear` after Flatten. **Global average pooling** shrinks it and frees you from a fixed image size.
4. Efficiency comes from **depthwise convolutions** (needs `groups`) and **skip connections** (needs matching shapes).
5. For real photo projects, start from a **pretrained** network instead of designing one.

**Previous:** [Chapter 3: Data pipeline](03_data_pipeline.md) &nbsp;|&nbsp; **Next:** [Chapter 5: Transfer learning](05_transfer_learning.md)
