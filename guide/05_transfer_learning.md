# Chapter 5: Transfer Learning

| | |
|---|---|
| **Level** | Core |
| **You will learn** | How to reuse a pretrained network for your own categories, what to freeze, and how to avoid the silent mistakes |
| **Prerequisites** | Chapter 1 (training recipe), Chapter 4 (models) |
| **Ready-made code** | `cv_toolkit/models.py` (`build_classifier`, `freeze_batchnorm_stats`) |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

**In this chapter**
[5.1 The idea](#51-the-idea) · [5.2 The five decisions](#52-the-five-decisions) · [5.3 Steps](#53-steps) · [5.4 Freezing and BatchNorm](#54-freezing-and-the-batchnorm-trap) · [5.5 Preprocessing](#55-preprocessing-must-match) · [5.6 Unfreezing](#56-next-step-unfreezing-the-last-block) · [5.7 Choosing a network](#57-choosing-a-network) · [5.8 Reloading your own model](#58-reloading-your-own-saved-model) · [5.9 Common problems](#59-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 5.1 The idea

Someone has already trained a large network on over a million photos to recognise 1,000 everyday things. Along the way it learned general visual skills: edges, textures, shapes, parts of objects. **You keep all of that and teach it only the last step: what *your* categories are.**

Think of hiring an experienced inspector and giving them an afternoon of training on your product, instead of raising a new person from childhood.

**Use it when:** almost always, for an image project with fewer than roughly 100,000 labelled images.
**Train from scratch (Chapter 4) when:** images are tiny or unusual (1-channel scientific data, spectrograms), or you have a very large dataset of your own.

---

## 5.2 The five decisions

Every transfer learning project comes down to five choices.

| # | Decision | Safe default | Change it when |
|---|---|---|---|
| 1 | Which pretrained network | `resnet18` | Phone or edge: `mobilenet_v3_large`. More accuracy: `resnet50`, `convnext_tiny` |
| 2 | Where the weights come from | Download the default ImageNet weights | No internet: load a local file |
| 3 | What to freeze | Everything except the new last layer | Several thousand images and accuracy plateaus: also unfreeze the last block (5.6) |
| 4 | Learning rate | `1e-3` for the new layer; `1e-4` or lower for any unfrozen pretrained part | Loss jumps or accuracy collapses: lower it |
| 5 | Preprocessing | Exactly what the pretrained weights expect | Never skip this (5.5) |

---

## 5.3 Steps

**Step 1. Load the pretrained network.**

```python
import torch.nn as nn
from torchvision import models

weights = models.ResNet18_Weights.DEFAULT        # name the exact weights you use
model = models.resnet18(weights=weights)         # downloads and caches on first use
```

**Step 2. Replace the last layer** with one that has *your* number of classes. Each network family hides this layer under a different name:

| Family | Replace | Example |
|---|---|---|
| ResNet | `model.fc` | `model.fc = nn.Linear(model.fc.in_features, n)` |
| MobileNet, EfficientNet, ConvNeXt | `model.classifier[-1]` | `model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n)` |
| Vision Transformer (ViT) | `model.heads.head` | `model.heads.head = nn.Linear(model.heads.head.in_features, n)` |

```python
n_classes = 5
model.fc = nn.Linear(model.fc.in_features, n_classes)
```

**Step 3. Freeze everything except the new layer.**

```python
for p in model.parameters():
    p.requires_grad = False        # "do not change these numbers"
for p in model.fc.parameters():
    p.requires_grad = True         # except the new layer
```

For ResNet18 with 5 classes this leaves about 2,500 numbers to learn out of about 11 million. That is why transfer learning is fast and needs little data.

**Step 4. Give the optimizer only what should learn, then train** with the loop from Chapter 1 (or Chapter 6).

```python
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-3)
```

**What to expect:** the new layer learns within a few epochs. If validation accuracy is already high after two or three epochs and then flat, that is normal for this setup, not a bug.

> [!IMPORTANT]
> **🟣 Outdated vs current: how to load pretrained weights.**
> Older tutorials write `models.resnet18(pretrained=True)`. That argument is deprecated. Use `weights=...` as above. Prefer naming the exact set (for example `ResNet18_Weights.IMAGENET1K_V1`) in your config rather than `DEFAULT`, because `DEFAULT` can point to a newer set when the library updates, which silently changes your results.

---

## 5.4 Freezing and the BatchNorm trap

> [!TIP]
> **🟢 Extra: a detail most tutorials miss.**

Setting `requires_grad = False` stops the **weights** from learning. But ResNet, MobileNet and EfficientNet also contain **BatchNorm** layers, which keep running averages of what they see. Those averages update on every training batch while the model is in `train()` mode, *even for frozen layers*. So your "frozen" backbone slowly drifts toward your dataset's statistics.

The fix is to put BatchNorm layers back in evaluation mode after every `model.train()`:

```python
def freeze_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

for epoch in range(epochs):
    model.train()
    freeze_batchnorm(model)       # must come AFTER model.train()
    ...
```

> [!WARNING]
> **Watch out.** Calling `freeze_batchnorm` *before* `model.train()` does nothing, because `train()` switches the layers back on. Order matters.

**Which to choose?** With a small dataset and a frozen backbone, freezing BatchNorm is usually the safer default. When your images look very different from everyday photos (X-rays, microscopy), letting BatchNorm adapt can help. Try both and compare on validation data.

---

## 5.5 Preprocessing must match

A pretrained network was trained on images resized and normalised in one specific way. Give it anything else and accuracy drops quietly, with no error.

```python
preprocess = weights.transforms()   # the exact resize, crop and normalisation these weights expect
print(preprocess)
```

- Use the **same** preprocessing at prediction time as in training. A mismatch between the two is one of the most common bugs found after deployment. Save the preprocessing settings together with the model (Chapter 14).
- With pretrained weights, **do not compute your own mean and standard deviation**. The network learned with the published values.
- Augmentation (random flips, small rotations, colour jitter) goes on the *training* images only, and *before* the final normalisation step. See Chapter 3.

---

## 5.6 Next step: unfreezing the last block

> [!NOTE]
> **🔵 Advanced alternative: gradual unfreezing with two learning rates. 🟢 Extra.**
>
> **What changes:** train the new layer first. If accuracy plateaus below what you need, and you have a few thousand images, also unfreeze the *last block* of the backbone and train it with a smaller learning rate.
>
> **Why teams do it:** it usually beats "new layer only" with a few thousand images, without wrecking the pretrained skills. The smaller learning rate is the safety brake: the pretrained part only needs gentle adjustment, the brand-new layer needs bold ones.
>
> ```python
> for p in model.layer4.parameters():          # ResNet's last block; name differs per family
>     p.requires_grad = True
>
> optimizer = torch.optim.AdamW([
>     {"params": model.layer4.parameters(), "lr": 1e-4},   # gentle: pretrained
>     {"params": model.fc.parameters(),     "lr": 1e-3},   # bolder: new
> ], weight_decay=1e-2)
> ```
>
> This unfreezes roughly 8 million more numbers in ResNet18. If you unfreeze blocks that contain BatchNorm, decide on 5.4 deliberately rather than by default. Whether it helps depends on your data, so judge it on validation accuracy.

---

## 5.7 Choosing a network

| Network | Size (approx.) | Good for | Notes |
|---|---|---|---|
| `resnet18` | 13.7 M | First baseline, CPU, small data | Simple, well understood |
| `resnet50` | 25.6 M | Stronger baseline | A common industry default |
| `mobilenet_v3_large` | 6.5 M | Phones, edge devices, speed | Small and fast |
| `efficientnet_b0` | 6.3 M | Good accuracy for its size | |
| `convnext_tiny` | 28.6 M | Modern accuracy | A CNN redesigned with transformer-era ideas |
| `vit_b_16` | 86.6 M | Large datasets, strong pretrained features | Heavier; often needs lower learning rates and more care |

> [!TIP]
> **🟢 Extra:** these are common choices; pick deliberately.

**Rule of thumb:** start with `resnet18` to prove the whole pipeline works end to end, then try one larger and one smaller network. In many projects the quality of the data matters more than the choice of architecture.

---

## 5.8 Reloading your own saved model

You will often fine-tune a model, save it, and load it later. The classic crash: loading your saved file into a model that still has the original 1,000-class last layer fails with a shape mismatch.

**Rule: build the model exactly as it was saved (including the new last layer), then load.**

```python
model = models.resnet18(weights=None)                       # no download needed, weights come from your file
model.fc = nn.Linear(model.fc.in_features, n_classes)        # same shape as when you saved
state = torch.load("my_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)                                  # strict by default: raises on any mismatch
```

> [!WARNING]
> **Watch out: `strict=False`.** Passing it makes mismatched layers load silently as "ignored", so a small architecture typo leaves you with a half-random model and no error. Leave strictness on.

> [!TIP]
> **🟢 Extra:** `weights_only=True` makes loading refuse arbitrary code hidden inside a file. Save only the `state_dict` (plain numbers), plus your class names in a separate list, never the whole model object.

---

## 5.9 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Forgot to replace the last layer | Output has 1,000 values instead of your classes | Replace it (5.3, step 2) |
> | Wrong class count | Shape error, or labels out of range | Use `len(class_names)` |
> | "Frozen" model still changes | Results differ when nothing should have learned | Freeze BatchNorm (5.4) |
> | Preprocessing differs between training and prediction | Great validation score, poor real-world accuracy | Use one shared preprocessing (5.5) |
> | Unfrozen backbone with a high learning rate | Accuracy drops after unfreezing | Use `1e-4` or lower for pretrained parts |
> | Silent loading mismatch | Model gives random-looking answers after loading | Strict loading (5.8) |
> | Class order differs at serving time | Right shapes, wrong labels | Save the class names with the weights |
> | Forgot `model.eval()` when scoring | Scores change from run to run | `model.eval()` and `torch.no_grad()` |

---

## Key takeaways

1. Load pretrained weights, **replace the last layer**, freeze the rest, train only the new layer.
2. Frozen layers need BatchNorm frozen too, or they keep drifting.
3. Preprocessing must match what the weights expect, in training **and** in prediction.
4. Start with `resnet18`. Unfreeze the last block with a smaller learning rate only if you need more accuracy.
5. Load saved models strictly, with the same last layer they were saved with.

**Previous:** [Chapter 4: Models](04_models.md) &nbsp;|&nbsp; **Next:** [Chapter 6: Training the model](06_training_loop.md)
