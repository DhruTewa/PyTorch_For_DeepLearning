# Chapter 3: Data, Transforms, Loaders and Imbalance

| | |
|---|---|
| **Level** | Core |
| **You will learn** | How to transform and augment images, build datasets and loaders, and handle imbalanced classes |
| **Prerequisites** | Chapters 1, 2 |
| **Ready-made code** | `cv_toolkit/data.py` |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

**In this chapter**
[3.1 The idea](#31-the-idea) · [3.2 Transforms](#32-transforms-and-augmentation) · [3.3 Datasets and loaders](#33-datasets-and-loaders) · [3.4 Imbalance](#34-imbalanced-classes) · [3.5 Common problems](#35-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 3.1 The idea

[Chapter 2](02_data_inspection_splitting.md) left you with checked data and an honest split. Three jobs remain before the model sees a batch:

1. **Transform** each image: resize and normalise it, and (for training only) vary it.
2. **Load**: wrap the images in a `Dataset` and a `DataLoader` that serves shuffled batches.
3. **Balance**: decide what to do if one class is much rarer than the others.

The rule from Chapter 2 still applies: anything computed from data (mean and std, class weights) uses the **training split only**.

---

## 3.2 Transforms and augmentation

Two recipes, always:

| | **Training recipe** | **Evaluation recipe** (validation, test, prediction) |
|---|---|---|
| Resize | yes | yes (same size) |
| Random changes (flip, rotate, colour) | **yes** | **no** |
| Convert to tensor | yes | yes |
| Normalise (mean, std) | yes | yes (same numbers) |

Random changes show the model many variations of each image, which fights memorising. At evaluation you want a fixed, repeatable score, so nothing is random.

```python
from torchvision import transforms

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # ImageNet values, for pretrained models

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

### Choosing augmentations

**The rule: an augmentation must keep the label true.**

| Augmentation | Fine for | Wrong for |
|---|---|---|
| Horizontal flip | Most natural objects | Text, digits, anything where left and right matter |
| Vertical flip | Leaves, aerial or microscope images | Faces, street scenes, most X-rays, digits (6 and 9) |
| Small rotation (5 to 15 degrees) | Most things | Cases where orientation is the label |
| Colour jitter (brightness, contrast) | Photos in varied lighting | Tasks where colour *is* the answer (ripeness, stain type) |
| Random resized crop | Object classification | Tasks where the object may get cut out (small defects) |
| Small shifts | Most things | Data where position is meaningful |

### Which mean and std?

| Situation | Use |
|---|---|
| Pretrained network (Chapter 5) | The values that network was trained with (ImageNet numbers above) |
| Training from scratch | Compute from your **training** images only |

```python
# from-scratch case: mean and std per colour channel, from training data only
total, total_sq, count = 0, 0, 0
for x, _ in DataLoader(train_ds_without_normalize, batch_size=64):
    total += x.sum(dim=(0, 2, 3)); total_sq += (x ** 2).sum(dim=(0, 2, 3)); count += x.numel() / 3
mean = total / count
std = (total_sq / count - mean ** 2).sqrt()
```

> [!NOTE]
> **🔵 Advanced alternative: `transforms.v2` and automatic augmentation.**
>
> **What changes:** use `torchvision.transforms.v2` (`v2.ToImage()` then `v2.ToDtype(torch.float32, scale=True)` instead of `ToTensor()`), and try ready-made policies such as `TrivialAugmentWide` instead of hand-picking flips and rotations.
>
> **Why teams do it:** the v2 transforms are the current recommendation, are faster, and also work on bounding boxes and masks if you later move to detection or segmentation. Automatic policies often match hand-tuned augmentation with no tuning.

---

## 3.3 Datasets and loaders

A **Dataset** answers "give me item number *i*": an image and its label. A **DataLoader** groups items into batches, shuffles them, and can load in parallel.

**Folder layout for `ImageFolder`** (the most common case): the folder name is the label.

```
data/train/cat/001.jpg
data/train/dog/002.jpg
data/val/cat/...
```

```python
from torchvision import datasets
from torch.utils.data import DataLoader

train_ds = datasets.ImageFolder("data/train", transform=train_tf)
val_ds = datasets.ImageFolder("data/val", transform=eval_tf)
print(train_ds.classes)         # class names; the position in this list is the label number

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)
```

**Labels in a spreadsheet instead of folders?** Write a small Dataset that reads the CSV (filename and label columns), opens the image in `__getitem__`, applies the transform and returns `(image, label)`. Three methods are enough: `__init__`, `__len__`, `__getitem__`.

### Loader settings

| Setting | What it does | Typical choice |
|---|---|---|
| `batch_size` | Images per step | 16 to 64. Limited by memory; larger is not always better. Lower it if you get out-of-memory errors |
| `shuffle` | Random order each epoch | `True` for training, `False` for validation and test |
| `num_workers` | Parallel loading processes | 0 in notebooks (safest); 2 to 8 in scripts |
| `pin_memory` | Faster copy to GPU | `True` when using a GPU |
| `persistent_workers` | Keep workers alive between epochs | `True` when `num_workers > 0` |

> [!WARNING]
> **Watch out.** Shuffling validation data is harmless but pointless, and *not* shuffling training data hurts learning. If training is slow and the GPU sits idle, loading is the bottleneck: raise `num_workers`.

**Check one batch before training** (a habit that saves hours):

```python
x, y = next(iter(train_loader))
print(x.shape, x.dtype, x.min().item(), x.max().item())   # e.g. [32, 3, 224, 224], float32, about -2 to 2.6
print(y[:10], train_ds.classes)
```

If values are 0 to 255, normalisation is missing. If the shape is wrong, resizing is wrong. Look at the pictures too: a plot of 8 images catches flipped labels and broken transforms immediately.

---

## 3.4 Imbalanced classes

If 90% of images are "normal", a model that always says "normal" is 90% accurate and useless. Two things go wrong: the **loss** is dominated by the common class, and **accuracy** hides it.

| Remedy | How | When |
|---|---|---|
| **Class weights** | Rare classes count more in the loss | First thing to try. Simple |
| **Weighted sampler** | Show rare-class images more often | Strong imbalance |
| **Collect more rare data** | | Always the best answer if possible |
| **Better metrics** | Look at per-class recall, precision, F1, not accuracy | Always (Chapter 10) |

```python
import torch, torch.nn as nn

counts = torch.bincount(torch.tensor(train_ds.targets))          # images per class
weights = counts.sum() / (len(counts) * counts.float())           # rare class -> bigger number
loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
```

> [!WARNING]
> **Watch out.** Compute weights from **training** data only. Move them to the same device as the model, or you get a device-mismatch error. Weights change the *loss*, not the metrics: always judge with per-class metrics.

> [!NOTE]
> **🔵 Advanced alternative: fix imbalance in the sampler, not the loss.**
>
> ```python
> from torch.utils.data import WeightedRandomSampler
> sample_w = (1.0 / counts.float())[train_ds.targets]
> sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
> train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)   # do not also pass shuffle=True
> ```
>
> **Why teams do it:** each batch then contains a balanced mix, which often trains more smoothly than reweighting the loss. Use one remedy at a time; combining both over-corrects.

---

---

## 3.5 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Augmentation on validation data | Validation score jumps around between runs | Separate train and evaluation transforms |
> | Wrong mean/std or missing normalisation | Batch values are 0 to 255, or pretrained model underperforms | Check one batch; match the pretrained recipe |
> | Augmentation breaks the label | A "6" flipped into a "9" | Choose label-preserving augmentations only |
> | Label order differs between runs | Predictions are right shape, wrong names | Save `class_names` with the model |
> | Extreme class imbalance | High accuracy, near-zero recall on the rare class | Class weights and per-class metrics |

---

## Key takeaways

1. Two transform recipes: **random for training, fixed for evaluation**. Augmentations must keep the label true.
2. Pretrained network: use its mean and std. From scratch: compute them on training data only.
3. Check one batch (shape, value range, a plot) before every first training run.
4. Imbalance: weights or a sampler, plus per-class metrics. Never trust accuracy alone.

**Previous:** [Chapter 2: Data, inspecting and splitting](02_data_inspection_splitting.md) &nbsp;|&nbsp; **Next:** [Chapter 4: Models](04_models.md)
