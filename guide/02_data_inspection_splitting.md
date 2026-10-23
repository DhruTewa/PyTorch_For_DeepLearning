# Chapter 2: Data, Inspecting and Splitting

| | |
|---|---|
| **Level** | Core |
| **You will learn** | How to look at image data, check its quality, and split it honestly into training, validation and test sets |
| **Prerequisites** | Chapter 1 |
| **Ready-made code** | `cv_toolkit/data.py` |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

**In this chapter**
[2.1 The idea](#21-the-idea) · [2.2 The pipeline](#22-the-pipeline) · [2.3 Look first](#23-look-at-the-data-first) · [2.4 Data quality](#24-data-quality-checks) · [2.5 Splitting](#25-splitting-honestly) · [2.6 Common problems](#26-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 2.1 The idea

A model can only learn what the data shows it, and can only be *judged* fairly if the data it is tested on was kept apart from the start. Most disappointing projects trace back to the data, not the network: mislabelled images, near-duplicates in both training and test sets, or a split that lets the answer leak.

Spend your first hours here. It is the cheapest place to find problems.

---

## 2.2 The pipeline

| Step | Question | Section |
|---|---|---|
| 1. Look | What is in here? How many per class? | 2.3 |
| 2. Check quality | Anything broken, duplicated or mislabelled? | 2.4 |
| 3. Split | Which images are for learning, tuning, final testing? | 2.5 |
| 4. Transform | How is each image resized, normalised, varied? | 3.2 |
| 5. Load | How are images fed to the model in batches? | 3.3 |
| 6. Balance | What if one class is rare? | 3.4 |

Steps 1 to 3 are in this chapter. Steps 4 to 6 continue in [Chapter 3: Data pipeline](03_data_pipeline.md).

**Order matters:** split *before* anything that learns from the data (statistics, class weights, feature selection). That single rule prevents the most common source of inflated scores.

---

## 2.3 Look at the data first

Before any code that trains, answer these in 10 minutes:

```python
from pathlib import Path
from collections import Counter

root = Path("data/raw")
counts = Counter(p.parent.name for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
print(counts)                                           # images per class
print(Counter(p.suffix.lower() for p in root.rglob("*") if p.is_file()))   # file types
```

**What to look for:**

| Look at | Why it matters |
|---|---|
| Images per class | A 95/5 split means "always say the common class" scores 95% |
| File types and sizes | Mixed formats or tiny files often mean broken data |
| Image dimensions | Wildly different sizes, or very small ones, change how you resize |
| 20 random images per class, by eye | Reveals wrong labels, watermarks, borders, or a background that gives the class away |

That last row is the most valuable and the most skipped. If every "rotten" photo has a dark background and every "fresh" one a white one, the model will learn the background.

---

## 2.4 Data quality checks

> [!TIP]
> **🟢 Extra: this whole section.** These checks are cheap and catch problems that no model tuning can fix.

**Corrupt files.** One unreadable image crashes training hours in.

```python
from PIL import Image

bad = []
for p in root.rglob("*"):
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:
            bad.append(p)
print(len(bad), "unreadable files")
```

**Duplicates across splits (leakage).** If the same image sits in training *and* validation, your score is meaningless: the model is being tested on what it memorised.

```python
import hashlib

def digest(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

train_hashes = {digest(p) for p in Path("data/train").rglob("*.jpg")}
overlap = [p for p in Path("data/val").rglob("*.jpg") if digest(p) in train_hashes]
print(len(overlap), "validation images also appear in training")
```

This finds **exact** duplicates only. Near-duplicates (the same photo resized, or consecutive video frames) need image-similarity tools (Chapter 11 explains the idea).

**Group leakage.** Several images of the *same patient, product, or scene* must all go to the same split. If they are scattered, the model recognises the individual rather than the condition. Split by group (2.5).

**Label noise.** Real datasets often have 1 to 5% wrong labels. After a first model, look at the images it is *most confident are wrong*: they are often labelling errors, not model errors (Chapter 10).

**Colour mode.** Pretrained networks expect 3 channels. A grayscale file loaded as 1 channel will fail. Convert on load: `Image.open(p).convert("RGB")`.

---

## 2.5 Splitting honestly

| Split | Purpose | Typical share |
|---|---|---|
| **Train** | The model learns from these | 70 to 80% |
| **Validation** | You tune settings and decide when to stop | 10 to 15% |
| **Test** | One final score, used **once** at the end | 10 to 15% |

Why three, not two: every time you look at the validation score and change something, you leak a little of it into your decisions. The test set stays untouched so the final number is honest.

**Make the split repeatable:**

```python
import torch
from torch.utils.data import random_split

n = len(dataset)
n_val, n_test = int(0.15 * n), int(0.15 * n)
train_set, val_set, test_set = random_split(
    dataset, [n - n_val - n_test, n_val, n_test],
    generator=torch.Generator().manual_seed(42),
)
```

> [!WARNING]
> **Watch out: `random_split` with augmentation.** The three subsets share the *same* underlying dataset, so they also share its transform. If that transform contains random augmentation, validation images get augmented too, and your validation score becomes noisy. Wrap each subset so it has its own transform:

```python
from torch.utils.data import Dataset

class WithTransform(Dataset):
    """Gives a subset its own transform. The base dataset must return PIL images (no transform of its own)."""
    def __init__(self, subset, transform):
        self.subset, self.transform = subset, transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, i):
        img, label = self.subset[i]
        return self.transform(img), label

train_ds = WithTransform(train_set, train_tf)
val_ds = WithTransform(val_set, eval_tf)
```

> [!IMPORTANT]
> **🟣 Outdated vs current: unseeded or random splits.** A split with no seed changes on every run, so you cannot compare two experiments fairly. Seed it, and better still, **save the list of which files went where**.

> [!NOTE]
> **🔵 Advanced alternative: stratified and group splits (scikit-learn).**
>
> **What changes:** `train_test_split(..., stratify=labels)` keeps the class proportions the same in every split. `GroupShuffleSplit` or `StratifiedGroupKFold` keep all images of one patient or product together.
>
> **Why teams do it:** with a rare class, a plain random split can leave the validation set with almost none of it, giving a meaningless score. With grouped data, only a group-aware split gives an honest result.
>
> ```python
> from sklearn.model_selection import train_test_split
> idx_train, idx_val = train_test_split(range(len(labels)), test_size=0.15, stratify=labels, random_state=42)
> ```

---

## 2.6 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Leakage (duplicates or groups across splits) | Suspiciously high validation score | Hash check; group-aware split |
> | Model learns the background | High accuracy but heatmaps show attention off the object (Chapter 12) | Clean or vary backgrounds |
> | Statistics computed on all data | Great validation, weaker real-world | Compute on training data only |
> | Corrupt image crashes training | Error many epochs in | Run the integrity check first (2.4) |
> | Unseeded or unsaved split | Different validation images every run | Seed the split and save the file lists (2.5) |

---

## Key takeaways

1. **Look at the data** (counts and real pictures) before you train anything.
2. **Split first**, with a seed, into train, validation and test. Keep groups together. Touch the test set once.
3. Run the quality checks (corrupt files, duplicates, label sanity) before the first training run.
4. Anything computed from data (statistics, class weights) comes from the **training split only**.

**Previous:** [Chapter 1: The training recipe](01_training_recipe.md) &nbsp;|&nbsp; **Next:** [Chapter 3: Data pipeline](03_data_pipeline.md)
