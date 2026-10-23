# Chapter 0: The Project Skeleton

| | |
|---|---|
| **Level** | Foundations (read this first) |
| **You will learn** | What a real computer vision project needs around the model, and the whole lifecycle at a glance |
| **Prerequisites** | None |
| **Ready-made code** | `cv_toolkit/config.py` |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

> [!TIP]
> **🟢 Extra: this whole chapter.** Tutorials teach the model. This chapter covers everything around it that decides whether a project can be repeated, shared and trusted.

> **Code conventions.** Snippets in this guide assume these imports unless a snippet shows its own:
> `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`, `from torchvision import datasets, models, transforms`, and `from torch.utils.data import DataLoader, Dataset`. A variable named `device` is your `torch.device` (Chapter 0.6).

**In this chapter**
[0.1 The idea](#01-the-idea) · [0.2 The lifecycle](#02-the-lifecycle-at-a-glance) · [0.3 Folder layout](#03-folder-layout) · [0.4 Settings in one place](#04-settings-in-one-place) · [0.5 Reproducibility](#05-reproducibility) · [0.6 Device and logging](#06-device-and-logging) · [0.7 Tracking experiments](#07-tracking-experiments) · [0.8 Version control and data](#08-version-control-and-data) · [0.9 Common problems](#09-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 0.1 The idea

A notebook that trains a model once is a demo. A project is something you can **rerun next month, hand to a colleague, and explain when someone asks "why does this model say that?"** The model code is maybe a fifth of the work. The rest is structure.

This chapter is a map of everything you should know exists before writing a line of model code.

---

## 0.2 The lifecycle at a glance

| Stage | The question it answers | Chapter |
|---|---|---|
| Set up | Can I rerun this exactly? | 0 |
| Learn the recipe | What is training, really? | 1 |
| Data | Is my data clean, split honestly, and prepared well? | 2 |
| Model | Which network, and why? | 3, 4 |
| Train | How do I train it and know it is working? | 5 |
| Fight overfitting | Why is it great on training data and poor on new data? | 6 |
| Tune | Which settings work best? | 7 |
| Evaluate | How good is it really, and where does it fail? | 8 |
| Search by similarity | Can I find "images like this one"? | 9 |
| Explain | Is it right for the right reasons? | 10 |
| Shrink | Can it be smaller and faster? | 11 |
| Ship | How do others use it? | 12 |
| Protect | Will it keep working next quarter? | 13, 14 |

**The loop you will actually live in:** data check, train a small baseline, look at the failures, improve the data or the model, repeat. Beginners over-invest in model choice. Experienced people spend most of their time on data and on reading mistakes.

---

## 0.3 Folder layout

A layout you can copy:

```
my_project/
├── README.md              # what this is, how to run it
├── pyproject.toml         # dependencies, pinned versions
├── configs/               # one file per experiment (settings only, no code)
├── data/                  # NOT in git (too big); see 0.8
│   ├── raw/               # untouched originals, never edited
│   └── processed/         # anything derived from raw
├── src/my_project/
│   ├── data.py            # datasets, transforms, loaders
│   ├── models.py          # network definitions
│   ├── train.py           # training loop
│   ├── evaluate.py        # metrics, error analysis
│   └── predict.py         # load a saved model and predict
├── scripts/               # thin command-line entry points
├── notebooks/             # exploration only; no logic you depend on
├── runs/                  # outputs: weights, logs, configs (NOT in git)
└── tests/                 # small automated checks (Chapter 15)
```

**Why it matters:**
- **Notebooks for exploring, `.py` files for anything you rely on.** Notebooks hide state (cells run out of order), cannot be tested easily, and make diffs unreadable.
- **`raw/` is read-only.** If you can regenerate everything else from it, you can recover from any mistake.
- **Settings live in `configs/`, not buried in code.** Then a run is "this config on this code", which is reproducible.

---

## 0.4 Settings in one place

Keep every setting that you might change in one object, and save it next to the results.

```python
from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class Config:
    seed: int = 42
    image_size: int = 224
    num_classes: int = 2
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    data_dir: str = "data/processed"
    out_dir: str = "runs/exp1"

cfg = Config(lr=3e-4)
Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.out_dir, "config.json").write_text(json.dumps(asdict(cfg), indent=2))
```

Months later, `runs/exp1/config.json` tells you exactly how that model was made.

**Rules of thumb:** one flat object, no logic inside it, and every number you might tune goes in it. Hard-coded numbers scattered through the code are how "which learning rate did I use?" becomes unanswerable.

> [!NOTE]
> **🔵 Advanced alternative: YAML files with command-line overrides (Hydra).**
>
> **What changes:** settings come from a YAML file in `configs/`, and you override any of them when launching: `python train.py lr=3e-4 batch_size=64`.
>
> **Why teams do it:** you can launch twenty experiments from a shell script without editing Python, and each run logs the exact configuration it used. The `pydantic` library adds the same idea with type checking, so a wrong type fails immediately rather than 40 minutes into training.

---

## 0.5 Reproducibility

Random choices appear everywhere: initial weights, shuffling, augmentation, the train/validation split. Fix the seed and two runs with the same settings give (nearly) the same result.

```python
import random, numpy as np, torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(cfg.seed)
```

> [!WARNING]
> **Watch out.**
> - On a GPU, some operations are not repeatable even with a seed, so two runs can differ slightly. `torch.use_deterministic_algorithms(True)` forces repeatable kernels at the cost of speed and some errors for operations with no deterministic version. Use it when debugging, not by default.
> - Data loader workers each have their own random state. Seed them too (Chapter 3).
> - A difference of a fraction of a percent between two runs can be pure luck. Before believing that a change helped, run it with two or three different seeds.

**What reproducibility really needs (beyond the seed):** the same code (git commit), the same settings (the config file), the same data (a fixed split, see 0.8), and the same library versions (pinned in `pyproject.toml` or `requirements.txt`).

---

## 0.6 Device and logging

**Pick the hardware automatically:**

```python
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")          # NVIDIA GPU
    if torch.backends.mps.is_available():
        return torch.device("mps")           # Apple chip
    return torch.device("cpu")
```

**Write down what happened.** `print` disappears when the window closes. Use the standard `logging` module so progress goes to the screen *and* a file:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("runs/exp1/run.log")],
)
log = logging.getLogger("cv")
log.info("epoch %d  val_acc %.3f", epoch, val_acc)
```

> [!IMPORTANT]
> **🟣 Outdated vs current: `print()` for progress.** It is fine while learning, but it leaves no history once the notebook closes. Real projects log to a file.

---

## 0.7 Tracking experiments

By the tenth run you will not remember which settings produced which score. An experiment tracker records settings, metrics and files for every run, and lets you compare them in a table.

> [!TIP]
> **🟢 Extra: how to choose.**
>
> | Tool | Good for |
> |---|---|
> | A CSV or JSON per run | One person, few runs. Costs nothing |
> | TensorBoard | Simple local charts of loss and accuracy per epoch |
> | MLflow | Self-hosted comparison of many runs, plus a model registry |
> | Weights & Biases | Hosted dashboards, team sharing |

> [!NOTE]
> **🔵 Advanced alternative: log to a tracker from the training loop.**
>
> ```python
> from torch.utils.tensorboard import SummaryWriter
> writer = SummaryWriter(log_dir=cfg.out_dir)
> writer.add_scalar("val/acc", val_acc, epoch)
> ```
>
> **Why teams do it:** compare 50 runs in a dashboard and see which settings won, instead of searching log files.

**Start simple:** save `config.json` plus a `metrics.csv` per run. Move to a tracker when you notice yourself comparing runs by hand.

---

## 0.8 Version control and data

- **Code and configs go in git.** Weights, datasets and run outputs do not (they are large). Put `data/` and `runs/` in `.gitignore`.
- **Data changes silently.** If someone adds images to a folder, your "same" experiment is now a different one. Options, from simplest: record a count and checksum of the data in your run folder, keep a frozen copy of the split, or use a data versioning tool (DVC is the common one).
- **Save the split itself** (a file listing which image is in train, validation and test), not just the seed. Seeds give the same split only while the data and library stay the same.

---

## 0.9 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Cannot reproduce a good result | Rerun gives a different score and you cannot say what changed | Seed, save the config, record the git commit and the split |
> | Results depend on cell order | Notebook gives different answers after "Run all" | Move logic into `.py` files; use notebooks for looking only |
> | Data silently changed | Same code, new score | Record data checksums; freeze the split |
> | A "gain" that is noise | Improvement disappears with another seed | Compare over several seeds before believing it |
> | Settings scattered in code | "Which learning rate was that?" | One config object, saved with every run |
> | Weights not tied to their code | A saved file that no longer loads | Save the config and the class names with the weights |

---

## Key takeaways

1. A project is something you can **rerun, share and explain**. The model is one part of it.
2. **Notebooks to explore, `.py` files to rely on.** Never edit raw data.
3. **One config object**, saved with every run.
4. **Seed everything**, and do not trust small differences from one run.
5. Log to a file, and track experiments once you have more than a handful.

**Next:** [Chapter 1: The training recipe](01_training_recipe.md)
