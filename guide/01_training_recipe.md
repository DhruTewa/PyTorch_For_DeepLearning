# Chapter 1: The Training Recipe

| | |
|---|---|
| **Level** | Foundations |
| **You will learn** | The five steps behind every neural network, how to prepare data honestly, and what to do when training misbehaves |
| **Prerequisites** | Basic Python, a rough idea of what a tensor is |
| **Ready-made code** | `cv_toolkit/models.py` (`build_mlp`) |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

**In this chapter**
[1.1 The idea](#11-the-idea) · [1.2 The five steps](#12-the-five-steps) · [1.3 The loop](#13-the-loop) · [1.4 Preparing data honestly](#14-preparing-data-honestly) · [1.5 When training misbehaves](#15-when-training-misbehaves) · [1.6 One step up: batches and AdamW](#16-one-step-up-batches-and-adamw) · [1.7 Common problems](#17-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 1.1 The idea

Training a neural network is one routine, repeated thousands of times: **guess, measure how wrong the guess was, work out which internal numbers caused the mistake, nudge them, repeat.**

The routine is the same for images, text and tables. Only what goes in changes. That is why this chapter uses a plain table of numbers: it shows the recipe with nothing else in the way. Every later chapter is this recipe plus better plumbing.

**Running example:** predict delivery time in minutes from distance, hour of day and whether it is a weekend.

---

## 1.2 The five steps

| Step | In plain words | PyTorch tool |
|---|---|---|
| **1. Data** | Numbers in, as tensors, on a similar scale | `torch.tensor`, normalisation |
| **2. Model** | Layers that turn inputs into a guess | `nn.Sequential` |
| **3. Loss** | One number saying how wrong the guess is | `nn.MSELoss` or `nn.CrossEntropyLoss` |
| **4. Optimizer** | The rule for nudging the model | `torch.optim.SGD`, `Adam`, `AdamW` |
| **5. Loop** | Repeat guess, measure, nudge | a `for` loop |

### Choosing the loss

| You are predicting | Use | Why |
|---|---|---|
| A number (minutes, price, temperature) | `nn.MSELoss()` | Penalises big misses much more than small ones |
| One category out of several | `nn.CrossEntropyLoss()` | Compares predicted probabilities with the right answer; expects raw scores (logits), so do not add a softmax before it |
| Yes or no (one output) | `nn.BCEWithLogitsLoss()` | The two-class version of the above |

### Choosing the model size (for a table like this)

A small stack is enough: a few `Linear` layers with `ReLU` between them. The shape is `inputs -> 64 -> 32 -> outputs`.

| Choice | Typical start | Why it matters |
|---|---|---|
| Hidden sizes | 32 to 128 per layer | Too small cannot learn the pattern; too large memorises the training rows |
| Number of hidden layers | 2 | Add more only if the model underfits |
| Activation | `ReLU` | Without it, stacked layers collapse into one straight line |

---

## 1.3 The loop

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1),
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(3000):
    loss = loss_fn(model(features), targets)  # 1. guess and measure the mistake
    optimizer.zero_grad()                     # 2. forget last round's blame
    loss.backward()                           # 3. work out who is to blame
    optimizer.step()                          # 4. nudge the numbers
```

Memorise this order. Every training loop in this guide, however large, is these four lines with more around them.

> [!WARNING]
> **Watch out: `zero_grad()`.** PyTorch *adds up* blame between rounds unless you clear it. Forget the call and training quietly goes wrong with no error message.

---

## 1.4 Preparing data honestly

Put every input column on a similar scale: subtract the average, divide by the spread. Otherwise a column in the thousands drowns out one between 0 and 1.

```python
# split first
train_x, val_x = X[:480], X[480:]

# learn the averages from TRAIN rows only
mean, std = train_x.mean(dim=0), train_x.std(dim=0)

# reuse the SAME numbers on every other split
train_x = (train_x - mean) / std
val_x   = (val_x   - mean) / std
```

> [!IMPORTANT]
> **🟣 Outdated vs current: compute statistics on training data only.**
> Computing the averages over *all* rows is a common shortcut. It lets information about the rows you will later test on leak into training, so your score looks better than it will be in real use. Split first, learn the numbers from the training part, and reuse them everywhere else. The same rule applies to every later chapter (image statistics, class weights, anything "learned from data").

> [!TIP]
> **🟢 Extra: keep a validation set.**
> The original exercise trained on every row and had no held-out data, so there was no honest score. Always hold back a slice (often 10 to 20%) that the model never trains on, and judge it on that slice.

**Feature engineering.** If you know a rule that matters, give it to the model as its own column. In the delivery example, weekday rush hour adds time, and a network will not easily discover "8-10 or 16-19, Monday to Friday" from a raw hour number. A single 0/1 column fixes that.

For images the equivalent is good preprocessing and augmentation (Chapter 3): you cannot hand-write rules for pixels, but you can control what the network sees.

---

## 1.5 When training misbehaves

The most common early failure: the **loss stops at one exact number from the first epochs and never moves**, and every prediction is the same value.

**Why it happens:** when the target has big values (minutes in the tens or hundreds) the loss is large, so the nudges are large. A learning rate that is a little too high throws many `ReLU` units into a state where they output zero for every input ("dead" units). The network then collapses to predicting one constant.

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss frozen at one exact value | Learning rate too high | Lower it (try 10 times smaller) |
| Loss becomes `nan` | Learning rate far too high, or bad input values | Lower it; check data for `nan` or infinity |
| Loss falls very slowly | Learning rate too low, or inputs not scaled | Raise it; scale inputs |
| All predictions identical | Dead units or unscaled target | Lower learning rate; scale the target too |

**Scaling the target too** is the cleaner cure for large-valued targets:

```python
t_mean, t_std = train_y.mean(), train_y.std()
train_y_scaled = (train_y - t_mean) / t_std     # train on this
# ...train...
prediction = model(x) * t_std + t_mean           # convert back to real units
```

---

## 1.6 One step up: batches and AdamW

> [!NOTE]
> **🔵 Advanced alternative: mini-batches and AdamW.**
>
> **What changes:** feed the data in small chunks through a `DataLoader`, and use `AdamW` instead of plain `SGD`.
>
> **Why teams do it:** using all rows in every step works for a few hundred rows but is impossible for images, which do not fit in memory. Small batches take cheaper steps and the randomness helps training settle. `AdamW` adapts its step size for each number separately, so the learning rate needs far less fine-tuning.
>
> ```python
> from torch.utils.data import DataLoader, TensorDataset
>
> loader = DataLoader(TensorDataset(train_x, train_y_scaled), batch_size=32, shuffle=True)
> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
>
> for epoch in range(60):
>     model.train()
>     for xb, yb in loader:
>         loss = loss_fn(model(xb), yb)
>         optimizer.zero_grad()
>         loss.backward()
>         optimizer.step()
> ```
>
> `shuffle=True` reshuffles every epoch so the model cannot learn the row order. This loop is the shape of the real training loop in Chapter 6, with a validation pass added.

---

## 1.7 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Forgot `zero_grad()` | Loss behaves erratically or gets worse | Call it before `backward()` every step |
> | Wrong loss for the task | Loss will not drop, or accuracy is stuck | Number: `MSELoss`. Category: `CrossEntropyLoss` |
> | Statistics from all data | Great validation score, poor real-world score | Learn averages from training rows only |
> | Judged on training data | Score looks perfect, new data fails | Always evaluate on held-out data |
> | Learning rate too high | Loss frozen or `nan` | Lower it |
> | Forgot `model.eval()` when scoring | Scores change between runs (once dropout or BatchNorm are added) | Call `model.eval()` and `torch.no_grad()` for evaluation |

---

## Key takeaways

1. Every network is trained by the same four lines: **loss, `zero_grad`, `backward`, `step`**.
2. Choose the loss from the *kind of answer*: number, category, or yes/no.
3. Scale inputs, and learn that scaling from **training data only**.
4. Hold out validation data, or you have no honest score.
5. A frozen loss almost always means a learning rate that is too high.

**Previous:** [Chapter 0: Project skeleton](00_project_skeleton.md) &nbsp;|&nbsp; **Next:** [Chapter 2: Data, inspecting and splitting](02_data_inspection_splitting.md)
