# Chapter 7: Training Health and Speed

| | |
|---|---|
| **Level** | Core |
| **You will learn** | How to check that a run is healthy before you trust it, read loss curves, and make training faster |
| **Prerequisites** | Chapter 6 |
| **Ready-made code** | `cv_toolkit/train.py` |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

**In this chapter**
[7.1 The idea](#71-the-idea) · [7.2 Sanity checks](#72-sanity-checks-before-you-trust-a-run) · [7.3 Reading the curves](#73-reading-the-curves) · [7.4 Faster training](#74-faster-training) · [7.5 Common problems](#75-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 7.1 The idea

A loop that runs without errors is not the same as a loop that works. A model can train for hours with a bug in the data, the labels or the learning rate and still print numbers that look plausible. This chapter is about three habits:

1. **Check before you commit.** A few cheap tests catch most silent bugs before a long run.
2. **Read the curves.** Training and validation loss show whether you are fighting overfitting, underfitting or an unstable learning rate.
3. **Speed up last.** Make the run correct first, then make it fast, and only when you actually need to.

---

## 7.2 Sanity checks before you trust a run

> [!TIP]
> **🟢 Extra: this whole section.** A broken run still prints numbers. These checks catch the common bugs in minutes, before you spend hours.

Do them in this order whenever you change the model, the data or the loop.

| # | Check | How | What failure means |
|---|---|---|---|
| 1 | **Look at a batch** | Plot 8 images with labels | Wrong labels, broken transforms, missing normalisation |
| 2 | **Initial loss** | Loss on an untrained model should be near `ln(number of classes)` (1.10 for 3 classes, 0.69 for 2, 6.9 for 1,000). A little above is normal | Many times larger: inputs not normalised, or a very large initialisation |
| 3 | **Overfit one tiny batch** | Train on 16 images for 100 steps. Loss should go to nearly zero | It cannot even memorise 16 images: bug in model, loss, labels or loop |
| 4 | **Learning rate** | Try 1e-2, 1e-3, 1e-4 for a few epochs | Loss frozen or `nan`: too high. Barely moving: too low |
| 5 | **Gradients exist** | After `loss.backward()`, check that parameters have non-zero `.grad` | All `None` or zero: frozen by mistake, or detached from the graph |
| 6 | **Train/eval mode** | Evaluate twice on the same data: results must be identical | Differences: still in `train()` mode |
| 7 | **Prediction spread** | Count predicted classes on validation | All one class: model has collapsed (learning rate, imbalance) |

```python
# Check 3: overfit one batch
xb, yb = next(iter(train_loader))
for step in range(100):
    loss = loss_fn(model(xb.to(device)), yb.to(device))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
print(loss.item())          # expect close to 0

# Check 5: gradient size
def grad_norm(model):
    return torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
```

> [!NOTE]
> **Gradient size as a gauge.** A gradient norm that suddenly explodes (thousands) or shrinks to nearly zero signals instability. If it spikes, add **gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` between `loss.backward()` and `optimizer.step()`.

---

## 7.3 Reading the curves

Plot training and validation **loss** against epoch (and accuracy next to it). The picture diagnoses most problems.

| What you see | Meaning | Next step |
|---|---|---|
| Both losses fall, close together, then flatten | Healthy | Done, or try a bigger model for more |
| Training loss keeps falling, validation loss rises | **Overfitting** | Chapter 8 |
| Both losses high and flat | **Underfitting** (model too small, learning rate too low, or training too short) | Bigger model, more epochs, higher learning rate |
| Loss jumps up and down | Learning rate too high, or batch too small | Lower the rate, use a scheduler |
| Validation loss lower than training loss | Normal early on (see 6.2). Persistent: validation set is easy or has leaked | Check the split (Chapter 2) |
| Loss stuck at one exact value | Learning rate too high (dead units), or `ln(C)` plateau from a bug | Lower rate; run checks 2 and 3 |
| Loss is `nan` | Learning rate far too high, or `nan` in data | Lower the rate, clip gradients, check data |

---

## 7.4 Faster training

> [!CAUTION]
> **Not tested.** The mixed-precision, `torch.compile` and multi-GPU snippets below were not run on a GPU. Treat them as patterns and check them against your PyTorch version.

> [!TIP]
> **🟢 Extra: this whole section.** Do these only when you actually need speed. First measure where the time goes: if the GPU sits idle, the data loading is the bottleneck (raise `num_workers`), not the model.

| Technique | What it does | Gain | Cost or risk |
|---|---|---|---|
| **Data loading** (`num_workers`, `pin_memory`) | Prepares the next batch while the GPU works | Often large | None |
| **Mixed precision** | Uses 16-bit numbers for most arithmetic | Faster, about half the memory, on modern GPUs | Needs a gradient scaler for `float16` |
| **Gradient accumulation** | Adds up gradients over several small batches before stepping | Lets you train with a large effective batch in small memory | Slightly slower per epoch |
| **`torch.compile`** | Optimises the model's execution | Often faster after a first slow step | Compile time; occasional edge-case errors |
| **Multi-GPU (DDP)** | Splits batches across GPUs | Near-linear speed-up | Needs launch scripts; scale the learning rate |

**Mixed precision (GPU):**

```python
scaler = torch.amp.GradScaler("cuda")
for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast("cuda", dtype=torch.float16):     # math in 16-bit
        loss = loss_fn(model(x), y)
    scaler.scale(loss).backward()                          # scaled so small gradients do not vanish
    scaler.unscale_(optimizer)                             # needed before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

On newer GPUs `dtype=torch.bfloat16` has a wider range and **needs no scaler**: just wrap the forward pass and loss in `autocast`.

**Gradient accumulation:**

```python
accum = 4                                       # effective batch = batch_size x 4
optimizer.zero_grad()
for i, (x, y) in enumerate(train_loader):
    loss = loss_fn(model(x.to(device)), y.to(device)) / accum     # divide so the sum matches a big batch
    loss.backward()
    if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
        optimizer.step()
        optimizer.zero_grad()
```

**Compile and multi-GPU:**

```python
model = torch.compile(model)        # one line; the first epoch is slower while it compiles
# Multi-GPU: launch with  torchrun --nproc_per_node=4 train.py  and wrap:  DistributedDataParallel(model)
# (also needs a DistributedSampler in the loader). Lightning does all of this with Trainer(devices=4).
```

> [!WARNING]
> **Watch out.** BatchNorm with a tiny batch (say 2 or 4) gives noisy statistics. With accumulation, BatchNorm still sees the *small* batch, so accumulation is not exactly equal to a true large batch for networks using BatchNorm.

---

---

## 7.5 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Loss frozen or `nan` | See 7.3 | Lower learning rate; run sanity checks |
> | Cannot overfit 16 images | Check 3 fails | Fix the bug before tuning anything |
> | GPU idle, training slow | Low GPU use while epochs crawl | More `num_workers`, `pin_memory`, bigger batch |
> | Speed-up changes the results | Scores shift after enabling mixed precision or compile | Compare against the plain run on the same seed before trusting it |

---

## Key takeaways

1. **Run the sanity checks**, especially "overfit one tiny batch", before spending hours on a run.
2. Read the loss curves: they tell you whether to fight overfitting, underfitting, or an unstable learning rate.
3. Measure where the time goes before optimising. Data loading is the usual first bottleneck.

**Previous:** [Chapter 6: Training the model](06_training_loop.md) &nbsp;|&nbsp; **Next:** [Chapter 8: Beating overfitting](08_overfitting.md)
