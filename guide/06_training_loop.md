# Chapter 6: Training the Model

| | |
|---|---|
| **Level** | Core |
| **You will learn** | How to write a trustworthy training loop, choose an optimizer, learning rate and schedule, and save and resume a run |
| **Prerequisites** | Chapters 1 to 3, then 4 or 5 |
| **Ready-made code** | `cv_toolkit/train.py` (`simple_fit`, `fit`, `build_optimizer`, `build_scheduler`, `save_checkpoint`), `cv_toolkit/lightning_tools.py` |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

**In this chapter**
[6.1 The idea](#61-the-idea) · [6.2 One epoch](#62-one-epoch-train-and-evaluate) · [6.3 The full loop](#63-the-full-loop) · [6.4 Optimizer and learning rate](#64-optimizer-and-learning-rate) · [6.5 Schedulers](#65-learning-rate-schedules) · [6.6 Checkpoints](#66-saving-and-resuming) · [6.7 Common problems](#67-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 6.1 The idea

Chapter 1 gave you the bare loop. A real training run adds five things around it:

1. A **validation pass** after each epoch, to see how the model does on images it has not trained on.
2. **Saving the best model**, not just the last one.
3. A **schedule** that lowers the learning rate as training matures.
4. **Early stopping**, so you do not waste time or overfit.
5. **Checks** that tell you whether the run is healthy (these get their own chapter: [Chapter 7](07_training_health.md)).

---

## 6.2 One epoch: train and evaluate

An **epoch** is one full pass over the training images. Training and evaluating are two separate functions, because they behave differently.

```python
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()                                   # dropout on, BatchNorm learns
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)       # weight by batch size (last batch is smaller)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()                                    # no gradients: faster, less memory
def evaluate(model, loader, loss_fn, device):
    model.eval()                                    # dropout off, BatchNorm uses stored averages
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n
```

| Mode switch | What changes |
|---|---|
| `model.train()` | Dropout randomly drops units. BatchNorm uses the current batch and updates its running averages |
| `model.eval()` | Dropout is off. BatchNorm uses the stored running averages. Results become repeatable |
| `torch.no_grad()` | Stops gradient bookkeeping. Does **not** change dropout or BatchNorm: you need `eval()` for that |

> [!WARNING]
> **Watch out.** Forgetting `model.eval()` when validating, or forgetting `model.train()` after, are the two most common silent bugs. After `evaluate()` the *next* `train_one_epoch` must call `model.train()` again, which is why it is inside the function.

**Two details that surprise people.**
- `loss.item()` copies a number off the GPU. Call it every step and it can slow training. Fine for most work.
- **Training accuracy can read *lower* than validation accuracy in early epochs.** Training numbers are averaged *while the model is changing* (and with dropout on); validation is measured at the end of the epoch with the finished weights. It is normal. Compare the curves' trend, not single numbers.

---

## 6.3 The full loop

```python
def fit(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, epochs, patience=5, ckpt="best.pt"):
    model.to(device)
    best_loss, best_epoch, bad = float("inf"), 0, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(va_loss)                      # ReduceLROnPlateau needs the metric
        for key, value in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[key].append(value)

        print(f"epoch {epoch:2d} | train {tr_loss:.3f}/{tr_acc:.3f} | val {va_loss:.3f}/{va_acc:.3f} "
              f"| lr {optimizer.param_groups[0]['lr']:.1e}")

        if va_loss < best_loss:                      # improvement: save it
            best_loss, best_epoch, bad = va_loss, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:                                        # no improvement
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch}; best was epoch {best_epoch}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))   # go back to the best
    return model, history
```

Usage:

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
model, history = fit(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, epochs=30)
```

**Key choices inside the loop:**

| Choice | Guidance |
|---|---|
| What to watch for "best" | **Validation loss** is smoother and more informative. Validation accuracy is what you care about but jumps in small datasets. If you watch accuracy, expect noise |
| `patience` (epochs without improvement before stopping) | 5 to 10 epochs is typical. Smaller for big datasets, larger for noisy ones |
| Scheduler patience | Smaller than the early-stop patience, so the learning rate drops *before* you give up |
| Restore the best weights at the end | Yes. The last epoch is rarely the best |

> [!NOTE]
> **🔵 Advanced alternative: PyTorch Lightning.**
>
> **What changes:** you describe *what to do for one batch* and Lightning supplies the loop: device moves, `train()`/`eval()` switching, mixed precision, checkpointing, early stopping and multiple GPUs.
>
> **Why teams do it:** less boilerplate to get wrong, and the same code scales from a laptop to several GPUs by changing a setting. The cost is another layer to learn when debugging.
>
> ```python
> import lightning.pytorch as pl
>
> class Classifier(pl.LightningModule):
>     def __init__(self, model, lr=1e-3):
>         super().__init__()
>         self.model, self.lr = model, lr
>     def training_step(self, batch, _):
>         x, y = batch
>         loss = nn.functional.cross_entropy(self.model(x), y)
>         self.log("train_loss", loss)
>         return loss
>     def validation_step(self, batch, _):
>         x, y = batch
>         self.log("val_loss", nn.functional.cross_entropy(self.model(x), y))
>     def configure_optimizers(self):
>         return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
>
> trainer = pl.Trainer(max_epochs=30, callbacks=[pl.callbacks.EarlyStopping("val_loss", patience=5)])
> trainer.fit(Classifier(model), train_loader, val_loader)
> ```
>
> Use the plain loop to learn and for small projects. Consider Lightning when you need multi-GPU, long runs or a team convention.

---

## 6.4 Optimizer and learning rate

The **learning rate** is the single most important setting. Too high and training jumps around or collapses; too low and it crawls.

| Optimizer | In plain words | Typical learning rate | Use when |
|---|---|---|---|
| `SGD` with `momentum=0.9` | Simple steady descent with a running push in the same direction | 0.01 to 0.1 | Training from scratch for many epochs, or when you want the best final accuracy and will tune carefully |
| `Adam` | Adapts the step for each number separately | 1e-3 (from scratch), 1e-4 (fine-tuning) | A good default; forgiving |
| `AdamW` | Adam with weight decay done correctly | 1e-3, with `weight_decay` 1e-2 to 1e-4 | **Preferred default today** |

**Learning rate by situation (starting points, then adjust):**

| Situation | Start at |
|---|---|
| Training a small CNN from scratch, Adam/AdamW | 1e-3 |
| New head on a frozen pretrained backbone | 1e-3 |
| Unfreezing pretrained layers | 1e-4 to 1e-5 |
| SGD with momentum | 1e-2 |
| Larger batch size | You can often raise it a bit |

**Weight decay** gently pulls weights toward zero, which discourages overfitting. Typical values: 1e-4 to 1e-2 with AdamW.

**Batch size:** 16 to 64 is typical. Smaller batches are noisier (sometimes helpful) and use less memory. Larger batches are faster per epoch but may need a higher learning rate. If you hit out-of-memory errors, lower it, or use gradient accumulation (7.4).

> [!IMPORTANT]
> **🟣 Outdated vs current: Adam with `weight_decay`.** In plain `Adam`, the `weight_decay` argument is applied in a way that interacts badly with the adaptive steps. `AdamW` fixes this and is the current default for almost everything.

> [!TIP]
> **🟢 Extra: find a learning rate in two minutes.** Run a few hundred steps while raising the learning rate from tiny (1e-7) to large (1), and record the loss. Pick a rate about **10 times lower than where the loss starts to blow up**. Libraries automate this ("learning rate finder"). If you do it by hand, a quick try of 1e-2, 1e-3, 1e-4 for two or three epochs each tells you a lot.

---

## 6.5 Learning rate schedules

A **schedule** lowers the learning rate over time: big steps early to make progress, small steps later to settle precisely.

| Scheduler | Behaviour | Use when |
|---|---|---|
| `ReduceLROnPlateau` | Multiplies the rate by a factor (for example 0.1) when the monitored metric stops improving | **Good default.** Needs no guess about how long training will last |
| `StepLR` | Cuts the rate every N epochs | Simple, predictable |
| `CosineAnnealingLR` | Smooth decline following a cosine curve to near zero | You know the total epochs. Popular and strong |
| `OneCycleLR` | Warms up, peaks, then declines (stepped **every batch**) | Fast training in a fixed budget; often the best results per hour |

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# call scheduler.step() once per epoch, AFTER optimizer.step()
```

> [!WARNING]
> **Watch out.**
> - `ReduceLROnPlateau.step(metric)` needs the metric; the others take no argument.
> - `OneCycleLR` steps every *batch*, the others every *epoch*. Mixing this up is a quiet bug.
> - Call `optimizer.step()` before `scheduler.step()`.
> - Print the learning rate each epoch (as in `fit` above) so you can see the schedule working.

> [!TIP]
> **🟢 Extra: warmup.** Starting with a very small learning rate and rising over the first few hundred steps prevents early instability, especially with large batches or Transformers. `OneCycleLR` includes it.

---

## 6.6 Saving and resuming

Always save the **`state_dict`** (a dictionary of numbers), not the whole model object.

```python
torch.save(model.state_dict(), "best.pt")

model = build_model()                                        # recreate the SAME architecture
model.load_state_dict(torch.load("best.pt", map_location="cpu", weights_only=True))
model.eval()
```

**Two kinds of checkpoint:**

| | Contains | Purpose |
|---|---|---|
| **Best** | Model weights only | The model you will use |
| **Last / resume** | Weights **and optimizer, scheduler, epoch** | Continue an interrupted run |

```python
torch.save({
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),     # Adam keeps running averages that matter
    "scheduler": scheduler.state_dict(),
    "best_loss": best_loss,
}, "last.pt")

ck = torch.load("last.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
start_epoch = ck["epoch"] + 1
```

> [!WARNING]
> **Watch out.**
> - Resuming without the optimizer state restarts its running averages, so the loss often jumps.
> - Move the model to the device *before* creating or loading the optimizer.
> - Save the **class names and preprocessing settings** next to the weights (Chapter 14), or a file you cannot interpret is useless in six months.
> - Saving the whole model with `torch.save(model, ...)` ties the file to your exact code layout and needs unsafe loading. Avoid it.

---

## 6.7 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Forgot `model.eval()` or `model.train()` | Validation score changes between identical runs, or never improves | Set the mode inside the train and evaluate functions |
> | Evaluating with gradients on | Memory grows, evaluation is slow | `@torch.no_grad()` |
> | Last epoch used instead of the best | Final score below the best seen | Restore best weights (6.3) |
> | Resume without optimizer state | Loss jumps after resuming | Save and load optimizer and scheduler |
> | Scheduler stepped in the wrong place | Learning rate is not what you expect | Print the rate each epoch; `OneCycle` per batch, others per epoch |
> | Whole model saved with `torch.save(model)` | Load fails after code changes | Save `state_dict` |
> | Different results every run | Scores differ without any change | Seed everything (Chapter 0); average over several seeds |

---

## Key takeaways

1. A training run is **train one epoch, evaluate, step the scheduler, save the best, stop early, restore the best**.
2. `train()` and `eval()` differ. `no_grad` is separate. Get both right every time.
3. **AdamW** with a learning rate of 1e-3 (1e-4 or lower for pretrained layers) is a safe default. The learning rate is the most important number you set.
4. Save `state_dict`s. Keep a separate "last" checkpoint with optimizer and scheduler state for resuming.

**Previous:** [Chapter 5: Transfer learning](05_transfer_learning.md) &nbsp;|&nbsp; **Next:** [Chapter 7: Training health and speed](07_training_health.md)
