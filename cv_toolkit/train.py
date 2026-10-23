"""
train.py  |  Chapters 6 and 7: Training
=========================================================================
  1. Optimizer / scheduler factories
  2. Checkpoints
  3. EarlyStopping
  4. train_one_epoch  (the heart of every loop)
  5. simple_fit       (the "learn it" loop, no extras)
  6. fit              (the "use it" loop: plus checkpoint, AMP,
                       early stopping, resume)
  7. train_triplet    (Siamese / similarity learning)

Lightning versions live in lightning_tools.py. Optuna lives in tuning.py.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from .evaluate import evaluate
from .models import freeze_batchnorm_stats

try:  # progress bars are nice-to-have, not required
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# =========================================================================
# 1. OPTIMIZER AND SCHEDULER
# =========================================================================
def build_optimizer(params: Iterable, name: str = "adamw", lr: float = 1e-3, weight_decay: float = 1e-4, **kwargs) -> torch.optim.Optimizer:
    """
    In plain words: creates the thing that nudges the weights after each batch.
    AdamW is a good default; SGD with momentum is the classic for CNNs.

    
    Change for your project: name, lr (the most important number), weight_decay.
    Gotcha: when part of the model is frozen, pass only the trainable weights:
        build_optimizer(trainable_parameters(model), ...)
    """
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    if name == "sgd":
        kwargs.setdefault("momentum", 0.9)
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, **kwargs)
    raise ValueError(f"unknown optimizer {name!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, name: Optional[str] = "plateau", **kwargs):
    """
    In plain words: lowers the learning rate over time (big steps first, small
    careful steps later). "plateau" waits until validation stops improving,
    then cuts the rate.

    
    """
    if name is None:
        return None
    name = name.lower()
    if name == "plateau":
        kwargs.setdefault("factor", 0.1)
        kwargs.setdefault("patience", 2)
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", **kwargs)
    if name == "step":
        kwargs.setdefault("step_size", 5)
        kwargs.setdefault("gamma", 0.1)
        return lr_scheduler.StepLR(optimizer, **kwargs)
    if name == "cosine":
        kwargs.setdefault("T_max", 10)
        return lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    raise ValueError(f"unknown scheduler {name!r}")


# ADVANCED (production alternative): warmup + one-cycle
# What changes: ramp the learning rate UP for the first steps, then down.
# Why teams do it: big models and big learning rates blow up if they start
#   at full speed; OneCycle often trains faster with the same final accuracy.
#
#   sched = lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=E, steps_per_epoch=len(train_loader))
#   # call sched.step() after EVERY BATCH (not every epoch) for this scheduler.


# =========================================================================
# 2. CHECKPOINTS
# =========================================================================
def save_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0, **extra) -> None:
    """
    In plain words: saves everything needed to continue training later (or to
    load the best model for use).
    
    Tip: save state_dict (weights), not the whole model object. A whole-model
    pickle breaks when you rename or move your code.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": int(epoch), "model_state_dict": model.state_dict(), **extra}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, map_location="cpu") -> dict:
    """
    In plain words: loads a checkpoint back into the model (and optimizer, if
    you want to resume). Returns the extra info you saved (epoch, metrics...).
    MODERNIZE: torch.load defaults to weights_only=True from torch 2.6. That is
    fine for this payload (tensors + numbers). Anything with custom Python
    objects inside needs weights_only=False, and then you must trust the file:
    pickle can run code.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# =========================================================================
# 3. EARLY STOPPING
# =========================================================================
class EarlyStopping:
    """
    In plain words: stop training when the validation score has not improved
    for `patience` epochs. Saves time and limits overfitting.

     This is the same idea for plain PyTorch loops.
    """

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 0.0):
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Feed this epoch's value. Returns True if it was a new best."""
        improved = self.best is None or (value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta)
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


# =========================================================================
# 4. ONE EPOCH OF TRAINING
# =========================================================================
def _make_scaler(enabled: bool):
    # torch.amp.GradScaler exists from torch 2.3; older versions: torch.cuda.amp.GradScaler
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
    amp: bool = False,
    grad_clip: Optional[float] = None,
    freeze_bn: bool = False,
    progress: bool = False,
) -> Dict[str, float]:
    """
    In plain words: one full pass over the training data. For every batch:
    predict, measure the error, work out how to improve, take a small step.

    
    Returns: {"loss": average per sample, "acc": fraction correct}.

    The five lines that matter (memorize this order):
        optimizer.zero_grad()      # forget the last batch's gradients
        outputs = model(inputs)    # predict
        loss = loss_fn(outputs, targets)   # how wrong?
        loss.backward()            # which way to move each weight?
        optimizer.step()           # move

    Gotchas:
      * loss.item() converts to a Python number. Adding `loss` itself keeps the
        whole graph in memory (a classic slow memory leak).
      * Average per SAMPLE (loss * batch_size / n), otherwise a smaller last
        batch counts as much as a full one.
    """
    model.train()
    if freeze_bn:
        freeze_batchnorm_stats(model)
    use_amp = amp and device.type == "cuda"
    scaler = scaler or _make_scaler(use_amp)
    iterator = tqdm(loader, leave=False) if (progress and tqdm) else loader

    loss_sum, correct, total = 0.0, 0, 0
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)  # clip real gradient sizes, not scaled ones
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = targets.size(0)
        loss_sum += loss.item() * bs
        correct += (outputs.argmax(1) == targets).sum().item()
        total += bs
    return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1)}


# ADVANCED (production alternative): gradient accumulation and torch.compile
# What changes: (a) add up gradients over several small batches before one
#   optimizer step; (b) compile the model for speed.
# Why teams do it: (a) simulate a big batch when the GPU memory is small;
#   (b) 10-50% faster training on modern GPUs for many models.
#
#   accum = 4
#   for i, (x, y) in enumerate(loader):
#       loss = loss_fn(model(x.to(device)), y.to(device)) / accum
#       loss.backward()
#       if (i + 1) % accum == 0:
#           optimizer.step(); optimizer.zero_grad(set_to_none=True)
#
#   model = torch.compile(model)      # PyTorch 2.x; first epoch is slower (compiling)
#
# Multi-GPU: launch with `torchrun --nproc_per_node=4 train.py`, wrap the model
#   in torch.nn.parallel.DistributedDataParallel and use DistributedSampler in
#   the DataLoader. (Or let Lightning do it: Trainer(devices=4, strategy="ddp").)


# =========================================================================
# 5. THE "LEARN IT" LOOP
# =========================================================================
def simple_fit(model, train_loader, val_loader, loss_fn, optimizer, device, epochs: int):
    """
    In plain words: the smallest complete training loop. Train a round, check
    on validation, remember the best weights, repeat. Read this one first.

    
    Use fit() below for real projects.
    """
    model.to(device)
    best_acc, best_state = 0.0, None
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        tr = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        va = evaluate(model, val_loader, device, loss_fn)
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        print(f"Epoch {epoch + 1}/{epochs} | train loss {tr['loss']:.4f} | val loss {va['loss']:.4f} | val acc {va['acc']:.4f}")
        if va["acc"] > best_acc:
            best_acc, best_state = va["acc"], copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# =========================================================================
# 6. THE "USE IT" LOOP
# =========================================================================
def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    scheduler=None,
    monitor: str = "val_acc",
    patience: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    amp: bool = False,
    grad_clip: Optional[float] = None,
    freeze_bn: bool = False,
    verbose: bool = True,
) -> tuple:
    """
    TEMPLATE: the training loop you can reuse in most classification projects.
    In plain words: train, validate, keep the best version, optionally stop
    early, optionally save to disk and resume after a crash.

    

    Change for your project: monitor ("val_acc" higher-is-better, or
    "val_loss" lower-is-better), patience, checkpoint_path.
    Returns: (model with the BEST weights loaded, history dict of lists).

    Scheduler rule built in: ReduceLROnPlateau needs the validation loss
    passed to step(); every other scheduler just steps once per epoch.
    """
    if monitor not in ("val_acc", "val_loss"):
        raise ValueError("monitor must be 'val_acc' or 'val_loss'")
    mode = "max" if monitor == "val_acc" else "min"
    stopper = EarlyStopping(patience=patience or epochs + 1, mode=mode)
    model.to(device)
    scaler = _make_scaler(amp and device.type == "cuda")
    history: Dict[str, List[float]] = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr")}
    best_state, best_epoch, start_epoch = None, 0, 0

    if resume_from and Path(resume_from).exists():
        ckpt = load_checkpoint(resume_from, model, optimizer, map_location=device)
        start_epoch = ckpt.get("epoch", -1) + 1
        stopper.best = ckpt.get("best")
        print(f"Resumed from {resume_from} at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler, amp, grad_clip, freeze_bn)
        va = evaluate(model, val_loader, device, loss_fn)

        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va["loss"])
            else:
                scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        for k, v in (("train_loss", tr["loss"]), ("train_acc", tr["acc"]), ("val_loss", va["loss"]), ("val_acc", va["acc"]), ("lr", lr_now)):
            history[k].append(v)

        improved = stopper.step(va["acc"] if monitor == "val_acc" else va["loss"])
        if improved:
            best_epoch = epoch + 1
            # keep a CPU copy so the best weights do not eat GPU memory
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path:
                save_checkpoint(checkpoint_path, model, optimizer, epoch, best=stopper.best, val_acc=va["acc"], val_loss=va["loss"])
        if verbose:
            star = " *" if improved else ""
            print(
                f"Epoch {epoch + 1:>3}/{epochs} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
                f"val loss {va['loss']:.4f} acc {va['acc']:.4f} | lr {lr_now:.2e} | {time.time() - t0:.1f}s{star}"
            )
        if stopper.should_stop:
            print(f"Early stopping: no improvement in {stopper.patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"Best {monitor} {stopper.best:.4f} at epoch {best_epoch}; best weights restored.")
    return model, history


# =========================================================================
# 7. TRIPLET (SIAMESE) TRAINING
# =========================================================================
def train_triplet(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
    verbose: bool = True,
):
    """
    In plain words: teach the network to put pictures of the same class close
    together and different classes far apart. Each batch has (anchor,
    positive, negative); the triplet loss pulls anchor-positive together and
    pushes anchor-negative apart.

    
    Use: loss_fn = nn.TripletMarginLoss(margin=1.0, p=2.0)
    There is no accuracy here. A falling loss is the signal; the real test
    is retrieval quality (evaluate.find_nearest on held-out images).
    Returns: (model with the lowest-loss weights restored, list of epoch losses).
    """
    model.to(device)
    best_loss, best_state, losses = float("inf"), None, []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for anchor, positive, negative in loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad(set_to_none=True)
            a, p, n = model(anchor, positive, negative)
            loss = loss_fn(a, p, n)
            loss.backward()
            optimizer.step()
            running += loss.item()
        epoch_loss = running / len(loader)
        losses.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss, best_state = epoch_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"Epoch {epoch}/{epochs} | triplet loss {epoch_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, losses
