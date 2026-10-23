"""
lightning_tools.py  |  Chapter 6: PyTorch Lightning version of the pipeline
=========================================================================
PyTorch Lightning removes the boilerplate you wrote by hand in train.py:
device moves, train()/eval() switching, gradient steps, mixed precision,
checkpoints, multi-GPU. You describe WHAT to do per batch; it handles HOW.

Contents:
  ImageFolderDataModule  (generic, any ImageFolder-style data)
  ImageClassifier        (takes ANY nn.Module)
  early_stopping         -> make_early_stopping
  run_training           -> make_trainer / run_training

When to choose which:
  plain loop (train.fit)  : small projects, full control, easy to debug
  Lightning               : multi-GPU, long runs, teams, many experiments

Requires: pip install lightning torchmetrics
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except ImportError as e:  # pragma: no cover
    raise ImportError("lightning_tools needs the `lightning` package: pip install lightning") from e

from torchmetrics.classification import Accuracy

from .models import freeze_batchnorm_stats, trainable_parameters


class ImageFolderDataModule(pl.LightningDataModule):
    """
    TEMPLATE: datasets + transforms + loaders in one reusable object.
    In plain words: tells Lightning where the images are and how to turn them
    into batches. Lightning calls setup() and the *_dataloader() methods itself.

    
    Change for your project: data_dir, batch_size, and PASS YOUR TRANSFORMS.
    Transforms are explicit arguments, not hidden globals, so the code
    can be reused.
    Expects: data_dir/train/<class>/..., data_dir/val/<class>/...
    """

    def __init__(
        self,
        data_dir: str,
        train_transform: Callable,
        val_transform: Callable,
        batch_size: int = 32,
        num_workers: int = 0,
        train_dir: str = "train",
        val_dir: str = "val",
        test_dir: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform, self.val_transform = train_transform, val_transform
        self.batch_size, self.num_workers = batch_size, num_workers
        self.train_dir, self.val_dir, self.test_dir = train_dir, val_dir, test_dir
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        import os

        if stage in (None, "fit"):
            self.train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, self.train_dir), self.train_transform)
            self.val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, self.val_dir), self.val_transform)
        if stage in (None, "test") and self.test_dir:
            self.test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, self.test_dir), self.val_transform)

    @property
    def class_names(self) -> List[str]:
        return (self.train_dataset or self.val_dataset).classes

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, False)


class ImageClassifier(pl.LightningModule):
    """
    TEMPLATE: wraps any classification network as a LightningModule.
    In plain words: says what to compute for one batch (training_step,
    validation_step) and which optimizer to use. Lightning runs the loops.

    It receives the network, so the same class serves a ResNet,
    a MobileNet (models.build_classifier) or your own ConvNet.
    Change for your project: the model you pass in, num_classes, lr,
    weight_decay, optional class_weights.
    freeze_bn=True keeps BatchNorm layers in eval mode while training. Use it
    with a frozen pretrained backbone, otherwise the "frozen" layers still
    change their running statistics every batch (see models.freeze_backbone).
    """

    def __init__(self, model: nn.Module, num_classes: int, lr: float = 1e-3, weight_decay: float = 1e-2,
                 class_weights: Optional[torch.Tensor] = None, scheduler_patience: int = 2, freeze_bn: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "class_weights"])  # logged with the run
        self.model = model
        self.register_buffer("class_weights", class_weights)  # moves to the right device automatically
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        if self.hparams.freeze_bn:
            freeze_batchnorm_stats(self.model)

    def _step(self, batch, acc_metric, prefix: str):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        acc_metric(logits, y)
        self.log_dict({f"{prefix}_loss": loss, f"{prefix}_acc": acc_metric}, prog_bar=True, on_epoch=True, on_step=(prefix == "train"))
        return loss

    def training_step(self, batch, batch_idx=None):
        return self._step(batch, self.train_acc, "train")

    def validation_step(self, batch, batch_idx=None):
        self._step(batch, self.val_acc, "val")

    def test_step(self, batch, batch_idx=None):
        self._step(batch, self.test_acc, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(trainable_parameters(self.model), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=self.hparams.scheduler_patience)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}


def make_early_stopping(monitor: str = "val_acc", mode: str = "max", patience: int = 5, stopping_threshold: Optional[float] = None) -> EarlyStopping:
    """
    In plain words: stop when the validation score stops improving for
    `patience` epochs, or as soon as it is good enough (stopping_threshold).
    
    """
    return EarlyStopping(monitor=monitor, mode=mode, patience=patience, stopping_threshold=stopping_threshold)


def make_trainer(
    max_epochs: int = 10,
    callbacks: Optional[Sequence] = None,
    mixed_precision: Optional[bool] = None,
    log_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    fast_dev_run: bool = False,
    progress_bar: bool = True,
    devices: int = 1,
) -> pl.Trainer:
    """
    In plain words: creates the object that runs everything: epochs, device,
    mixed precision, logging, checkpoints.

    
    fast_dev_run=True runs ONE batch of train+val and stops: use it first to
    catch bugs in seconds.
    For real work keep the logger, checkpointing and model summary on: log_dir gives you a CSV of
    metrics per epoch, checkpoint_dir keeps the best weights.
    """
    callbacks = list(callbacks or [])
    if checkpoint_dir:
        callbacks.append(ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_acc", mode="max", save_top_k=1, filename="best-{epoch}-{val_acc:.3f}"))
    if mixed_precision is None:
        mixed_precision = torch.cuda.is_available()
    logger = False
    if log_dir:
        from lightning.pytorch.loggers import CSVLogger

        logger = CSVLogger(log_dir, name="run")
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=devices,
        precision="16-mixed" if mixed_precision else "32-true",
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=progress_bar,
        enable_checkpointing=bool(checkpoint_dir),
        enable_model_summary=True,
        fast_dev_run=fast_dev_run,
    )


def run_training(model: pl.LightningModule, datamodule: pl.LightningDataModule, trainer: pl.Trainer):
    """ Returns (trainer, model)."""
    trainer.fit(model, datamodule)
    return trainer, model


# ADVANCED (production alternative): config-driven runs with LightningCLI
# What changes: no training script at all; the model, data and trainer are
#   described in a YAML file, and the command line overrides anything.
# Why teams do it: reproducible experiments, easy sweeps, one entry point.
#
#   # train.py
#   from lightning.pytorch.cli import LightningCLI
#   LightningCLI(ImageClassifier, ImageFolderDataModule)
#   # run: python train.py fit --config config.yaml --trainer.max_epochs=20
#
# Other production habits:
#   trainer.test(model, datamodule, ckpt_path="best")   # final number on the TEST set, once
#   trainer.predict(model, dataloaders)                 # batched inference
#   Trainer(devices=4, strategy="ddp")                  # multi-GPU, no code change
