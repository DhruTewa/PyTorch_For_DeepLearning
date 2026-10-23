"""
config.py  |  Chapter 0: Project skeleton
=========================================================================
The boring-but-essential things every project needs BEFORE any model code:
one place for settings, reproducible randomness, picking a device, logging.

This file is the "things you should be aware of" layer that real projects
add on top of a model.

Tag legend used across the whole toolkit
----------------------------------------
  In plain words : one or two lines anyone can follow
  Change for your project : what you edit when you reuse it
  ADVANCED       : commented production alternative (not executed)
  MODERNIZE      : an older common approach is dated; here is the newer way
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    """
    In plain words: one object that holds every setting of a run, so you
    change numbers in one place and can save them next to the results.

    Change for your project: add/remove fields. Keep it flat and simple.
    """

    seed: int = 42
    image_size: int = 224
    num_classes: int = 2
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD
    data_dir: str = "data"
    out_dir: str = "runs/exp1"
    amp: bool = False  # mixed precision, see train.fit
    grad_clip: Optional[float] = None  # e.g. 1.0

    def save(self, path: Optional[str] = None) -> Path:
        """Write the config to JSON so a finished run can be reproduced."""
        path = Path(path or Path(self.out_dir) / "config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        data = json.loads(Path(path).read_text())
        # JSON turns tuples into lists; turn them back
        for key in ("mean", "std"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


# ADVANCED (production alternative)
# What changes: settings come from a YAML file plus command-line overrides,
#   and are validated (wrong types fail early).
# Why teams do it: you can launch 20 experiments from a shell script without
#   editing Python, and every run logs the exact config it used.
#
#   # pip install hydra-core    (or: pydantic-settings)
#   import hydra
#   from omegaconf import DictConfig
#
#   @hydra.main(config_path="conf", config_name="train", version_base=None)
#   def main(cfg: DictConfig):
#       print(cfg.lr, cfg.batch_size)
#
#   # run:  python train.py lr=3e-4 batch_size=64


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """
    In plain words: makes random choices repeatable, so two runs with the
    same settings give (nearly) the same result.

    
    Gotcha: GPU runs can still differ slightly. deterministic=True forces
    repeatable GPU kernels but can slow training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Raises an error if an op has no deterministic version
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(prefer: str = "auto") -> torch.device:
    """
    In plain words: picks the fastest hardware you have (GPU, Apple chip,
    or plain CPU) unless you ask for a specific one.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(out_dir: Optional[str] = None, name: str = "cv", level=logging.INFO) -> logging.Logger:
    """
    In plain words: prints progress AND saves it to a file, so you can read
    what happened after the run is over. Use logger.info(...) not print(...).

    MODERNIZE: tutorials often use print() everywhere. That is fine for learning
    but gives you no history once the notebook is closed.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(out_dir) / "run.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ADVANCED (production alternative): experiment tracking
# What changes: metrics go to a tracking tool instead of only a log file.
# Why teams do it: compare 50 runs in a dashboard, see which settings won.
#
#   # Option 1: TensorBoard (simple, local)
#   from torch.utils.tensorboard import SummaryWriter
#   writer = SummaryWriter(log_dir=cfg.out_dir)
#   writer.add_scalar("val/acc", val_acc, epoch)
#
#   # Option 2: MLflow (self-hosted, model registry)
#   import mlflow
#   mlflow.log_params(asdict(cfg)); mlflow.log_metric("val_acc", val_acc, step=epoch)
#
#   # Option 3: Weights & Biases (hosted)
#   import wandb; wandb.init(project="my-cv", config=asdict(cfg)); wandb.log({"val_acc": val_acc})
