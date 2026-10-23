"""
tuning.py  |  Chapter 9: Hyperparameter tuning with Optuna
=========================================================================
Let the computer try many settings (layers, filters, learning rate, image
size...) and keep the best, instead of guessing by hand.

Contents: design_search_space, objective_function, trainable-parameter and
efficiency metrics. Data loaders, the training epoch and evaluation live in
data.py, train.py and evaluate.py.

Vocabulary
  trial    one attempt with one set of settings
  study    the collection of all trials
  search space   the allowed range for each setting
  objective      a function: settings in, score out (higher or lower = better)

Optuna is imported lazily so the rest of the toolkit works without it.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from .evaluate import evaluate
from .models import ConvNet, count_parameters
from .train import train_one_epoch
from .optimize import model_size_mb

# (low, high, step) for ints, (low, high) for floats, list for categorical.
# Sensible defaults for a small CNN search.
DEFAULT_SPACE = {
    "n_layers": (1, 3),
    "n_filters": (8, 64, 8),
    "kernel_size": (3, 5, 2),
    "dropout_rate": (0.1, 0.5),
    "fc_size": (64, 512, 64),
    "learning_rate": (1e-4, 1e-2),  # sampled on a log scale
    "resolution": [16, 32, 64],
    "batch_size": [8, 16],
}


def suggest_cnn_params(trial, space: Optional[dict] = None) -> dict:
    """
    In plain words: asks Optuna to pick one value for every setting, within
    the ranges you allow.

    Setting names are stable, so a saved study can be loaded and rebuilt.
    Change for your project: the `space` dict (ranges), or add new settings.
    Gotcha: learning rates must be sampled on a LOG scale (log=True). Sampling
    them linearly almost never tries the small values that matter.
    """
    s = {**DEFAULT_SPACE, **(space or {})}
    n_layers = trial.suggest_int("n_layers", s["n_layers"][0], s["n_layers"][1])
    f_lo, f_hi, f_step = s["n_filters"]
    k_lo, k_hi, k_step = s["kernel_size"]
    d_lo, d_hi = s["dropout_rate"]
    fc_lo, fc_hi, fc_step = s["fc_size"]
    lr_lo, lr_hi = s["learning_rate"]
    return {
        "n_layers": n_layers,
        "n_filters": [trial.suggest_int(f"n_filters_layer{i}", f_lo, f_hi, step=f_step) for i in range(n_layers)],
        "kernel_sizes": [trial.suggest_int(f"kernel_size_layer{i}", k_lo, k_hi, step=k_step) for i in range(n_layers)],
        "dropout_rate": trial.suggest_float("dropout_rate", d_lo, d_hi),
        "fc_size": trial.suggest_int("fc_size", fc_lo, fc_hi, step=fc_step),
        "learning_rate": trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True),
        "resolution": trial.suggest_categorical("resolution", s["resolution"]),
        "batch_size": trial.suggest_categorical("batch_size", s["batch_size"]),
    }


def unflatten_params(flat: dict) -> dict:
    """
    In plain words: Optuna stores settings flat ("n_filters_layer0": 16, ...).
    This turns them back into the lists the model builder wants.
    
    """
    n = flat["n_layers"]
    out = dict(flat)
    out["n_filters"] = [flat[f"n_filters_layer{i}"] for i in range(n)]
    out["kernel_sizes"] = [flat[f"kernel_size_layer{i}"] for i in range(n)]
    return out


def build_model_from_params(params: dict, num_classes: int, in_channels: int = 3) -> ConvNet:
    """Builds the CNN described by a settings dict (see suggest_cnn_params)."""
    return ConvNet(
        num_classes=num_classes,
        in_channels=in_channels,
        conv_channels=params["n_filters"],
        kernel_sizes=params["kernel_sizes"],
        fc_size=params["fc_size"],
        dropout=params["dropout_rate"],
        image_size=params["resolution"],
    )


def make_objective(
    make_loaders: Callable[[dict], Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
    num_classes: int,
    device: torch.device,
    epochs: int = 4,
    space: Optional[dict] = None,
    in_channels: int = 3,
) -> Callable:
    """
    TEMPLATE: builds the objective function Optuna calls once per trial.

    YOU pass `make_loaders`, a
    function that receives the trial's settings and returns
    (train_loader, val_loader). That keeps the tuning code independent of
    your folder layout.

        def make_loaders(params):          # params["resolution"], params["batch_size"]
            tf = transforms.Compose([transforms.Resize((params["resolution"],) * 2), transforms.ToTensor()])
            ...
            return train_loader, val_loader

    The score returned is VALIDATION accuracy after `epochs` epochs.
    It also reports each epoch so a pruner can stop bad trials early.
    Gotcha: tune on the validation set, never the test set.
    """
    import optuna  # lazy import

    def objective(trial) -> float:
        params = suggest_cnn_params(trial, space)
        train_loader, val_loader = make_loaders(params)
        model = build_model_from_params(params, num_classes, in_channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        loss_fn = nn.CrossEntropyLoss()
        acc = 0.0
        for epoch in range(epochs):
            train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            acc = evaluate(model, val_loader, device)["acc"]
            trial.report(acc, epoch)  # tell Optuna the progress so far
            if trial.should_prune():  # a pruner may say: this trial is hopeless
                raise optuna.TrialPruned()
        return acc

    return objective


def create_or_load_study(name: str = "cv_study", storage: Optional[str] = None, direction: str = "maximize", seed: int = 42):
    """
    In plain words: starts a new study, or continues an existing one saved in
    a database file, so you can stop and resume tuning.
    
    storage example: "sqlite:///tuning.db"  (None = in memory, lost on exit)
    The default pruner (MedianPruner) stops trials that are doing worse than
    the median of earlier trials at the same epoch.
    """
    import optuna

    return optuna.create_study(
        study_name=name,
        storage=storage,
        direction=direction,
        load_if_exists=storage is not None,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )


def top_trials(study, k: int = 5):
    """
    In plain words: the k best finished trials as a table (settings + score).
    
    """
    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"].sort_values("value", ascending=False).head(k)
    return df.set_index("number", drop=False)


def efficiency_table(study, num_classes: int, k: int = 5, in_channels: int = 3):
    """
    In plain words: for the best trials, how accurate AND how big is each
    model? Often the 2nd-best model is 10x smaller and is the better choice.
    
    Returns a DataFrame: trial, accuracy, model_size (parameters), size_mb.
    """
    import pandas as pd

    rows = []
    for trial in study.get_trials(deepcopy=True):
        if trial.state.name != "COMPLETE":
            continue
        model = build_model_from_params(unflatten_params(trial.params), num_classes, in_channels)
        rows.append({"trial": trial.number, "accuracy": trial.value, "model_size": count_parameters(model), "size_mb": model_size_mb(model)})
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).head(k)
    return df.reset_index(drop=True)


# ADVANCED (production alternative): optimise accuracy AND size at the same time
# What changes: a multi-objective study returns a set of best trade-offs
#   (the "Pareto front"), not one winner.
# Why teams do it: in production you almost always have a size/latency budget.
#
#   study = optuna.create_study(directions=["maximize", "minimize"])
#   def objective(trial):
#       params = suggest_cnn_params(trial)
#       ... train ...
#       return val_acc, count_parameters(model)
#   study.optimize(objective, n_trials=50)
#   study.best_trials          # the Pareto-optimal trials
#
# Also useful:
#   study.optimize(objective, n_trials=100, n_jobs=4)    # parallel trials (CPU-bound)
#   optuna.visualization.plot_param_importances(study)   # which settings mattered?
#   optuna.visualization.plot_optimization_history(study)
# For big jobs: Optuna + a shared database on a server, or Ray Tune.
