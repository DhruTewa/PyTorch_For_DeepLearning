"""
viz.py  |  Plotting for every chapter
=========================================================================
Plotting helpers, with no hard-coded class names, mean/std or grid sizes:

  show_samples / show_samples_per_class   look at your data
  show_predictions                        look at what the model gets right/wrong
  plot_history                            training curves
  plot_confusion_matrix                   where the mistakes are
  plot_metric_scatter                     compare trials (accuracy vs size...)
  activation_grid / show_feature_maps /
  show_feature_strip / overlay_heatmap    interpretability pictures
  show_retrieval                          visual search results

Every function takes `show=True` (draws it, returns None) or `show=False`
(returns the Figure so you can save it: fig.savefig("plot.png", dpi=150)).
Mean/std are arguments, never hard-coded: pass the values your pipeline used.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import IMAGENET_MEAN, IMAGENET_STD
from .data import denormalize, get_class_names, get_targets


def _finish(fig, show: bool):
    fig.tight_layout()
    if show:
        plt.show()
        return None
    return fig


def _as_image(img, mean=None, std=None) -> np.ndarray:
    """Any image-like (PIL, numpy, tensor C,H,W) -> numpy H,W,C in [0,1] or H,W."""
    if isinstance(img, Image.Image):
        return np.asarray(img)
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu().float()
        if mean is not None and std is not None and t.ndim == 3 and len(mean) == t.shape[0]:
            t = denormalize(t, mean, std)
        else:  # no stats given: stretch to [0, 1] so it is at least visible
            lo, hi = t.min(), t.max()
            t = (t - lo) / (hi - lo + 1e-8)
        t = t.permute(1, 2, 0) if t.ndim == 3 else t
        return t.squeeze(-1).numpy() if t.ndim == 3 and t.shape[-1] == 1 else t.numpy()
    return np.asarray(img)


# =========================================================================
# LOOK AT THE DATA
# =========================================================================
def show_samples(dataset: Dataset, n: int = 8, cols: int = 4, class_names: Optional[Sequence[str]] = None,
                 mean=None, std=None, seed: Optional[int] = None, show: bool = True):
    """
    In plain words: shows a random handful of images with their labels. Do
    this before anything else: wrong labels and odd images show up instantly.

    
    Pass mean/std if the dataset applies Normalize, so colours look right.
    """
    rng = random.Random(seed)
    idxs = rng.sample(range(len(dataset)), min(n, len(dataset)))
    names = class_names or (get_class_names(dataset) if hasattr(dataset, "classes") else None)
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idxs):
        img, label = dataset[i]
        arr = _as_image(img, mean, std)
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
        ax.set_title(names[label] if names else str(label), fontsize=10)
    return _finish(fig, show)


def show_samples_per_class(dataset: Dataset, per_class: int = 2, class_names: Optional[Sequence[str]] = None,
                           mean=None, std=None, seed: Optional[int] = None, show: bool = True):
    """
    In plain words: one column per class, a few random images each. Good for
    seeing what separates the classes (and what makes them hard).
    
    """
    rng = random.Random(seed)
    targets = get_targets(dataset)
    names = class_names or get_class_names(dataset)
    by_class: Dict[int, List[int]] = {}
    for i, t in enumerate(targets):
        by_class.setdefault(t, []).append(i)
    classes = sorted(by_class)
    fig, axes = plt.subplots(per_class, len(classes), figsize=(2.6 * len(classes), 2.6 * per_class), squeeze=False)
    for c_pos, c in enumerate(classes):
        picks = rng.sample(by_class[c], min(per_class, len(by_class[c])))
        for r in range(per_class):
            ax = axes[r, c_pos]
            ax.axis("off")
            if r < len(picks):
                img, _ = dataset[picks[r]]
                arr = _as_image(img, mean, std)
                ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
            if r == 0:
                ax.set_title(names[c], fontsize=10)
    return _finish(fig, show)


@torch.no_grad()
def show_predictions(model: torch.nn.Module, dataset: Dataset, device: torch.device, class_names: Sequence[str], n: int = 6,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, only_wrong: bool = False, seed: Optional[int] = None, show: bool = True):
    """
    In plain words: shows images with "True" and "Pred" labels. Green = right,
    red = wrong. With only_wrong=True it hunts for mistakes, which is
    usually the more useful view.

    """
    model.eval()
    rng = random.Random(seed)
    order = list(range(len(dataset)))
    rng.shuffle(order)
    picked = []  # (image_tensor, true, pred)
    for i in order:
        img, label = dataset[i]
        pred = int(model(img.unsqueeze(0).to(device)).argmax(1))
        if only_wrong and pred == label:
            continue
        picked.append((img, label, pred))
        if len(picked) == n:
            break
    if not picked:
        print("Nothing to show (no mistakes found)." if only_wrong else "Empty dataset.")
        return None
    cols = min(3, len(picked))
    rows = int(np.ceil(len(picked) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (img, t, p) in zip(axes.ravel(), picked):
        arr = _as_image(img, mean, std)
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
        ax.set_title(f"True: {class_names[t]}\nPred: {class_names[p]}", color="green" if t == p else "red", fontsize=10)
    return _finish(fig, show)


# =========================================================================
# TRAINING AND EVALUATION PLOTS
# =========================================================================
def plot_history(history: Dict[str, Sequence[float]], show: bool = True):
    """
    In plain words: the two charts you read after every run.
      Left: loss. If train loss keeps falling but validation loss turns up,
            the model is memorizing (overfitting).
      Right: accuracy.
    
    Works with the dict returned by train.fit / train.simple_fit.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, label in (("train_loss", "Train loss"), ("val_loss", "Validation loss")):
        if key in history and len(history[key]):
            ax1.plot(range(1, len(history[key]) + 1), history[key], marker="o", label=label)
    ax1.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    for key, label in (("train_acc", "Train accuracy"), ("val_acc", "Validation accuracy")):
        if key in history and len(history[key]):
            ax2.plot(range(1, len(history[key]) + 1), history[key], marker="o", label=label)
    ax2.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)
    return _finish(fig, show)


def plot_confusion_matrix(cm, class_names: Sequence[str], normalize: bool = False, show: bool = True):
    """
    In plain words: a heat-map of mistakes. Row = what it really was, column =
    what the model said. Bright diagonal = good. Bright spot off the diagonal
    = a specific confusion to investigate.
    
    normalize=True shows row percentages (better when classes differ in size).
    """
    cm = np.asarray(cm, dtype=float)
    data = cm / cm.sum(axis=1, keepdims=True).clip(1) if normalize else cm
    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 3, 1.0 * len(class_names) + 2.5))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix")
    thresh = data.max() / 2 if data.size else 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            txt = f"{data[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", color="white" if data[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    return _finish(fig, show)


def plot_metric_scatter(df, x: str = "accuracy", y: str = "model_size", color: Optional[str] = None, label: str = "trial", show: bool = True):
    """
    In plain words: one dot per experiment. Use it to see trade-offs, e.g.
    accuracy vs model size, so you pick the smallest model that is good enough.
    
    `df` is a pandas DataFrame with the columns you name.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sc = ax.scatter(df[x], df[y], c=df[color] if color else None, cmap="viridis", s=90, edgecolor="k", alpha=0.85)
    if color:
        fig.colorbar(sc, ax=ax).set_label(color)
    if label in df:
        for _, row in df.iterrows():
            ax.annotate(str(row[label]), (row[x], row[y]), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set(xlabel=x, ylabel=y, title=f"{y} vs {x}")
    ax.grid(alpha=0.3)
    return _finish(fig, show)


# =========================================================================
# INTERPRETABILITY PICTURES (use with interpret.py)
# =========================================================================
def activation_grid(feat: torch.Tensor, num_filters: int = 4) -> np.ndarray:
    """
    In plain words: tiles a few channels of one layer's feature map into a
    single image. Channels are spread evenly through the layer.
     `feat` has shape (1, C, H, W).
    """
    f = feat.squeeze(0).detach().cpu().numpy()
    C, H, W = f.shape
    picks = np.linspace(0, C - 1, num_filters, dtype=int)
    g = int(np.ceil(np.sqrt(num_filters)))
    grid = np.zeros((g * H, g * W))
    for k, c in enumerate(picks):
        r, col = divmod(k, g)
        grid[r * H : (r + 1) * H, col * W : (col + 1) * W] = f[c]
    grid -= grid.min()
    return grid / (grid.max() + 1e-5)


def show_feature_maps(image: torch.Tensor, activations: Dict[str, torch.Tensor], mean=IMAGENET_MEAN, std=IMAGENET_STD,
                      num_filters: int = 4, show: bool = True):
    """
    In plain words: the original image next to what each chosen layer "sees".
    Early layers pick up edges and colours; deeper layers pick up shapes and
    object parts.
     `image` is the normalized (1,C,H,W)
    input you fed to capture_activations.
    """
    names = list(activations)
    cols = min(3, len(names) + 1)
    rows = int(np.ceil((len(names) + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    axes.ravel()[0].imshow(_as_image(image[0], mean, std))
    axes.ravel()[0].set_title("Original")
    for ax, name in zip(axes.ravel()[1:], names):
        ax.imshow(activation_grid(activations[name], num_filters), cmap="viridis")
        ax.set_title(f"{name}: {activations[name].shape[1]} channels")
    return _finish(fig, show)


def show_feature_strip(maps: Sequence[torch.Tensor], names: Optional[Sequence[str]] = None, show: bool = True):
    """
    In plain words: one row of heat-maps showing the most active channel of
    each layer, from shallow to deep.
    
    """
    fig, axes = plt.subplots(1, len(maps), figsize=(3 * len(maps), 3.2), squeeze=False)
    for k, (ax, m) in enumerate(zip(axes[0], maps)):
        ax.imshow(m.squeeze().detach().cpu().numpy(), cmap="viridis")
        ax.axis("off")
        if names:
            ax.set_title(names[k])
    return _finish(fig, show)


def overlay_heatmap(image: torch.Tensor, heatmap: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD, alpha: float = 0.5,
                    title: str = "", show: bool = True):
    """
    In plain words: lays a coloured "importance" map on top of the picture.
    Red = the model paid a lot of attention here.
    
    `image` is the normalized (1,C,H,W) or (C,H,W) tensor; `heatmap` is (H,W).
    """
    img = image[0] if image.ndim == 4 else image
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(_as_image(img, mean, std))
    ax.imshow(heatmap.detach().cpu().numpy(), cmap="jet", alpha=alpha)
    ax.axis("off")
    ax.set_title(title)
    return _finish(fig, show)


# =========================================================================
# VISUAL SEARCH
# =========================================================================
def show_retrieval(query: Image.Image, results: Sequence[Image.Image], titles: Optional[Sequence[str]] = None, show: bool = True):
    """
    In plain words: the query picture on the left, its nearest matches next to it.
    
    """
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(3 * (len(results) + 1), 3.4), squeeze=False)
    axes[0][0].imshow(query)
    axes[0][0].set_title("Query")
    for k, (ax, im) in enumerate(zip(axes[0][1:], results)):
        ax.imshow(im)
        ax.set_title(titles[k] if titles else f"Match {k + 1}", fontsize=9)
    for ax in axes[0]:
        ax.axis("off")
    return _finish(fig, show)
