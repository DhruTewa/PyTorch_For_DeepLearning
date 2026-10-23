"""
evaluate.py  |  Chapters 10 and 11: Evaluation, and embeddings / visual search
=========================================================================
  1. evaluate()            one number for loss and accuracy
  2. predictions + confusion matrix + per-class metrics (no sklearn needed)
  3. error analysis        (beyond accuracy)
  4. predicting one image
  5. embeddings + nearest-neighbour search (visual search)

All metric functions work on plain tensors, so you can test them without a
model. Accuracy everywhere is a FRACTION from 0 to 1 (never a percentage).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# =========================================================================
# 1. EVALUATE
# =========================================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: Optional[Callable] = None) -> Dict[str, float]:
    """
    In plain words: runs the model over a dataset WITHOUT learning and reports
    how well it did.

    
    Returns: {"acc": ..., "loss": ...} (loss only if you pass loss_fn).
    Gotchas: model.eval() turns off dropout and freezes BatchNorm statistics;
    no_grad saves memory. Forgetting either gives wrong or noisy scores.
    Calling torch.cuda.empty_cache() every batch only slows things down, so
    it is left out.
    """
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        if loss_fn is not None:
            loss_sum += loss_fn(outputs, targets).item() * targets.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)
    result = {"acc": correct / max(total, 1)}
    if loss_fn is not None:
        result["loss"] = loss_sum / max(total, 1)
    return result


# =========================================================================
# 2. PREDICTIONS, CONFUSION MATRIX, PER-CLASS METRICS
# =========================================================================
@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    In plain words: collects the model's answer for every image in a loader.
    Returns CPU tensors: (predicted_class, true_class, probabilities).
    
    """
    model.eval()
    preds, labels, probs = [], [], []
    for inputs, targets in loader:
        logits = model(inputs.to(device))
        p = logits.softmax(dim=1).cpu()
        probs.append(p)
        preds.append(p.argmax(1))
        labels.append(targets.cpu())
    return torch.cat(preds), torch.cat(labels), torch.cat(probs)


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    In plain words: a table. Row = true class, column = what the model said.
    The diagonal is correct answers; everything else is a specific mistake.
    
    """
    idx = labels.long() * num_classes + preds.long()
    return torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def per_class_accuracy(cm: torch.Tensor, class_names: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """
    In plain words: for each class, what share of its images were recognised?
    (This is also called per-class recall.) Spots a model that ignores a
    small class while the overall accuracy still looks fine.
    
    """
    totals = cm.sum(dim=1).clamp_min(1)
    accs = cm.diag().float() / totals
    names = class_names or [str(i) for i in range(len(accs))]
    return {n: float(a) for n, a in zip(names, accs)}


def precision_recall_f1(cm: torch.Tensor, average: str = "macro") -> Dict[str, float]:
    """
    In plain words:
      precision = when the model says "class X", how often is it right?
      recall    = of all real X, how many did it find?
      f1        = one number balancing the two.
    "macro" = average over classes with equal weight (fair to small classes).
    
    Zero-division cases count as 0.
    """
    cm = cm.float()
    tp = cm.diag()
    precision = tp / cm.sum(dim=0).clamp_min(1)
    recall = tp / cm.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    if average == "macro":
        return {"precision": precision.mean().item(), "recall": recall.mean().item(), "f1": f1.mean().item()}
    if average == "micro":
        acc = (tp.sum() / cm.sum()).item()
        return {"precision": acc, "recall": acc, "f1": acc}
    raise ValueError("average must be 'macro' or 'micro'")


def classification_report(model: nn.Module, loader: DataLoader, device: torch.device, class_names: Optional[Sequence[str]] = None, verbose: bool = True) -> dict:
    """
    In plain words: the one-call evaluation report for a classifier: accuracy,
    per-class accuracy, precision/recall/F1 and the confusion matrix.
    """
    preds, labels, probs = predict_loader(model, loader, device)
    num_classes = probs.shape[1]
    cm = confusion_matrix(preds, labels, num_classes)
    report = {
        "acc": (preds == labels).float().mean().item(),
        "per_class_acc": per_class_accuracy(cm, class_names),
        **precision_recall_f1(cm),
        "confusion_matrix": cm,
        "preds": preds,
        "labels": labels,
        "probs": probs,
    }
    if verbose:
        print(f"accuracy {report['acc']:.4f} | macro precision {report['precision']:.4f} | recall {report['recall']:.4f} | f1 {report['f1']:.4f}")
        for name, a in report["per_class_acc"].items():
            print(f"  {name:<25} {a:.4f}")
    return report


# ADVANCED (production alternative): torchmetrics and sklearn
# What changes: let a library compute metrics, and update them batch by batch.
# Why teams do it: handles multi-GPU syncing, many metric types, edge cases.
#
#   import torchmetrics
#   metrics = torchmetrics.MetricCollection({
#       "acc": torchmetrics.Accuracy(task="multiclass", num_classes=N, average="macro"),
#       "prec": torchmetrics.Precision(task="multiclass", num_classes=N, average="macro"),
#       "rec": torchmetrics.Recall(task="multiclass", num_classes=N, average="macro"),
#   }).to(device)
#   metrics.update(preds, labels)  # inside the loop
#   metrics.compute()              # after the loop
#
#   from sklearn.metrics import classification_report
#   print(classification_report(labels, preds, target_names=class_names))
# NOTE: "macro average accuracy" (mean of per-class recall)
#   is not plain accuracy, and differs from plain accuracy when
#   classes are imbalanced.


# =========================================================================
# 3. ERROR ANALYSIS  
# =========================================================================
def misclassified_indices(preds: torch.Tensor, labels: torch.Tensor) -> List[int]:
    """Positions of every wrong answer. Open those images and LOOK at them."""
    return (preds != labels).nonzero(as_tuple=True)[0].tolist()


def most_confused_pairs(cm: torch.Tensor, class_names: Optional[Sequence[str]] = None, top: int = 5) -> List[Tuple[str, str, int]]:
    """
    In plain words: which two classes does the model mix up the most?
    Returns (true_class, predicted_class, count), biggest first.
    """
    cm = cm.clone()
    cm.fill_diagonal_(0)
    names = class_names or [str(i) for i in range(cm.shape[0])]
    flat = cm.flatten()
    values, idx = flat.topk(min(top, flat.numel()))
    n = cm.shape[0]
    return [(names[i // n], names[i % n], int(v)) for v, i in zip(values, idx) if v > 0]


def threshold_sweep(pos_probs: torch.Tensor, labels: torch.Tensor, thresholds: Optional[Sequence[float]] = None) -> List[Dict[str, float]]:
    """
    In plain words (binary tasks): the model outputs a probability; where you
    cut it (0.5? 0.3?) trades missed cases against false alarms. This table
    shows precision and recall at each cut so YOU choose, based on which
    mistake costs more (e.g. missing pneumonia is worse than a false alarm).
    """
    thresholds = thresholds if thresholds is not None else [i / 10 for i in range(1, 10)]
    rows = []
    for t in thresholds:
        pred = (pos_probs >= t).long()
        tp = ((pred == 1) & (labels == 1)).sum().item()
        fp = ((pred == 1) & (labels == 0)).sum().item()
        fn = ((pred == 0) & (labels == 1)).sum().item()
        rows.append({"threshold": t, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)})
    return rows


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    """
    In plain words: when the model says "90% sure", is it right 90% of the
    time? Lower is better. A model can be accurate and still overconfident.
    """
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece = torch.zeros(1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.float().mean() * (correct[mask].mean() - conf[mask].mean()).abs()
    return float(ece)


def subgroup_accuracy(preds: torch.Tensor, labels: torch.Tensor, groups: Sequence) -> Dict[str, float]:
    """
    In plain words: accuracy for each slice of your data (camera type,
    hospital, lighting, season...). An average can hide one slice failing.
    `groups` has one group label per image, in the same order as preds.
    """
    groups = np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        mask = torch.as_tensor(groups == g)
        out[str(g)] = (preds[mask] == labels[mask]).float().mean().item()
    return out


# =========================================================================
# 4. PREDICT ONE IMAGE
# =========================================================================
@torch.no_grad()
def predict_image(
    model: nn.Module,
    image: Union[str, Path, Image.Image],
    transform: Callable,
    device: torch.device,
    class_names: Optional[Sequence[str]] = None,
    topk: int = 1,
) -> List[Tuple[Union[str, int], float]]:
    """
    In plain words: give it one picture (a file or a PIL image), get back the
    top answers with their probabilities.

    
    Gotcha: `transform` MUST be the validation transform you trained with
    (same size, same mean/std). Hard-coding them inside a helper
    lets them drift apart from training; pass your own.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    probs = model(x).softmax(dim=1)[0]
    p, idx = probs.topk(min(topk, probs.numel()))
    return [((class_names[i] if class_names else int(i)), float(v)) for v, i in zip(p.cpu(), idx.cpu())]


# =========================================================================
# 5. EMBEDDINGS AND VISUAL SEARCH
# =========================================================================
@torch.no_grad()
def extract_embeddings(encoder: nn.Module, dataset: Dataset, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """
    In plain words: turns every image of a dataset into its embedding vector
    (the list of numbers that describes it). Do this once, then search fast.
    
    `encoder` is any module that maps a batch of images to (B, D).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    encoder.eval()
    out = [encoder(images.to(device)).cpu().numpy() for images, _ in loader]
    return np.concatenate(out, axis=0)


@torch.no_grad()
def embed_image(encoder: nn.Module, image: Image.Image, transform: Callable, device: torch.device) -> np.ndarray:
    """ Returns shape (1, D)."""
    encoder.eval()
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    return encoder(x).cpu().numpy()


def find_nearest(embeddings: np.ndarray, query: np.ndarray, k: int = 5, exclude_self: bool = False, metric: str = "l2") -> Tuple[np.ndarray, np.ndarray]:
    """
    In plain words: finds the k stored images whose embeddings are closest to
    the query. Returns (indices, distances).

    Dropping the first hit is
    right ONLY when the query image is itself in the gallery. If you search with
    a brand-new photo, dropping the first hit throws away your best match, so
    it is now a switch: exclude_self=True for "query is in the gallery".
    metric: "l2" or "cosine".
    """
    q = query.reshape(1, -1)
    if metric == "cosine":
        a = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
        b = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
        dist = 1.0 - (a @ b.T).ravel()
    elif metric == "l2":
        dist = np.linalg.norm(embeddings - q, axis=1)
    else:
        raise ValueError("metric must be 'l2' or 'cosine'")
    order = np.argsort(dist)
    start = 1 if exclude_self else 0
    sel = order[start : start + k]
    return sel, dist[sel]


# ADVANCED (production alternative): approximate nearest-neighbour index
# What changes: build an index once; search millions of vectors in milliseconds.
# Why teams do it: the brute-force distance above is O(N) per query and
#   keeps every vector in memory as a NumPy array.
#
#   # pip install faiss-cpu
#   import faiss
#   emb = embeddings.astype("float32"); faiss.normalize_L2(emb)   # cosine via inner product
#   index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
#   scores, ids = index.search(query_normalized.astype("float32"), k)
#   # for >1M vectors: faiss.IndexIVFFlat / IndexHNSWFlat (approximate, much faster)
