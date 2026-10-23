"""
data.py  |  Chapters 2 and 3: Data
=========================================================================
Everything between "files on disk" and "batches ready for the model".

Contents (in the order you would use them in a project)
  1. Inspect and sanity-check the raw data
  2. Datasets: folder-based, CSV-based, triplet (pairs) + class filtering
  3. Transforms and normalization stats
  4. Splitting, DataLoaders, class weights


"""
from __future__ import annotations

import copy
import hashlib
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from .config import IMAGENET_MEAN, IMAGENET_STD

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# =========================================================================
# 1. INSPECT THE RAW DATA
# =========================================================================
def count_images_per_class(root: str, splits: Sequence[str] = ("train", "val"), verbose: bool = True) -> Dict[str, Dict[str, int]]:
    """
    In plain words: counts how many pictures each class has in each split,
    so you spot imbalance before you train.

    
    Change for your project: split folder names (train/val/test).
    Expects: root/<split>/<class_name>/<image files>
    """
    result: Dict[str, Dict[str, int]] = {}
    for split in splits:
        split_dir = Path(root) / split
        if not split_dir.is_dir():
            if verbose:
                print(f"[skip] {split_dir} not found")
            continue
        counts = {}
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            counts[class_dir.name] = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
        result[split] = counts
        if verbose:
            print(f"--- {split} (total {sum(counts.values())}) ---")
            for name, n in counts.items():
                print(f"  {name:<25} {n}")
    return result


def list_extensions(root: str) -> Dict[str, int]:
    """
    In plain words: tells you which file types are in the folder, so a stray
    .txt or .csv does not surprise you later.

    Returns counts per extension (easier to read than full path lists).
    """
    counts: Counter = Counter()
    for _, _, files in os.walk(root):
        for f in files:
            counts[os.path.splitext(f)[1].lower() or "<none>"] += 1
    return dict(counts)


def print_folder_tree(root: str, max_depth: int = 2) -> None:
    """
    In plain words: prints the folder layout like the `tree` command.

    Uses only the standard library (no extra package).
    """
    root_path = Path(root)
    print(root_path.name + "/")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = Path(dirpath).relative_to(root_path).parts
        if len(depth) >= max_depth:
            dirnames[:] = []
            continue
        dirnames.sort()
        indent = "  " * (len(depth) + 1)
        for d in dirnames:
            print(f"{indent}{d}/")
        shown = sorted(filenames)[:3]
        for f in shown:
            print(f"{indent}{f}")
        if len(filenames) > 3:
            print(f"{indent}... (+{len(filenames) - 3} files)")


def summarize_loader(loader: DataLoader) -> dict:
    """
    In plain words: peeks at the first batch and reports the shapes and value
    ranges. Run this before training: most bugs show up here.

    
    """
    images, labels = next(iter(loader))
    info = {
        "num_samples": len(loader.dataset),
        "num_batches": len(loader),
        "image_shape": tuple(images.shape),
        "image_dtype": str(images.dtype),
        "pixel_min": float(images.min()),
        "pixel_max": float(images.max()),
        "label_shape": tuple(labels.shape),
        "label_min": int(labels.min()),
        "label_max": int(labels.max()),
    }
    for k, v in info.items():
        print(f"{k:>12}: {v}")
    return info


def find_corrupt_images(root: str) -> List[str]:
    """
     data quality check.
    In plain words: opens every image and lists the ones that are broken,
    before they crash training at epoch 7.
    """
    bad = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                bad.append(str(p))
    return bad


def check_split_leakage(train_root: str, val_root: str) -> List[Tuple[str, str]]:
    """
     data quality check.
    In plain words: finds the exact same image sitting in both the train and
    validation folders. If it exists, your validation score is too good to
    be true.
    Limit: catches byte-identical files only, not resized or re-saved copies.
    """

    def digest(path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

    train_hashes = {digest(p): str(p) for p in Path(train_root).rglob("*") if p.suffix.lower() in IMG_EXTS}
    clashes = []
    for p in Path(val_root).rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            h = digest(p)
            if h in train_hashes:
                clashes.append((train_hashes[h], str(p)))
    return clashes


# =========================================================================
# 2. DATASETS
# =========================================================================
class TransformedDataset(Dataset):
    """
    In plain words: wraps any dataset (or split) and applies its own
    transform. This is how train gets augmentation and validation does not,
    even when both come from the same random split.

    
    Gotcha: the wrapped dataset must return raw PIL images (transform=None),
    otherwise the transform is applied twice.
    """

    def __init__(self, dataset: Dataset, transform: Optional[Callable] = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


SubsetWithTransform = TransformedDataset


def make_imagefolder_datasets(
    root: str,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    train_dir: str = "train",
    val_dir: str = "val",
    test_dir: Optional[str] = None,
    test_transform: Optional[Callable] = None,
) -> Dict[str, datasets.ImageFolder]:
    """
    TEMPLATE 1: folder-based datasets (the most common layout).
    In plain words: one folder per class, torchvision does the rest.

    
    Change for your project: the three folder names.
    Expects: root/train/<class>/img.jpg, root/val/<class>/img.jpg
    """
    out = {
        "train": datasets.ImageFolder(os.path.join(root, train_dir), transform=train_transform),
        "val": datasets.ImageFolder(os.path.join(root, val_dir), transform=val_transform),
    }
    if test_dir:
        out["test"] = datasets.ImageFolder(os.path.join(root, test_dir), transform=test_transform or val_transform)
    return out


class CsvImageDataset(Dataset):
    """
    TEMPLATE 2: images listed in a CSV (image path column + label column).
    In plain words: use this when the labels live in a spreadsheet instead of
    in folder names.

    
    Change for your project: csv_name, image_col, label_col, class_names_file.
    Expects: root_dir/<csv_name> with one row per image, labels as integers.

    It exposes `.targets` and `.classes` like torchvision datasets, so
    get_targets(), compute_class_weights() and TripletDataset work on it.
    """

    def __init__(
        self,
        root_dir: str,
        csv_name: str = "labels.csv",
        image_col: str = "image",
        label_col: str = "label",
        class_names_file: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        df = pd.read_csv(self.root_dir / csv_name)
        self.paths: List[Path] = [self.root_dir / p for p in df[image_col]]
        self.targets: List[int] = df[label_col].astype(int).tolist()
        if class_names_file:
            self.classes = (self.root_dir / class_names_file).read_text().splitlines()
        else:
            self.classes = [str(i) for i in sorted(set(self.targets))]

    def __len__(self) -> int:
        return len(self.targets)

    def get_label_description(self, label: int) -> str:
        return self.classes[label]

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            image = img.convert("RGB")  # always 3 channels, whatever the file has
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx]


# ADVANCED (production alternative): faster, safer CSV dataset
# What changes: validate the CSV once at start-up, and cache decoded images
#   (or use a faster decoder) so epochs do not re-read slow disks.
# Why teams do it: fail on a bad path in second 1, not hour 3; keep the GPU fed.
#
#   def _validate(self):
#       missing = [p for p in self.paths if not p.exists()]
#       if missing:
#           raise FileNotFoundError(f"{len(missing)} images missing, first: {missing[0]}")
#
#   # faster JPEG decode straight to a tensor (torchvision >= 0.10):
#   from torchvision.io import decode_image, ImageReadMode
#   img = decode_image(str(path), mode=ImageReadMode.RGB)   # uint8 tensor CxHxW
#   # and set DataLoader(num_workers>0, persistent_workers=True)


def get_targets(dataset: Dataset) -> List[int]:
    """
    In plain words: gets the list of labels from almost any dataset WITHOUT
    loading the images (fast), including splits made with random_split.

    
    """
    if isinstance(dataset, Subset):
        base = get_targets(dataset.dataset)
        return [base[i] for i in dataset.indices]
    if isinstance(dataset, TransformedDataset):
        return get_targets(dataset.dataset)
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            return [int(t) for t in getattr(dataset, attr)]
    if hasattr(dataset, "samples"):
        return [int(s[1]) for s in dataset.samples]
    # Slow fallback: loads every item
    return [int(y) for _, y in dataset]


def load_pil_image(dataset, index: int) -> Tuple[Image.Image, int]:
    """
    In plain words: fetch the ORIGINAL picture (not the normalized tensor) and
    its label from an ImageFolder-style dataset. Handy for showing search
    results and prediction examples.
    
    Works with Subset / TransformedDataset wrappers too.
    """
    if isinstance(dataset, Subset):
        return load_pil_image(dataset.dataset, dataset.indices[index])
    if isinstance(dataset, TransformedDataset):
        return load_pil_image(dataset.dataset, index)
    path, label = dataset.samples[index]
    return Image.open(path).convert("RGB"), int(label)


def get_class_names(dataset: Dataset) -> List[str]:
    """Finds `.classes` through Subset / TransformedDataset wrappers."""
    if isinstance(dataset, Subset):
        return get_class_names(dataset.dataset)
    if isinstance(dataset, TransformedDataset):
        return get_class_names(dataset.dataset)
    return list(dataset.classes)


class TripletDataset(Dataset):
    """
    TEMPLATE 3: (anchor, positive, negative) triplets for similarity learning.
    In plain words: for each picture, also hand back one picture of the same
    class (should be close) and one of a different class (should be far).

    Reads labels without loading images (so building it is fast), and
    avoids picking the anchor itself as its own positive.
    Change for your project: nothing, it wraps any labelled dataset.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.labels_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(get_targets(dataset)):
            self.labels_to_indices[label].append(idx)
        self.labels = sorted(self.labels_to_indices)
        if len(self.labels) < 2:
            raise ValueError("TripletDataset needs at least 2 classes")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        anchor, label = self.dataset[idx]
        candidates = self.labels_to_indices[label]
        pos_idx = idx
        if len(candidates) > 1:
            while pos_idx == idx:
                pos_idx = random.choice(candidates)
        neg_label = random.choice([lab for lab in self.labels if lab != label])
        neg_idx = random.choice(self.labels_to_indices[neg_label])
        positive, _ = self.dataset[pos_idx]
        negative, _ = self.dataset[neg_idx]
        return anchor, positive, negative


# ADVANCED (production alternative): hard-negative mining
# What changes: pick negatives that the model currently confuses with the
#   anchor, not random ones.
# Why teams do it: random negatives become too easy after a few epochs and
#   training stalls; hard negatives keep the embedding space improving.
#
#   # Every epoch: embed the whole train set, then for each anchor choose the
#   # closest embedding that has a DIFFERENT label (semi-hard: still farther
#   # than the positive). Libraries: pytorch-metric-learning (miners.TripletMarginMiner).


def filter_dataset_classes(dataset, target_classes: Sequence[str]):
    """
    In plain words: keeps only some classes of a bigger dataset and renumbers
    the labels 0..N-1, so you can train on a small, focused subset.

    
    Works with: ImageFolder-style (.samples) and CIFAR/MNIST-style (.data).
    Returns a copy; the original dataset object is left untouched
    (it never modifies it in place).
    """
    all_classes = list(dataset.classes)
    missing = [c for c in target_classes if c not in all_classes]
    if missing:
        raise ValueError(f"Classes not in dataset: {missing}")
    old_ids = [all_classes.index(c) for c in target_classes]
    remap = {old: new for new, old in enumerate(old_ids)}

    out = copy.copy(dataset)
    if hasattr(dataset, "samples"):  # ImageFolder family
        out.samples = [(p, remap[t]) for p, t in dataset.samples if t in remap]
        out.imgs = out.samples
        out.targets = [t for _, t in out.samples]
        out.class_to_idx = {c: i for i, c in enumerate(target_classes)}
    else:  # CIFAR / MNIST family
        targets = np.array(dataset.targets)
        keep = np.isin(targets, old_ids)
        out.data = dataset.data[keep]
        out.targets = [remap[int(t)] for t in targets[keep]]
    out.classes = list(target_classes)
    return out


# =========================================================================
# 3. TRANSFORMS
# =========================================================================
def build_transforms(
    image_size: Optional[int] = None,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    *,
    hflip: bool = True,
    vflip: bool = False,
    rotation: float = 0,
    color_jitter: float = 0.0,
    random_resized_crop: bool = False,
    translate: Optional[Tuple[float, float]] = None,
    saturation: float = 0.0,
    hue: float = 0.0,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    In plain words: builds two recipes for preparing images. The TRAIN recipe
    adds random changes (flip, rotate...) so the model sees more variety.
    The VALIDATION recipe only resizes and normalizes, because you want an
    honest, repeatable score.

    
    Change for your project: image_size, mean/std, and WHICH augmentations.
    color_jitter sets brightness AND contrast strength; saturation and hue are
    optional extras (for example saturation=0.2, hue=0.1).
    Gotcha: an augmentation must keep the label true. A vertical flip is fine
    for plant leaves, wrong for digits and for most X-rays.
    """
    mean, std = tuple(mean), tuple(std)
    train: list = []
    evaluation: list = []
    if image_size:
        if random_resized_crop:
            train.append(transforms.RandomResizedCrop((image_size, image_size)))
        else:
            train.append(transforms.Resize((image_size, image_size)))
        evaluation.append(transforms.Resize((image_size, image_size)))
    if hflip:
        train.append(transforms.RandomHorizontalFlip())
    if vflip:
        train.append(transforms.RandomVerticalFlip())
    if rotation:
        train.append(transforms.RandomRotation(rotation))
    if color_jitter or saturation or hue:
        train.append(transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=saturation, hue=hue))
    if translate:
        train.append(transforms.RandomAffine(degrees=0, translate=translate))
    finish = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(train + finish), transforms.Compose(evaluation + finish)


# ADVANCED (production alternative): transforms v2 and GPU-side augmentation
# What changes: use torchvision.transforms.v2 (faster, handles boxes/masks
#   too) and optionally augment on the GPU.
# Why teams do it: the same pipeline then works for detection/segmentation,
#   and CPU data loading stops being the bottleneck.
#
#   from torchvision.transforms import v2
#   train_tf = v2.Compose([
#       v2.ToImage(),                                   # PIL -> tensor image
#       v2.RandomResizedCrop(224, antialias=True),
#       v2.RandomHorizontalFlip(),
#       v2.ToDtype(torch.float32, scale=True),          # replaces ToTensor
#       v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#   ])
# MODERNIZE: transforms.ToTensor() still works but v2.ToImage()+v2.ToDtype is
#   the current recommendation. Also consider TrivialAugmentWide / RandAugment
#   instead of hand-picking flips and rotations.


def compute_mean_std(dataset: Dataset, image_size: int, batch_size: int = 64, num_workers: int = 0):
    """
    In plain words: measures the average colour and spread of YOUR images, so
    normalization fits your data. Skip this when you use a pretrained model:
    use the ImageNet numbers it was trained with.

    
    This version does one pass in batches. It uses E[x^2] - E[x]^2 with float64
    sums, which is accurate enough for images; a two-pass method is
    more numerically careful if you ever see a negative variance.
    Change for your project: image_size.
    Gotcha: compute it on the TRAIN split only. The dataset must return PIL
    images (transform=None).
    """
    tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    loader = DataLoader(TransformedDataset(dataset, tf), batch_size=batch_size, num_workers=num_workers)
    total = 0
    channel_sum = channel_sq = None
    for images, _ in loader:
        b, c, h, w = images.shape
        flat = images.double().reshape(b, c, -1)
        if channel_sum is None:
            channel_sum = torch.zeros(c, dtype=torch.float64)
            channel_sq = torch.zeros(c, dtype=torch.float64)
        channel_sum += flat.sum(dim=(0, 2))
        channel_sq += (flat**2).sum(dim=(0, 2))
        total += b * h * w
    mean = channel_sum / total
    std = (channel_sq / total - mean**2).clamp_min(0).sqrt()
    return mean.float(), std.float()


def denormalize(tensor: torch.Tensor, mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD) -> torch.Tensor:
    """
    In plain words: undoes normalization so you can SHOW a tensor as a
    normal-looking picture. Works for one image (C,H,W) or a batch (B,C,H,W).

    
    """
    m = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    s = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor * s + m).clamp(0, 1)


# =========================================================================
# 4. SPLITS, DATALOADERS, CLASS WEIGHTS
# =========================================================================
def split_dataset(dataset: Dataset, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    """
    In plain words: randomly cuts one dataset into train / validation / test.

    
    Takes a seed. An unseeded split differs on every
    run, so your validation images change and scores are not comparable.
    Returns: (train, val, test); test is None when test_frac == 0.
    """
    n = len(dataset)
    n_val, n_test = int(n * val_frac), int(n * test_frac)
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(seed)
    parts = random_split(dataset, [n_train, n_val, n_test] if n_test else [n_train, n_val], generator=gen)
    return (parts[0], parts[1], parts[2] if n_test else None)


# ADVANCED (production alternative): stratified split
# What changes: keep the class proportions the same in every split.
# Why teams do it: with rare classes, a plain random split can leave
#   validation with 2 examples of a class, so its score means nothing.
#
#   from sklearn.model_selection import train_test_split
#   targets = get_targets(dataset)
#   idx_train, idx_val = train_test_split(
#       range(len(dataset)), test_size=0.2, stratify=targets, random_state=42)
#   train_ds, val_ds = Subset(dataset, idx_train), Subset(dataset, idx_val)
#   # Also: if one object (patient, plant, product) has many photos, split BY
#   # that object (sklearn GroupShuffleSplit) so photos of it never land in
#   # both train and validation.


def seed_worker(worker_id: int) -> None:
    """Gives each DataLoader worker its own repeatable random seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
) -> Dict[str, DataLoader]:
    """
    In plain words: turns datasets into loaders that hand the model batches.
    Train is shuffled, validation and test are not.

    
    Change for your project: batch_size, num_workers.
    Tip: num_workers=0 is safest in notebooks; try 2 to 8 in scripts.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    gen = torch.Generator().manual_seed(seed)
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )
    loaders = {
        "train": DataLoader(train_ds, shuffle=True, generator=gen, **common),
        "val": DataLoader(val_ds, shuffle=False, **common),
    }
    if test_ds is not None:
        loaders["test"] = DataLoader(test_ds, shuffle=False, **common)
    return loaders


def make_split_loaders(
    dataset: Dataset,
    batch_size: int,
    val_frac: float,
    test_frac: float,
    train_transform: Callable,
    eval_transform: Callable,
    seed: int = 42,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    In plain words: the one-call version. Split one dataset, give train its
    augmenting transform and val/test the plain one, return the loaders.

     Use when you have ONE pool of labelled images
    (no ready-made train/val folders).
    Gotcha: `dataset` must be built with transform=None.
    """
    train, val, test = split_dataset(dataset, val_frac, test_frac, seed)
    train = TransformedDataset(train, train_transform)
    val = TransformedDataset(val, eval_transform)
    test = TransformedDataset(test, eval_transform) if test is not None else None
    return build_dataloaders(train, val, test, batch_size=batch_size, num_workers=num_workers, seed=seed)


# ADVANCED (production alternative): a Lightning DataModule
# What changes: datasets + transforms + loaders live in ONE class with
#   setup()/train_dataloader()/val_dataloader().
# Why teams do it: the same object plugs into Trainer, multi-GPU and
#   prediction without changes. See lightning_tools.ImageFolderDataModule.


def compute_class_weights(targets_or_dataset, num_classes: Optional[int] = None) -> torch.Tensor:
    """
    In plain words: rare classes get a bigger weight so the loss does not
    just learn to predict the common class. weight = total / (classes * count).

    
    Use: nn.CrossEntropyLoss(weight=weights.to(device))
    Gotcha: move the weights to the same device as the model, or you get a
    device-mismatch error.
    """
    targets = targets_or_dataset if isinstance(targets_or_dataset, (list, tuple, np.ndarray, torch.Tensor)) else get_targets(targets_or_dataset)
    t = torch.as_tensor(targets, dtype=torch.long)
    n_classes = num_classes or int(t.max()) + 1
    counts = torch.bincount(t, minlength=n_classes).float()
    weights = torch.zeros(n_classes)
    nonzero = counts > 0
    weights[nonzero] = len(t) / (n_classes * counts[nonzero])
    return weights


# ADVANCED (production alternative): fix imbalance in the sampler, not the loss
# What changes: oversample rare classes when building batches.
# Why teams do it: every batch then contains rare classes, which often trains
#   more steadily than loss weights alone. (Do not do both blindly: pick one.)
#
#   from torch.utils.data import WeightedRandomSampler
#   w = compute_class_weights(train_ds)
#   sample_w = [w[t] for t in get_targets(train_ds)]
#   sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
#   DataLoader(train_ds, batch_size=32, sampler=sampler)   # note: no shuffle=True
