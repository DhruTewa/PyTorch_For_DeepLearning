"""
models.py  |  Chapters 4 and 5: Models and Transfer learning
=========================================================================
Building blocks:
  1. MLP
  2. ConvNet        (configurable CNN)
  3. Efficient CNN  (inverted residual / MobileNet-style)
  4. Pretrained     (MobileNetV3, ResNet18, ResNet50, any torchvision name)
  5. Embedding nets (Siamese)
  6. Small utilities (count parameters, freeze, summary)
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torchvision


# =========================================================================
# 1. MLP
# =========================================================================
def build_mlp(
    in_features: int,
    out_features: int,
    hidden: Sequence[int] = (64, 32),
    flatten_input: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    In plain words: a plain stack of Linear + ReLU layers. Fine for tables of
    numbers, and a quick baseline for small images.

    
    Change for your project: in_features, out_features, hidden sizes.
    Use out_features=1 + MSELoss for regression, out_features=num_classes +
    CrossEntropyLoss for classification (no softmax layer: the loss does it).
    """
    layers: list = [nn.Flatten()] if flatten_input else []
    prev = in_features
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_features))
    return nn.Sequential(*layers)


# =========================================================================
# 2. CONVNET
# =========================================================================
class ConvBlock(nn.Module):
    """
    In plain words: the standard CNN building brick. Look for patterns (Conv),
    keep training stable (BatchNorm), add flexibility (ReLU), shrink the
    picture by half (MaxPool).

    
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvNet(nn.Module):
    """
    TEMPLATE: a configurable CNN classifier.
    In plain words: a few ConvBlocks to find patterns, then a small
    fully-connected head to decide the class.

     One class covers both.
    Change for your project: conv_channels, kernel_sizes, fc_size, dropout,
    image_size (must match the size of the images you feed in).

    Note: a lazily-built classifier would create its classifier
    lazily inside forward(). That forces you to run a dummy batch BEFORE
    making the optimizer, and loading a saved state_dict fails until you do.
    Here the flattened size is computed once in __init__, so the model is
    complete the moment it exists.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        conv_channels: Sequence[int] = (32, 64, 128),
        kernel_sizes: Union[int, Sequence[int]] = 3,
        fc_size: int = 512,
        dropout: float = 0.5,
        image_size: int = 32,
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(conv_channels)
        blocks, prev = [], in_channels
        for out_c, k in zip(conv_channels, kernel_sizes):
            blocks.append(ConvBlock(prev, out_c, k))
            prev = out_c
        self.features = nn.Sequential(*blocks)

        # Work out the flattened size with a dummy image (eval mode so
        # BatchNorm does not complain about a batch of one).
        was_training = self.features.training
        self.features.eval()
        with torch.no_grad():
            flat = self.features(torch.zeros(1, in_channels, image_size, image_size)).numel()
        self.features.train(was_training)
        if flat == 0:
            raise ValueError(f"image_size={image_size} is too small for {len(conv_channels)} pooling layers")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ADVANCED (production alternative): let the framework infer the size
# What changes: nn.LazyLinear figures out in_features on the first forward.
# Why teams do it: no manual size maths when you change the architecture.
#   Same catch applies: run one dummy batch before
#   creating the optimizer.
#
#   self.classifier = nn.Sequential(nn.Flatten(), nn.LazyLinear(fc_size), nn.ReLU(), ...)
#   model(torch.zeros(1, 3, H, W))      # materialize, THEN build the optimizer
#
# And for real projects, prefer a ready-made backbone over a hand-written CNN:
#   import timm; model = timm.create_model("resnet50", pretrained=True, num_classes=N)


# =========================================================================
# 3. EFFICIENT CNN (MobileNet-style)
# =========================================================================
class InvertedResidualBlock(nn.Module):
    """
    In plain words: a cheap-but-strong CNN brick. Widen the channels (1x1),
    look at each channel's spatial pattern on its own (3x3), squeeze back down
    (1x1), and add the input back in (skip connection) so learning stays easy.

    
    Design notes (read these before comparing numbers with other implementations):
      * depthwise=True uses a TRUE depthwise conv (groups=hidden_dim). A
        layer labelled "depthwise" without a groups argument is an ordinary
        3x3 conv with far more parameters. Set depthwise=False for the
        ordinary conv.
      * The 1x1 shortcut is built automatically when shapes change (handled
        inside the block, not by the caller).
      * The skip is always added and a ReLU follows. The
        original MobileNetV2 only adds the skip when shapes match and has no
        ReLU after the add. Kept for simplicity.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion_factor: int = 6, depthwise: bool = True):
        super().__init__()
        hidden = in_channels * expansion_factor
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden if depthwise else 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        needs_projection = stride != 1 or in_channels != out_channels
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
            if needs_projection
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x if self.shortcut is None else self.shortcut(x)
        out = self.project(self.depthwise(self.expand(x)))
        return torch.relu(out + skip)


class MobileNetBackbone(nn.Module):
    """
    In plain words: a small stack of InvertedResidualBlocks. It turns an image
    into a smaller grid of feature channels.

    
    Change for your project: block_specs = ((out_channels, stride, expansion), ...).
    The default is 16 -> 24 -> 32 -> 64.
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 16,
        block_specs: Sequence[Tuple[int, int, int]] = ((24, 2, 3), (32, 2, 3), (64, 2, 6)),
        depthwise: bool = True,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        blocks, prev = [], stem_channels
        for out_c, stride, expansion in block_specs:
            blocks.append(InvertedResidualBlock(prev, out_c, stride, expansion, depthwise))
            prev = out_c
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


class MobileNetLikeClassifier(nn.Module):
    """
    In plain words: backbone + "average the grid, then decide the class".
    
    """

    def __init__(self, num_classes: int, backbone: Optional[MobileNetBackbone] = None):
        super().__init__()
        self.backbone = backbone or MobileNetBackbone()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# =========================================================================
# 4. PRETRAINED MODELS (TRANSFER LEARNING)
# =========================================================================
def get_head(model: nn.Module) -> nn.Linear:
    """
    In plain words: finds the last "decide the class" layer of a torchvision
    model. Different families name it differently (fc, classifier, heads).
    """
    if isinstance(getattr(model, "fc", None), nn.Linear):  # ResNet, RegNet
        return model.fc
    if hasattr(model, "classifier"):  # MobileNet, EfficientNet, VGG, ConvNeXt, DenseNet
        c = model.classifier
        if isinstance(c, nn.Linear):
            return c
        for layer in reversed(list(c.children())):
            if isinstance(layer, nn.Linear):
                return layer
    if hasattr(model, "heads") and hasattr(model.heads, "head"):  # ViT
        return model.heads.head
    raise ValueError("Unknown model family: set the head by hand")


def replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    """
    In plain words: swaps the final layer for a fresh one with YOUR number of
    classes. The rest of the pretrained network is untouched.

    
    """
    old = get_head(model)
    new = nn.Linear(old.in_features, num_classes)
    if isinstance(getattr(model, "fc", None), nn.Linear):
        model.fc = new
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Linear):
            model.classifier = new
        else:
            idx = max(i for i, m in enumerate(model.classifier) if isinstance(m, nn.Linear))
            model.classifier[idx] = new
    else:
        model.heads.head = new
    return model


def set_trainable(module: nn.Module, flag: bool) -> None:
    """Turn learning on (True) or off (False) for every weight in a module."""
    for p in module.parameters():
        p.requires_grad = flag


def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    In plain words: lock the pretrained part so only the new head learns.
    Fast, needs little data, and cannot damage what the backbone already knows.

    
    Gotcha: frozen BatchNorm layers STILL update their running averages while
    model.train() is on. Call freeze_batchnorm_stats(model) after every
    model.train() if you want the backbone fully frozen.
    """
    set_trainable(model, False)
    set_trainable(get_head(model), True)
    return model


def freeze_batchnorm_stats(model: nn.Module) -> None:
    """Keep BatchNorm layers in eval mode so frozen layers stay frozen."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


# ADVANCED (production alternative): gradual unfreezing + different learning rates
# What changes: after the head has learned, unfreeze the last block and train
#   it with a smaller learning rate than the head.
# Why teams do it: usually beats "head only" when you have a few thousand
#   images, without wrecking the pretrained features.
#
#   set_trainable(model.layer4, True)                     # ResNet example
#   optimizer = torch.optim.AdamW([
#       {"params": model.layer4.parameters(), "lr": 1e-4},
#       {"params": get_head(model).parameters(), "lr": 1e-3},
#   ], weight_decay=1e-2)


def build_classifier(
    arch: str = "mobilenet_v3_large",
    num_classes: int = 2,
    weights: Optional[str] = "DEFAULT",
    checkpoint: Optional[str] = None,
    checkpoint_has_new_head: bool = False,
    freeze: bool = True,
    strict: bool = True,
) -> nn.Module:
    """
    TEMPLATE: pretrained torchvision classifier for YOUR classes.
    In plain words: start from a network that already knows general visual
    patterns (trained on ImageNet), replace its last layer, and train only
    what you need.

    

    Three common situations:
      1. Download ImageNet weights (normal projects):
            build_classifier("resnet18", 5)                 # weights="DEFAULT"
      2. Load ImageNet weights from a local file (offline use):
            build_classifier("mobilenet_v3_large", 2, weights=None, checkpoint="mnv3.pth")
      3. Load YOUR OWN fine-tuned file (its head already has your classes):
            build_classifier("resnet50", 2, weights=None, checkpoint="mine.pth", checkpoint_has_new_head=True)

    Change for your project: arch (any torchvision name), num_classes.
    strict=True raises if the file does not match the model. strict=False
    silently ignores mismatched layers, so it is not the default.

    MODERNIZE: older tutorials build with weights=None and load a local
    file. In real projects pass weights="DEFAULT"
    (or an explicit enum such as ResNet18_Weights.IMAGENET1K_V1) and let
    torchvision download and cache it. The old `pretrained=True` argument is
    deprecated.
    """
    ctor = getattr(torchvision.models, arch)
    model = ctor(weights=None if checkpoint else weights)

    def _load(m: nn.Module) -> None:
        # weights_only=True: safe loading of plain tensors, and the default in
        # torch >= 2.6. A whole pickled model object would need False.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        report = m.load_state_dict(state, strict=False)
        if strict and (report.missing_keys or report.unexpected_keys):
            raise RuntimeError(f"Checkpoint mismatch. missing={report.missing_keys[:5]} unexpected={report.unexpected_keys[:5]}")

    if checkpoint and not checkpoint_has_new_head:
        _load(model)
    replace_head(model, num_classes)
    if checkpoint and checkpoint_has_new_head:
        _load(model)
    if freeze:
        freeze_backbone(model)
    return model


# =========================================================================
# 5. EMBEDDING NETWORKS (Siamese)
# =========================================================================
class EmbeddingNet(nn.Module):
    """
    In plain words: turns an image into a short list of numbers (an
    "embedding") so that similar images get similar numbers.

    
    Options added: a projection layer to a fixed embedding size, and L2
    normalization (makes distances comparable; standard for retrieval).
    Change for your project: the backbone (any module that outputs a
    feature map), embedding_dim.
    """

    def __init__(self, backbone: nn.Module, backbone_out: Optional[int] = None, embedding_dim: Optional[int] = None, normalize: bool = False):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.projection = None
        if embedding_dim is not None:
            if backbone_out is None:
                raise ValueError("backbone_out is required when embedding_dim is set")
            self.projection = nn.Linear(backbone_out, embedding_dim)
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pool(self.backbone(x))
        if self.projection is not None:
            z = self.projection(z)
        return nn.functional.normalize(z, dim=1) if self.normalize else z


class SiameseNetwork(nn.Module):
    """
    In plain words: runs three images through the SAME embedding network
    (shared weights) so the loss can compare them.
     Use with nn.TripletMarginLoss(margin=1.0).
    """

    def __init__(self, embedding_net: nn.Module):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, positive, negative):
        return self.embedding_net(anchor), self.embedding_net(positive), self.embedding_net(negative)

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.embedding_net(images)


# =========================================================================
# 6. SMALL UTILITIES
# =========================================================================
def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    In plain words: how many numbers the model has to learn. A quick size and
    cost indicator.
    
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def trainable_parameters(model: nn.Module):
    """Pass this to the optimizer so frozen weights are skipped."""
    return [p for p in model.parameters() if p.requires_grad]


def summarize_model(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 224, 224), depth: int = 3):
    """
    In plain words: a layer-by-layer table with output shapes and parameter
    counts.

     The plain torchinfo
    call is all you need. If torchinfo is not installed, falls back to counts.
    """
    try:
        from torchinfo import summary

        return summary(model, input_size=input_size, depth=depth, verbose=0)
    except ImportError:
        print(f"total params: {count_parameters(model, False):,} | trainable: {count_parameters(model):,}")
        print("(pip install torchinfo for the full table)")
        return None
