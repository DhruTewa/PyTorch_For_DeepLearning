"""
interpret.py  |  Chapter 12: Interpretability ("why did it say that?")
=========================================================================
Three views into a trained CNN, :
  1. Feature maps   what each layer responds to          (forward hooks)
  2. Saliency       which input pixels move the answer   (gradient w.r.t. input)
  3. CAM / Grad-CAM which image REGIONS drove the class  (feature map x weights)

Pictures of the results are drawn by viz.py (overlay_heatmap, show_feature_maps).

Why bother in a real project: to catch a model that is right for the wrong
reason (a ruler in the photo, a hospital logo, the watermark that only
appears on one class). Look at these BEFORE you trust a good accuracy number.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivationRecorder:
    """
    In plain words: a "wiretap" you attach to chosen layers. While it is
    attached, it copies what each layer outputs. Leaving the `with` block
    removes the wiretaps, even if an error happens.

    Removing hooks by hand after the forward
    pass is fragile: if the pass raises an error the hooks stayed on the model and kept
    leaking memory. A context manager makes cleanup automatic.

        with ActivationRecorder({"layer1": model.layer1}) as rec:
            model(image)
        rec.activations["layer1"].shape
    """

    def __init__(self, layers: Dict[str, nn.Module]):
        self.layers = layers
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles: list = []

    def __enter__(self) -> "ActivationRecorder":
        for name, layer in self.layers.items():
            self._handles.append(layer.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self.activations[name] = output.detach()

        return hook


def resnet_probe_layers(model: nn.Module) -> Dict[str, nn.Module]:
    """
    The five probe points used here for ResNet-style models:
    the first conv and the first conv of each of the 4 stages.
    Change for your project: for another architecture, build your own dict
    {"name": module}. List candidates with: [n for n, _ in model.named_modules()]
    """
    return {
        "conv1": model.conv1,
        "layer1": model.layer1[0].conv1,
        "layer2": model.layer2[0].conv1,
        "layer3": model.layer3[0].conv1,
        "layer4": model.layer4[0].conv1,
    }


@torch.no_grad()
def capture_activations(model: nn.Module, image: torch.Tensor, layers: Dict[str, nn.Module]) -> Dict[str, torch.Tensor]:
    """
    In plain words: run one image through the model and return what the chosen
    layers produced. `image` is the normalized batch tensor (1, C, H, W).
    
    """
    model.eval()
    with ActivationRecorder(layers) as rec:
        model(image)
    return dict(rec.activations)


def _scale01(t: torch.Tensor) -> torch.Tensor:
    return (t - t.min()) / (t.max() - t.min() + 1e-8)


def top_channel_maps(activations: Dict[str, torch.Tensor], size: Tuple[int, int] = (224, 224)) -> Tuple[List[str], List[torch.Tensor]]:
    """
    In plain words: from each layer, take the single most active channel,
    blow it up to the image size, scale to 0..1. Shows how "what the network
    looks at" goes from fine detail to big shapes as you go deeper.
     Returns (names, maps).
    """
    names, maps = [], []
    for name, fm in activations.items():
        best = fm.mean(dim=(2, 3)).argmax().item()  # channel with highest average activation
        sel = fm[:, best : best + 1]
        sel = F.interpolate(sel, size=size, mode="bilinear", align_corners=False)
        names.append(name)
        maps.append(_scale01(sel))
    return names, maps


def saliency_map(model: nn.Module, image: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
    """
    In plain words: nudge each input pixel a tiny bit and see how much the
    class score changes. Pixels with big effect are "salient". Result is an
    (H, W) map scaled 0..1. Fast, but noisy and often speckled.

    
    class_idx=None explains the model's own top prediction.
    Gotcha: this needs gradients, so no torch.no_grad() around it.
    """
    model.eval()
    x = image.clone().detach().requires_grad_(True)
    out = model(x)
    idx = int(out.argmax(1)) if class_idx is None else class_idx
    model.zero_grad()
    out[0, idx].backward()
    grads = x.grad.abs().sum(dim=1)[0]  # strongest effect across colour channels
    return _scale01(grads).detach()


def class_activation_map(
    model: nn.Module,
    image: torch.Tensor,
    class_idx: Optional[int] = None,
    feature_layer: Optional[nn.Module] = None,
    fc_layer: Optional[nn.Linear] = None,
) -> torch.Tensor:
    """
    In plain words: the model's final decision is a weighted sum of "feature
    channels". Re-do that sum but KEEP the picture grid instead of averaging
    it away, and you get a map of where in the image the evidence for the
    class sits. Result is (H, W) in 0..1, same size as the input.

    
    Works ONLY for models shaped "conv features -> global average pool -> one
    Linear layer" (ResNet, many others). For other models use grad_cam below.

    Hooking layer4[-1].conv3 would give the raw
    conv output BEFORE BatchNorm, the residual add and ReLU, so it is only an
    approximation of the real last feature map. Default here is the whole
    model.layer4 (the true last feature map). You can check it: the mean of
    the CAM plus the fc bias equals the class logit exactly.
    """
    feature_layer = feature_layer if feature_layer is not None else model.layer4
    fc_layer = fc_layer if fc_layer is not None else model.fc
    model.eval()
    captured: Dict[str, torch.Tensor] = {}
    handle = feature_layer.register_forward_hook(lambda _m, _i, o: captured.__setitem__("f", o.detach()))
    try:
        with torch.no_grad():
            out = model(image)
    finally:
        handle.remove()
    idx = int(out.argmax(1)) if class_idx is None else class_idx
    cam = torch.einsum("c,chw->hw", fc_layer.weight[idx].detach(), captured["f"].squeeze(0))
    cam = F.relu(cam)  # keep evidence FOR the class only
    cam = _scale01(cam)
    cam = F.interpolate(cam[None, None], size=image.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    return cam.cpu()


def grad_cam(model: nn.Module, image: torch.Tensor, target_layer: nn.Module, class_idx: Optional[int] = None) -> torch.Tensor:
    """
     the standard upgrade of CAM).
    In plain words: like CAM, but it works with ANY architecture. It weighs
    each feature channel by how strongly the class score depends on it
    (the gradient), instead of using the last Linear layer's weights.

    target_layer: usually the last convolutional stage (ResNet: model.layer4).
    Gotcha: if the layer's output is later modified in place (ReLU(inplace=True)
    straight after it), the hook can see a changed tensor. Pick a layer
    followed by a non-in-place op (e.g. pooling).
    """
    model.eval()
    store: Dict[str, torch.Tensor] = {}

    def fwd(_m, _i, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    handle = target_layer.register_forward_hook(fwd)
    try:
        out = model(image)
        idx = int(out.argmax(1)) if class_idx is None else class_idx
        model.zero_grad()
        out[0, idx].backward()
    finally:
        handle.remove()
    weights = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["a"]).sum(dim=1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    return _scale01(cam).cpu()


# ADVANCED (production alternative): use a maintained library
# What changes: Grad-CAM variants (Score-CAM, Eigen-CAM...), integrated
#   gradients, SHAP-style attributions, all tested on many architectures.
# Why teams do it: fewer subtle hook bugs, more methods to cross-check with.
#
#   # pip install grad-cam
#   from pytorch_grad_cam import GradCAM
#   from pytorch_grad_cam.utils.image import show_cam_on_image
#   cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
#   heat = cam(input_tensor=image)[0]              # (H, W) in 0..1
#
#   # pip install captum
#   from captum.attr import IntegratedGradients
#   attr = IntegratedGradients(model).attribute(image, target=class_idx)
