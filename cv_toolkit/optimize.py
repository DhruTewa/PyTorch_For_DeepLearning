"""
optimize.py  |  Chapter 13: Shrinking models for deployment
=========================================================================
Make a trained model smaller and faster, then check what it cost in accuracy.

  1. Measure          size, latency, sparsity            (always do this first)
  2. Pruning          zero out the least useful weights
  3. Dynamic quant    store Linear weights as 8-bit integers (easiest win)
  4. Fusing + QAT     fuse layers, train with 8-bit rounding in the loop

Golden rule: measure accuracy on a held-out set after EVERY step. A smaller
model that lost 5 points of accuracy is not a win.

MODERNIZE (applies to the whole file) - VERIFIED on torch 2.14.1:
this is PyTorch's "eager mode" quantization API, torch.ao.quantization (also reachable through the older torch.quantization alias). Running it
prints PyTorch's own warning: "torch.ao.quantization is deprecated and will
be removed in 2.10", with the advice to migrate eager-mode quantization to
the `torchao` library and FX/PT2E quantization to torchao's pt2e API.
It still ran correctly in every test here, so the concepts and the code in
this file are fine for learning. For a new long-lived project, build on
torchao instead (see https://github.com/pytorch/ao). Not covered here because
its API is changing; check its docs for the
current names.
"""
from __future__ import annotations

import copy
import io
import time
from typing import Dict, Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import torch.ao.quantization as aoq

from .evaluate import evaluate


# =========================================================================
# 1. MEASURE
# =========================================================================
def model_size_mb(model: nn.Module) -> float:
    """
    In plain words: how big the model is on disk, in megabytes.
    Parameter counts alone are misleading:
    quantization changes bytes per parameter, not the count, so size on disk
    is the honest number.
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


@torch.no_grad()
def benchmark_latency(model: nn.Module, iters: int = 20, shape: Tuple[int, ...] = (16, 3, 224, 224), device: str = "cpu", warmup: int = 3) -> float:
    """
    In plain words: average seconds per forward pass on a dummy batch.
     Added warm-up runs (the first calls are always slower
    and would skew a short test) and a GPU sync so timings are real.
    Gotcha: quantized models run on CPU only; compare on CPU.
    """
    model.eval()
    x = torch.randn(shape, device=device)
    for _ in range(warmup):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def prunable_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    """Yields every Conv2d and Linear layer (the layers that hold the weights)."""
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            yield name, m


def sparsity_report(model: nn.Module) -> Dict[str, object]:
    """
    In plain words: what fraction of the weights are exactly zero, per layer
    and overall. 
    Returns {"layers": {name: fraction}, "global_sparsity": fraction}.
    """
    layers, zeros, total = {}, 0, 0
    for name, m in prunable_modules(model):
        w = m.weight.detach()
        z, n = int((w == 0).sum()), w.numel()
        layers[f"{name}.weight"] = z / n if n else 0.0
        zeros, total = zeros + z, total + n
    return {"layers": layers, "global_sparsity": zeros / total if total else 0.0}


def compare_models(variants: Dict[str, nn.Module], loader, input_shape: Tuple[int, ...] = (16, 3, 224, 224), device: str = "cpu") -> List[dict]:
    """
     the one table that matters. For each model
    variant: accuracy, size on disk, latency. Pick the smallest/fastest one
    whose accuracy you can live with.
    """
    dev = torch.device(device)
    rows = []
    for name, m in variants.items():
        m = m.to(dev)
        rows.append({
            "model": name,
            "acc": evaluate(m, loader, dev)["acc"],
            "size_mb": model_size_mb(m),
            "latency_s": benchmark_latency(m, shape=input_shape, device=device),
        })
    for r in rows:
        print(f"{r['model']:<14} acc {r['acc']:.4f} | {r['size_mb']:.2f} MB | {r['latency_s'] * 1000:.1f} ms/batch")
    return rows


# =========================================================================
# 2. PRUNING
# =========================================================================
def prune_model(model: nn.Module, amount: float = 0.3, mode: str = "l1_unstructured") -> nn.Module:
    """
    In plain words: switch off the weights that matter least. "amount=0.3" =
    remove 30% in every Conv/Linear layer.
      l1_unstructured : zero the smallest individual weights (keeps shape)
      ln_structured   : zero whole filters/neurons (n=2, dim=0)

     Works IN PLACE.
    Gotcha 1: this only puts zeros in; the file and the speed do not shrink
    unless you use sparse-aware kernels or remove the structure. Structured
    pruning is the one that can actually become a smaller dense model.
    Gotcha 2: fine-tune a few epochs afterwards; accuracy recovers a lot.
    """
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount must be in [0,1], got {amount}")
    for _, m in prunable_modules(model):
        if mode == "l1_unstructured":
            prune.l1_unstructured(m, name="weight", amount=amount)
        elif mode == "ln_structured":
            prune.ln_structured(m, name="weight", amount=amount, n=2, dim=0)
        else:
            raise ValueError("mode must be 'l1_unstructured' or 'ln_structured'")
    return model


def finalize_pruning(model: nn.Module) -> nn.Module:
    """
    In plain words: pruning keeps a hidden "mask" next to the original weights.
    This bakes the zeros in permanently and deletes the mask, so you can save
    a clean model. 
    """
    for _, m in prunable_modules(model):
        if hasattr(m, "weight_orig") and hasattr(m, "weight_mask"):
            prune.remove(m, "weight")
    return model


# ADVANCED (production alternative): global pruning, then actually shrink
# What changes: rank ALL weights together (not layer by layer), so unimportant
#   layers lose more and important ones lose less.
# Why teams do it: the same accuracy at a higher total sparsity.
#
#   params = [(m, "weight") for _, m in prunable_modules(model)]
#   prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.5)
#
# To get a genuinely smaller/faster dense network from structured pruning,
# remove the zeroed channels and rebuild the layers (torch-pruning library),
# or export to a runtime that exploits sparsity.


# =========================================================================
# 3. DYNAMIC QUANTIZATION
# =========================================================================
def _set_engine(backend: str) -> None:
    try:
        torch.backends.quantized.engine = backend
    except Exception:
        pass  # keep whatever the platform supports


def quantize_dynamic_linear(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    """
    In plain words: store the Linear layers' weights as 8-bit integers instead
    of 32-bit decimals (about 4x smaller for those layers) and do the maths
    in integers. No retraining, no calibration data. The easiest win; best
    for models heavy in Linear layers (transformers, MLP heads). It does little
    for pure-conv CNNs because it skips Conv layers.

     Returns a COPY.
    backend: "fbgemm" for x86 CPUs, "qnnpack" for ARM (phones, Apple silicon).
    Gotcha: CPU only.
    """
    fp32 = copy.deepcopy(model).eval()
    _set_engine(backend)
    q = aoq.quantize_dynamic(fp32, {nn.Linear}, dtype=torch.qint8)
    return q.eval()


# =========================================================================
# 4. FUSING + QUANTIZATION-AWARE TRAINING (QAT)
# =========================================================================
def fuse_conv_bn_relu(model: nn.Module) -> nn.Module:
    """
    In plain words: merge neighbouring layers into one (Conv+BatchNorm+ReLU,
    Conv+BatchNorm, Conv+ReLU, Linear+ReLU). Fewer operations, less rounding
    error once quantized. Results are the same (check it!).

     Works IN PLACE and returns the
    same object. It only finds patterns that sit next to each other inside an
    nn.Sequential; blocks with residual adds (ResNet) need explicit lists.
    """
    for _, child in model.named_children():
        fuse_conv_bn_relu(child)
        if isinstance(child, nn.Sequential) and len(child) >= 2:
            was_training = child.training
            child.eval()  # BatchNorm folding needs eval mode
            i = 0
            while i < len(child) - 1:
                a, b = child[i], child[i + 1]
                c = child[i + 2] if i + 2 < len(child) else None
                if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d) and isinstance(c, nn.ReLU):
                    aoq.fuse_modules(child, [str(i), str(i + 1), str(i + 2)], inplace=True)
                    i += 3
                elif isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                    aoq.fuse_modules(child, [str(i), str(i + 1)], inplace=True)
                    i += 2
                elif isinstance(a, nn.Conv2d) and isinstance(b, nn.ReLU):
                    aoq.fuse_modules(child, [str(i), str(i + 1)], inplace=True)
                    i += 2
                elif isinstance(a, nn.Linear) and isinstance(b, nn.ReLU):
                    aoq.fuse_modules(child, [str(i), str(i + 1)], inplace=True)
                    i += 2
                else:
                    i += 1
            child.train(was_training)
    return model


def count_fused_layers(model: nn.Module) -> Dict[str, int]:
    """ How many fused modules exist, by type."""
    names = [m.__class__.__name__ for m in model.modules() if any(k in m.__class__.__name__ for k in ("ConvReLU2d", "ConvBn2d", "ConvBnReLU2d", "LinearReLU"))]
    return {k: names.count(k) for k in sorted(set(names))}


class QATWrapper(nn.Module):
    """
    In plain words: puts a "convert to int8" door at the model's entrance and
    a "convert back" door at the exit. Quantized models need these markers to
    know where the integer world begins and ends. 
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = aoq.QuantStub()
        self.model = model
        self.dequant = aoq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequant(self.model(self.quant(x)))


def prepare_qat(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    """
    In plain words: get a model ready for quantization-aware training. It
    inserts "fake rounding" into the forward pass, so during training the
    weights learn to survive being squeezed into 8 bits. Usually keeps more
    accuracy than quantizing afterwards.

     Returns a trained-able COPY in train mode.
    Flow:  prepare_qat -> train a few epochs (small learning rate) ->
           convert_qat -> evaluate on CPU.
    """
    qat = copy.deepcopy(model).eval()
    _set_engine(backend)
    fuse_conv_bn_relu(qat)
    qat.qconfig = aoq.get_default_qat_qconfig(backend)
    qat.train()
    aoq.prepare_qat(qat, inplace=True)
    return qat


def convert_qat(qat_model: nn.Module) -> nn.Module:
    """
    In plain words: after QAT training, replace the fake rounding with real
    int8 layers. The result runs on CPU only.
    
    """
    qat_model.eval()
    return aoq.convert(qat_model)


# ADVANCED (production alternative): post-training static quantization
# What changes: no retraining. Run a few hundred real images through the model
#   once so it can measure typical value ranges (calibration), then convert.
# Why teams do it: quick, no training code; good first try before QAT.
#
#   m = QATWrapper(fuse_conv_bn_relu(copy.deepcopy(model).eval())).eval()
#   m.qconfig = aoq.get_default_qconfig("fbgemm")
#   aoq.prepare(m, inplace=True)
#   with torch.no_grad():
#       for images, _ in calibration_loader: m(images)      # calibration
#   int8 = aoq.convert(m)
#
# Decision guide:
#   dynamic quantization  -> Linear-heavy models, zero effort
#   static (PTQ)          -> CNNs, small accuracy drop is OK, no training budget
#   QAT                   -> CNNs where PTQ lost too much accuracy
# For deployment on phones/edge or GPUs, export (deploy.py) and use that
#   runtime's own quantization tools (TensorRT, ONNX Runtime, Core ML, TFLite).
