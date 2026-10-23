"""
deploy.py  |  Chapter 14: Export and serving   
=========================================================================
Getting a trained model out of the notebook and into something that answers
requests. Three layers, simplest first:

  1. Bundle      weights + class names + preprocessing, saved TOGETHER
  2. Predictor   load the bundle, predict from a PIL image or a batch
  3. Export      ONNX / TorchScript for runtimes that do not run PyTorch

The most common production bug is not the model: it is that the serving code
uses different resize/normalization than training, or a different class
order. The bundle exists to make that impossible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD
from .models import build_classifier


# =========================================================================
# 1. BUNDLE
# =========================================================================
def save_bundle(path: str, model: nn.Module, class_names: Sequence[str], arch: str, image_size: int,
                mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD) -> None:
    """
    In plain words: one file that carries everything the server needs: the
    weights, the class names IN ORDER, and the exact image preprocessing.
    Only plain types are stored (tensors, lists, numbers, strings), so it loads
    safely with torch.load(weights_only=True).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "num_classes": len(class_names),
        "arch": arch,
        "image_size": int(image_size),
        "mean": [float(m) for m in mean],
        "std": [float(s) for s in std],
    }, path)


class Predictor:
    """
    TEMPLATE: the object your API or batch job holds.
    In plain words: load once, then call predict() for each image.

        p = Predictor.from_bundle("model.pt")
        p.predict(Image.open("photo.jpg"), topk=3)
    Change for your project: for a custom network pass `builder`, a function
    (num_classes) -> nn.Module, instead of a torchvision arch name.
    """

    def __init__(self, model: nn.Module, class_names: Sequence[str], transform: Callable, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.class_names = list(class_names)
        self.transform = transform

    @classmethod
    def from_bundle(cls, path: str, builder: Optional[Callable[[int], nn.Module]] = None, device: str = "cpu") -> "Predictor":
        b = torch.load(path, map_location="cpu", weights_only=True)
        model = builder(b["num_classes"]) if builder else build_classifier(b["arch"], b["num_classes"], weights=None, freeze=False)
        model.load_state_dict(b["state_dict"])
        tf = transforms.Compose([
            transforms.Resize((b["image_size"], b["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(b["mean"], b["std"]),
        ])
        return cls(model, b["class_names"], tf, device)

    @torch.no_grad()
    def predict(self, image: Image.Image, topk: int = 1) -> List[Tuple[str, float]]:
        return self.predict_batch([image], topk)[0]

    @torch.no_grad()
    def predict_batch(self, images: Sequence[Image.Image], topk: int = 1, batch_size: int = 32) -> List[List[Tuple[str, float]]]:
        """Batches are MUCH faster than one-by-one, on CPU and especially on GPU."""
        results: List[List[Tuple[str, float]]] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            x = torch.stack([self.transform(im.convert("RGB")) for im in chunk]).to(self.device)
            probs = self.model(x).softmax(dim=1).cpu()
            top_p, top_i = probs.topk(min(topk, probs.shape[1]), dim=1)
            for p_row, i_row in zip(top_p, top_i):
                results.append([(self.class_names[int(j)], float(p)) for p, j in zip(p_row, i_row)])
        return results


# ADVANCED (production alternative): a minimal web API
# What changes: wrap the Predictor in an HTTP service. (NOT run in this
#   toolkit's tests; treat as a starting sketch.)
# Why teams do it: other systems can call your model over the network.
#
#   # pip install fastapi uvicorn python-multipart
#   import io
#   from fastapi import FastAPI, File, UploadFile
#   app = FastAPI()
#   predictor = Predictor.from_bundle("model.pt")           # load ONCE at start-up
#
#   @app.post("/predict")
#   async def predict(file: UploadFile = File(...)):
#       img = Image.open(io.BytesIO(await file.read()))
#       return {"top3": predictor.predict(img, topk=3)}
#
#   # run: uvicorn serve:app --host 0.0.0.0 --port 8000
# Production extras: input size limits, a /health route, request timeouts,
#   a Docker image with pinned versions, and model/version in every response.
# Serving at scale: NVIDIA Triton, TorchServe-style servers, or a managed endpoint.


# =========================================================================
# 3. EXPORT
# =========================================================================
def export_onnx(model: nn.Module, example_input: torch.Tensor, path: str, dynamic_batch: bool = True, opset: int = 18,
                single_file: bool = True) -> str:
    """
    In plain words: save the model in ONNX, a neutral format that ONNX
    Runtime, TensorRT, OpenVINO and many mobile/edge runtimes can run, with no
    PyTorch needed on the target machine.

    Gotchas:
      * Export in eval() mode (done here) so dropout/BatchNorm behave.
      * `example_input` fixes the image size/channels. dynamic_batch=True keeps
        the batch dimension flexible.
      * ALWAYS compare outputs after export (see verify_onnx).

    MODERNIZE - VERIFIED on torch 2.14.1:
      * PyTorch's default ONNX exporter is now the torch.export-based one. It
        takes `dynamic_shapes` (used here), not the old `dynamic_axes`.
      * It supports opset 18 and up. Asking for 17 printed a failed conversion
        and silently kept 18. Default is 18.
      * By default it writes the weights to a SEPARATE file (model.onnx.data)
        next to a tiny model.onnx. Move or upload only the .onnx and it will
        not load. single_file=True (default here) embeds everything in one file;
        models over 2 GB need external data.
    """
    model.eval()
    kwargs = {}
    if dynamic_batch:
        kwargs["dynamic_shapes"] = ({0: torch.export.Dim("batch")},)
    torch.onnx.export(
        model, (example_input,), path,
        input_names=["input"], output_names=["logits"],
        opset_version=opset, external_data=not single_file, **kwargs,
    )
    return path


def verify_onnx(model: nn.Module, onnx_path: str, example_input: torch.Tensor, atol: float = 1e-4) -> float:
    """
    In plain words: run the same input through PyTorch and through the ONNX
    file and report the biggest difference. If it is not tiny, do not ship.
    Needs: pip install onnxruntime
    """
    import onnxruntime as ort

    model.eval()
    with torch.no_grad():
        ref = model(example_input).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": example_input.numpy()})[0]
    diff = float(abs(ref - out).max())
    if diff > atol:
        raise AssertionError(f"ONNX output differs from PyTorch by {diff:.2e} (> {atol})")
    return diff


def export_torchscript(model: nn.Module, example_input: torch.Tensor, path: str) -> str:
    """
    In plain words: save a self-contained model file that can load in C++ or
    Python without your model's source code.
    MODERNIZE - VERIFIED on torch 2.14.1: torch.jit.trace and torch.jit.load now
    print a FutureWarning ("deprecated, please switch to torch.compile or
    torch.export"). It still worked correctly (zero output difference), and a
    lot of existing systems use it, but start new work on torch.export.
    """
    model.eval()
    traced = torch.jit.trace(model, example_input)
    traced.save(path)
    return path


# ADVANCED (production alternative): torch.export and other targets
#   ep = torch.export.export(model.eval(), (example_input,))   # PyTorch 2.x graph capture
#   torch.save(ep, "model.pt2")                                # (see current docs for save/load and AOTInductor)
# Targets:  TensorRT (NVIDIA GPUs), OpenVINO (Intel CPU), Core ML (Apple),
#           TFLite / ExecuTorch (phones, edge). Each has its own quantization.
