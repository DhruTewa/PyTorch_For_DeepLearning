# Chapter 14: Export and Serving

| | |
|---|---|
| **Level** | Intermediate to advanced |
| **You will learn** | How to package a trained model so someone else can use it correctly, serve predictions, and export to other runtimes |
| **Prerequisites** | Chapters 5, 6, 10 |
| **Ready-made code** | `cv_toolkit/deploy.py` |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

> [!TIP]
> **🟢 Extra: this whole chapter.** Many tutorials end at a trained model. This chapter covers getting it out of the notebook.

**In this chapter**
[14.1 The idea](#141-the-idea) · [14.2 The bundle](#142-the-bundle) · [14.3 The predictor](#143-the-predictor) · [14.4 Serving](#144-serving-over-http) · [14.5 Export formats](#145-export-formats) · [14.6 ONNX](#146-exporting-to-onnx) · [14.7 Choosing](#147-choosing-a-deployment-style) · [14.8 Common problems](#148-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 14.1 The idea

A good model in a notebook helps nobody. To be useful, it must run somewhere else (a server, a phone, a camera), on images it has never seen, with the **same preprocessing and the same class order as in training**.

The most common production bug is not the model. It is that the serving code resizes or normalises differently from training, or lists the classes in a different order. Everything in this chapter exists to make that impossible.

---

## 14.2 The bundle

Save everything the server needs **together, in one file**: the weights, the class names *in order*, the architecture name, and the exact preprocessing.

```python
torch.save({
    "state_dict": model.state_dict(),
    "class_names": class_names,            # a plain list, in the order of the model's outputs
    "arch": "resnet18",
    "image_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}, "model_bundle.pt")
```

Only plain types (tensors, lists, numbers, strings) are stored, so it loads with `weights_only=True`, which refuses to run hidden code. Never save the whole model object (`torch.save(model, ...)`): it breaks when your code changes and needs unsafe loading.

> [!TIP]
> **🟢 Extra: also record** the date, the git commit, the training config file, the validation score, and the dataset version. Six months from now, "which model is this?" must be answerable from the file or its folder (Chapter 0).

---

## 14.3 The predictor

Load once; call many times.

```python
class Predictor:
    def __init__(self, path, device="cpu"):
        b = torch.load(path, map_location="cpu", weights_only=True)
        model = getattr(models, b["arch"])(weights=None)                       # no download: weights come from the bundle
        model.fc = nn.Linear(model.fc.in_features, len(b["class_names"]))      # same last layer as when saved
        model.load_state_dict(b["state_dict"])
        self.model, self.device = model.to(device).eval(), device
        self.class_names = b["class_names"]
        self.tf = transforms.Compose([
            transforms.Resize((b["image_size"], b["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(b["mean"], b["std"]),
        ])

    @torch.no_grad()
    def predict(self, images, topk=3):
        x = torch.stack([self.tf(im.convert("RGB")) for im in images]).to(self.device)     # a batch
        probs = self.model(x).softmax(dim=1).cpu()
        top_p, top_i = probs.topk(min(topk, probs.shape[1]), dim=1)
        return [[(self.class_names[int(j)], float(p)) for p, j in zip(pr, ix)] for pr, ix in zip(top_p, top_i)]

predictor = Predictor("model_bundle.pt")
predictor.predict([Image.open("photo.jpg")], topk=2)       # [[('cat', 0.91), ('dog', 0.07)]]
```

**Points to note:**
- The line that rebuilds the last layer is architecture-specific; for a custom network, pass a builder function instead.
- `convert("RGB")` handles grayscale and PNG-with-alpha inputs.
- **Batches are much faster** than one image at a time, on CPU and especially on GPU.
- `eval()` and `no_grad` (Chapter 6).

---

## 14.4 Serving over HTTP

> [!NOTE]
> **🔵 Advanced alternative: a minimal web API with FastAPI.**
>
> ```python
> # pip install fastapi uvicorn python-multipart
> import io
> from fastapi import FastAPI, File, UploadFile, HTTPException
>
> app = FastAPI()
> predictor = Predictor("model_bundle.pt")            # load ONCE at start-up, not per request
>
> @app.get("/health")
> def health():
>     return {"status": "ok"}
>
> @app.post("/predict")
> async def predict(file: UploadFile = File(...)):
>     data = await file.read()
>     if len(data) > 5_000_000:
>         raise HTTPException(413, "image too large")
>     try:
>         img = Image.open(io.BytesIO(data))
>     except Exception:
>         raise HTTPException(400, "not an image")
>     return {"model_version": "1.0.0", "top3": predictor.predict([img], topk=3)[0]}
>
> # run:  uvicorn serve:app --host 0.0.0.0 --port 8000
> ```
>
> **What changes:** the model sits behind a network call, so other systems can use it.
>
> **Why teams do it:** one place to update the model, any language can call it. Production extras: input size limits (as above), a `/health` route for the load balancer, request timeouts, **the model version in every response** (so you can trace a bad prediction to a model), a Docker image with pinned library versions, and logging of requests (Chapter 16).
>
> **For heavy traffic:** serving systems such as NVIDIA Triton or TorchServe-style servers add dynamic batching and multi-model management; managed cloud endpoints remove the operations work.

> [!WARNING]
> **Watch out: untrusted images.** A user can upload a file that decodes to an enormous image. Pillow guards against this with `Image.MAX_IMAGE_PIXELS`: leave the guard on, and cap file size and dimensions. Treat all uploads as untrusted input.

---

## 14.5 Export formats

PyTorch is not always available where the model runs. Export to a neutral format:

| Format | Runs on | Use when |
|---|---|---|
| **PyTorch `state_dict`** | Anywhere PyTorch runs | Your own Python services |
| **ONNX** | ONNX Runtime, TensorRT, OpenVINO, many mobile and edge runtimes | The most common neutral format. **The default choice** |
| **TorchScript** | C++ and Python without your source | Existing systems that use it |
| **`torch.export`** | PyTorch 2.x graph capture; basis of newer flows | New projects staying in the PyTorch ecosystem |
| **TensorRT** | NVIDIA GPUs | Maximum GPU speed |
| **Core ML** | Apple devices | iOS and macOS apps |
| **TFLite, ExecuTorch** | Phones and microcontrollers | On-device inference |

> [!IMPORTANT]
> **🟣 Outdated vs current: TorchScript.** `torch.jit.trace` and `torch.jit.load` now print a deprecation warning pointing to `torch.compile` and `torch.export`. TorchScript still works and many systems use it, but start new work on ONNX or `torch.export`.

---

## 14.6 Exporting to ONNX

```python
model.eval()                                            # export in eval mode so dropout and BatchNorm behave
example = torch.randn(1, 3, 224, 224)                   # fixes the image size and channels

torch.onnx.export(
    model, (example,), "model.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_shapes=({0: torch.export.Dim("batch")},),   # keep the batch size flexible
    opset_version=18,
    external_data=False,                                # keep everything in ONE file
)
```

**Then verify. Always.** Run the same input through PyTorch and through ONNX and compare:

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
batch = torch.randn(4, 3, 224, 224)                     # a different batch size from the export example
onnx_out = session.run(None, {"input": batch.numpy()})[0]
with torch.no_grad():
    torch_out = model(batch).numpy()
print(abs(onnx_out - torch_out).max())                  # should be tiny, around 1e-5 or smaller
```

If the difference is not tiny, do not ship.

> [!WARNING]
> **Watch out.**
> - **Version differences.** Recent PyTorch versions use a `torch.export`-based exporter that takes `dynamic_shapes` and a minimum opset (18 in the version this was written against). Older versions use `dynamic_axes`. If an argument is rejected or the opset is silently changed, check your version's documentation.
> - **Separate weight file.** By default the new exporter may write the weights to a second file (`model.onnx.data`) beside a tiny `model.onnx`. Move or upload only the `.onnx` and it will not load. `external_data=False` keeps one file; models over 2 GB need the external file.
> - The image **size and channels are fixed** by the example input unless you declare them dynamic.
> - **Preprocessing is not part of the ONNX graph**: resizing and normalising must be repeated in your serving code, with the same numbers. Store them in the bundle (14.2).
> - Custom or unusual operations may not export. Fix by using standard layers, or a newer opset.

> [!NOTE]
> **🔵 Advanced alternative: `torch.export` and other targets.**
>
> ```python
> exported = torch.export.export(model.eval(), (example,))      # PyTorch 2.x graph capture
> ```
>
> **Why teams do it:** it is the foundation of the current PyTorch compile and export story, and of newer quantization and on-device flows (ExecuTorch). Check the current documentation for saving, loading and ahead-of-time compilation.
>
> Each target brings its own tooling: TensorRT (NVIDIA GPUs), OpenVINO (Intel CPUs), Core ML tools (Apple), TFLite or ExecuTorch (phones and edge). Each has its own quantization step; shrink the model first (Chapter 13).

---

## 14.7 Choosing a deployment style

| Style | When | Notes |
|---|---|---|
| **Batch job** (script over a folder or database) | Results are not needed instantly | Simplest. Big batches, no web server |
| **Web API** (FastAPI, 14.4) | Other systems call it on demand | The usual default |
| **On-device** (phone, camera) | No network, privacy, low latency | Shrink first (Chapter 13); export to the device's runtime |
| **In the browser** (ONNX Runtime Web, TensorFlow.js) | No server cost, private | Small models only |

**Also decide:** CPU or GPU (a small model on CPU is often cheaper and fast enough), the latency you promised, and how updates reach the users: a new model file, a new container, an app release.

---

## 14.8 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Different preprocessing in serving | Great validation score, poor live accuracy | Store the transform in the bundle; use one code path |
> | Class order differs | Right shapes, wrong names | Save `class_names` in order, inside the bundle |
> | Model in `train()` mode | Predictions vary between calls | `model.eval()` |
> | Whole model object pickled | Loading breaks after code changes, or needs unsafe loading | Save `state_dict` and rebuild |
> | Loaded the model per request | Every request is slow | Load once at start-up |
> | ONNX not verified | Subtle differences in production | Compare against PyTorch on real inputs |
> | ONNX opened without its `.data` file | Load error on another machine | Single-file export |
> | Fixed batch size in the export | Errors on batches other than 1 | Declare the batch dimension dynamic |
> | Untrusted uploads | Crashes, huge memory use | Size and dimension limits, validation, timeouts |
> | No model version in responses | Cannot trace a bad prediction | Include the version in every response |
> | Library versions drift | Works here, breaks there | Pin versions; build from a container |

---

## Key takeaways

1. **Bundle** weights, class names (in order), architecture and preprocessing in one file. Most deployment bugs are mismatches between training and serving.
2. Wrap it in a **predictor** that loads once and predicts in batches, in `eval()` mode.
3. **ONNX** is the default neutral export. **Verify it against PyTorch** and remember preprocessing is not included.
4. TorchScript still works but is deprecated: prefer ONNX or `torch.export` for new work.
5. A web API needs a health route, input limits, versioned responses and pinned dependencies.
6. Shrink first (Chapter 13), then export for the target runtime.

**Previous:** [Chapter 13: Shrinking models](13_optimization.md) &nbsp;|&nbsp; **Next:** [Chapter 15: Testing and packaging](15_testing_packaging.md)
