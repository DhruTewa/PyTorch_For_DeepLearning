# Chapter 13: Shrinking Models for Deployment

| | |
|---|---|
| **Level** | Intermediate to advanced |
| **You will learn** | How to measure size and speed, and the options for making a model smaller and faster (smaller design, quantization, pruning, distillation) and which are worth it |
| **Prerequisites** | Chapters 4, 5, 6, 10 |
| **Ready-made code** | `cv_toolkit/optimize.py` |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

**In this chapter**
[13.1 The idea](#131-the-idea) · [13.2 Measure first](#132-measure-first) · [13.3 The ladder](#133-the-ladder-cheapest-first) · [13.4 Quantization](#134-quantization) · [13.5 Pruning](#135-pruning) · [13.6 Distillation](#136-distillation) · [13.7 Choosing](#137-choosing-and-checking) · [13.8 Common problems](#138-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 13.1 The idea

A model that wins on a GPU server may be too big or too slow for a phone, a camera, a browser or a budget. **Shrinking** trades a little accuracy for large savings in **size** (storage, download), **speed** (latency, throughput) and **memory and energy**.

Always start from a target: "under 10 MB", "under 50 ms per image on this device", "runs on CPU". Without a target you cannot tell when to stop.

---

## 13.2 Measure first

Three numbers, measured on **the hardware you will deploy to** (a laptop GPU's speed says little about a phone CPU):

```python
import io, time

def size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6

def latency_ms(model, shape=(1, 3, 224, 224), iters=50, warmup=5):
    model.eval()
    x = torch.randn(*shape)
    with torch.no_grad():
        for _ in range(warmup):          # first runs are slow (caches, allocation): discard them
            model(x)
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
    return (time.perf_counter() - start) / iters * 1000

# GPU timing: call torch.cuda.synchronize() before reading the clock, or you time only the launch
```

| Number | Why |
|---|---|
| **File size** | Download and storage limits |
| **Latency** (ms per image, **batch size 1** for real-time use) | User-facing speed |
| **Accuracy** (overall and per class, Chapter 10) | What you are trading away |

**Compare against the original on the same validation data**, always. Latency depends on batch size, image size and hardware; state all three when you report it.

> [!WARNING]
> **Watch out.** Parameter count and speed are **not** the same thing. A network with fewer parameters can be slower if it uses operations your hardware runs poorly. Measure latency; do not infer it.

---

## 13.3 The ladder, cheapest first

| Rung | What | Effort | Typical gain | Risk |
|---|---|---|---|---|
| 1 | **Use a smaller architecture** (for example `mobilenet_v3_large` instead of `resnet50`) or **lower the input resolution** | Lowest | Large | Some accuracy loss; fine-tune and re-measure |
| 2 | **Quantization** (8-bit numbers) | Low | About 4x smaller | Small accuracy loss; CPU-focused |
| 3 | **Pruning** (remove weights) | Medium | Size/speed gains only with the right tools | Needs fine-tuning; often no real speed-up |
| 4 | **Distillation** (small model learns from a big one) | Higher | Often the best accuracy for a given small size | Extra training |

> [!TIP]
> **🟢 Extra: rung 1 first, always.** Choosing a smaller network and training it well beats most clever compression tricks, and costs no extra engineering. Compress a model you have already made small by design, not a huge one.

---

## 13.4 Quantization

**In plain words:** store the model's numbers as **8-bit integers** instead of 32-bit decimals. The file is about 4 times smaller, integer maths is faster on many CPUs, and accuracy usually drops very little.

| Type | How | Needs | Best for |
|---|---|---|---|
| **Dynamic** | Weights stored as int8; activations converted on the fly | Nothing: no data, no retraining | Models heavy in `Linear` layers (transformers, MLP heads). Does little for pure-convolution CNNs |
| **Static (post-training)** | Weights *and* activations in int8, using a few hundred sample images to calibrate | A small **calibration** set | CNNs, when accuracy loss is acceptable |
| **Quantization-aware training (QAT)** | Simulate int8 rounding *during* training so the weights adapt | A short extra training run | When static quantization loses too much accuracy |

**Dynamic (the easy win):**

```python
q = torch.ao.quantization.quantize_dynamic(model.eval(), {nn.Linear}, dtype=torch.qint8)
```

For a model dominated by `Linear` layers the file shrinks about 4 times. It skips convolution layers.

**Static or QAT for CNNs.** Three steps:

1. **Fuse** neighbouring layers (Conv + BatchNorm + ReLU) into one operation: faster and more accurate once quantized. Fusing only finds patterns that sit *next to each other inside an `nn.Sequential`* (or are listed by name), and the model must be in `eval()` mode for fusing.
2. **Mark where the int8 world starts and ends** with `QuantStub` and `DeQuantStub`.
3. **Prepare, train or calibrate, convert.**

```python
ao = torch.ao.quantization

class QuantReady(nn.Module):
    def __init__(self, features, head):
        super().__init__()
        self.quant, self.features, self.head, self.dequant = ao.QuantStub(), features, head, ao.DeQuantStub()
    def forward(self, x):
        return self.dequant(self.head(self.features(self.quant(x))))

model = QuantReady(features, head).eval()
torch.backends.quantized.engine = "fbgemm"                 # "fbgemm" for x86 CPUs, "qnnpack" for ARM (phones, Apple silicon)
ao.fuse_modules(model.features, [["0", "1", "2"]], inplace=True)    # names of the Conv, BatchNorm, ReLU in the Sequential

# QAT
model.qconfig = ao.get_default_qat_qconfig("fbgemm")
model.train()
ao.prepare_qat(model, inplace=True)
# ... train a few epochs with a SMALL learning rate (for example 1e-4) ...
model.eval()
int8_model = ao.convert(model)                             # real int8 layers; runs on CPU only

# Static post-training instead of QAT:
#   model.qconfig = ao.get_default_qconfig("fbgemm"); ao.prepare(model, inplace=True)
#   run 100 to 500 representative images through model; ao.convert(model, inplace=True)
```

> [!WARNING]
> **Watch out.**
> - **Quantized models run on CPU only** (in this workflow). Evaluate them on CPU.
> - Calibration images must be **representative**: a handful of unusual images gives poor scaling.
> - Residual additions (ResNet's skip connections) need quantization-friendly operations (`FloatFunctional`), so ready-made ResNets often need a rewrite before they quantize cleanly.
> - Re-check accuracy **per class**: overall accuracy can hold while one class drops.

> [!IMPORTANT]
> **🟣 Outdated vs current: `torch.ao.quantization`.** Recent PyTorch versions print a notice that this module is deprecated in favour of the separate `torchao` library, and the newer export-based quantization flows are the direction for new projects. The workflow above is the long-standing one and still the clearest way to learn the ideas; for a new production project, check the current PyTorch and `torchao` documentation and consider the export-based flow, ONNX Runtime quantization, or TensorRT.

---

## 13.5 Pruning

**In plain words:** switch off the weights that matter least. "Prune 30%" removes the smallest-magnitude 30% of weights in each layer.

```python
import torch.nn.utils.prune as prune

for module in model.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        prune.l1_unstructured(module, name="weight", amount=0.3)      # zero the smallest 30%

# ...fine-tune a few epochs to recover accuracy...

for module in model.modules():                                         # make the zeros permanent, delete the masks
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        prune.remove(module, "weight")
```

| Type | What is removed | Effect |
|---|---|---|
| **Unstructured** (`l1_unstructured`) | Individual weights | High sparsity, but the tensors keep their size |
| **Structured** (`ln_structured`) | Whole filters or neurons | Can become a genuinely smaller network, but costs more accuracy |

> [!WARNING]
> **Watch out: pruning usually does not make the file smaller or the model faster by itself.** Unstructured pruning only puts zeros into the same-size tensors. Dense hardware still multiplies every zero, and the saved file is the same size. You gain only with sparse-aware storage or kernels, or structured pruning followed by *rebuilding* a smaller network.
> Always **fine-tune after pruning**: accuracy recovers a lot.

> [!NOTE]
> **🔵 Advanced alternative: global pruning, then actually shrink.**
>
> ```python
> params = [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
> prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.5)
> ```
>
> **What changes:** all weights are ranked together, so unimportant layers lose more and important ones lose less, giving the same accuracy at a higher total sparsity.
>
> **Why teams do it:** better accuracy per removed weight. To turn structured pruning into a smaller, faster dense model, remove the zeroed channels and rebuild the layers (the `torch-pruning` library automates this), or export to a runtime that exploits sparsity.

---

## 13.6 Distillation

> [!TIP]
> **🟢 Extra.**

**In plain words:** train a small **student** to imitate a large, accurate **teacher**. The teacher's full probability output carries more information than the hard label ("90% cat, 8% fox, 2% dog" tells the student that cats look a bit like foxes).

```python
def distill_loss(student_logits, teacher_logits, y, T=4.0, alpha=0.5):
    soft = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean") * (T * T)               # match the teacher's softened probabilities
    hard = F.cross_entropy(student_logits, y)                      # still learn the true labels
    return alpha * soft + (1 - alpha) * hard

# in the training loop:
with torch.no_grad():
    teacher_logits = teacher.eval()(x)
loss = distill_loss(student(x), teacher_logits, y)
```

| Setting | Meaning | Typical |
|---|---|---|
| `T` (temperature) | Softens the probabilities so small ones become visible | 2 to 5 |
| `alpha` | Balance between imitating the teacher and the true labels | 0.3 to 0.7 |

It often gives the best accuracy for a given small size, and it combines with quantization afterwards. The cost is an extra training stage and a good teacher.

---

## 13.7 Choosing and checking

| Your target | Start with |
|---|---|
| Mobile or edge, CPU | A small architecture (`mobilenet_v3_large`, `efficientnet_b0`), then static quantization or QAT |
| Just smaller downloads | Dynamic quantization, or half precision (`model.half()`) for storage |
| A server GPU with a latency goal | Half precision, `torch.compile`, an export runtime such as TensorRT (Chapter 14) |
| Smallest model at a given accuracy | Distillation, then quantization |

**Checklist after every shrink step:**
1. Re-measure **size** and **latency** on the target hardware, at the batch size you will use.
2. Re-evaluate **accuracy overall and per class** on the validation set (Chapter 10).
3. Compare **predictions**, not only scores: how many images changed their answer versus the original?
4. Test on a few **real inputs** from the target device (camera quality differs from your dataset).

---

## 13.8 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | No target set | Endless tweaking | Write size, latency and accuracy targets first |
> | Measured on the wrong hardware | Fast on your laptop, slow on the device | Measure on the target |
> | Pruned model has the same file size | Size unchanged after pruning | Unstructured pruning only adds zeros; use structured pruning plus a rebuild |
> | Accuracy collapses after quantization | Big drop on static quantization | Better calibration data, or QAT |
> | Dynamic quantization shows no gain | Little change on a CNN | It only handles `Linear` layers; use static or QAT for convolutions |
> | Fusing does nothing | No fused layers reported | Layers must be adjacent inside a `Sequential`, in `eval()` mode |
> | Quantized model errors on GPU | Backend error | Quantized int8 here is CPU only |
> | GPU timings too optimistic | Suspiciously fast | Synchronise before reading the clock; discard warmup runs |
> | Overall accuracy fine, one class broke | Per-class recall drops | Always check per-class metrics |
> | Compressed a huge model | Still too big | Start from a smaller architecture (rung 1) |

---

## Key takeaways

1. **Set a target** (size, latency, accuracy) and **measure on the deployment hardware**, with warmup and a fixed batch size.
2. Climb the ladder cheapest first: **smaller architecture or resolution, quantization, pruning, distillation**.
3. **Quantization** (8-bit) gives about 4x smaller files; dynamic handles `Linear` layers only, CNNs need static or QAT with fused layers.
4. **Pruning** alone rarely shrinks the file or speeds up inference; it needs fine-tuning and sparse-aware tools or a rebuild.
5. **Distillation** often gives the best small model.
6. After each step, recheck size, latency and **per-class** accuracy.

**Previous:** [Chapter 12: Interpretability](12_interpretability.md) &nbsp;|&nbsp; **Next:** [Chapter 14: Export and serving](14_deployment.md)
