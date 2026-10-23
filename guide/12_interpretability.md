# Chapter 12: Interpretability

| | |
|---|---|
| **Level** | Intermediate |
| **You will learn** | How to see what a CNN is paying attention to, and how to tell whether it is right for the right reason |
| **Prerequisites** | Chapters 4, 5, 10 |
| **Ready-made code** | `cv_toolkit/interpret.py`, `cv_toolkit/viz.py` |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

**In this chapter**
[12.1 The idea](#121-the-idea) · [12.2 The three views](#122-the-three-views) · [12.3 Hooks](#123-forward-hooks-and-feature-maps) · [12.4 Saliency](#124-saliency-which-pixels-matter) · [12.5 CAM and Grad-CAM](#125-cam-and-grad-cam-which-regions-matter) · [12.6 Showing it](#126-drawing-the-heatmap) · [12.7 Reading heatmaps](#127-how-to-read-heatmaps) · [12.8 Common problems](#128-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 12.1 The idea

A model can score 98% **for the wrong reason**: a ruler beside the tumour photos, a hospital logo on one class, a tag only present on rotten fruit, a grass background for "cow". It then fails the moment the shortcut disappears. Accuracy cannot reveal this; looking at *what the model looks at* can.

**Interpretability tools answer: "which parts of this image drove that prediction?"** Use them *before* you trust a good accuracy number, and again when the model fails.

---

## 12.2 The three views

| View | Question it answers | Speed | Output |
|---|---|---|---|
| **Feature maps** | What patterns does each layer respond to? | Fast | Pictures per layer (edges early, shapes later) |
| **Saliency** | Which individual **pixels**, if changed slightly, would change the answer? | Fast | Noisy, speckled map |
| **CAM / Grad-CAM** | Which **regions** of the image support the class? | Fast | Smooth heatmap, the most useful of the three |

---

## 12.3 Forward hooks and feature maps

A **forward hook** is a "wiretap" you attach to a layer: whenever data passes through, it copies the layer's output.

```python
activations = {}

def make_hook(name):
    def hook(module, inputs, output):
        activations[name] = output.detach()        # keep a copy, no gradient needed
    return hook

handles = [model.layer1.register_forward_hook(make_hook("layer1")),
           model.layer4.register_forward_hook(make_hook("layer4"))]
model.eval()
with torch.no_grad():
    model(image)                                   # image: [1, 3, H, W], preprocessed as in training
for h in handles:
    h.remove()                                     # ALWAYS remove hooks, or they leak memory and slow every later pass
```

`activations["layer1"]` has shape `[1, 64, 56, 56]`: 64 channels, each a 56x56 grid showing where one pattern was found.

**To pick layers:** list the candidates with `[n for n, _ in model.named_modules()]`. For ResNet, the first conv plus one point per stage (`layer1` to `layer4`) works well.

**What to expect:** early layers react to edges, colours and fine textures; late layers react to larger, more object-like structures. Show the single most active channel per layer as a row of heatmaps.

> [!WARNING]
> **Watch out: hooks.** Forgetting `handle.remove()` leaves the hook on the model. Use `try/finally` or a context manager that removes it automatically. Also, `ReLU(inplace=True)` right after a hooked layer can change the stored tensor afterwards: hook a layer followed by a non-in-place operation.

---

## 12.4 Saliency: which pixels matter

**In plain words:** nudge each input pixel slightly and see how much the class score moves. Pixels with a big effect are "salient". It uses the gradient of the class score with respect to the *input image*.

```python
def saliency(model, x, class_idx=None):
    model.eval()
    x = x.clone().requires_grad_(True)             # we need gradients with respect to the pixels
    out = model(x)
    class_idx = out.argmax(1).item() if class_idx is None else class_idx
    model.zero_grad()
    out[0, class_idx].backward()
    g = x.grad.abs().amax(dim=1)[0]                # strongest effect across colour channels: [H, W]
    return (g - g.min()) / (g.max() - g.min() + 1e-8)    # scale to 0..1
```

- Pass `class_idx` to explain a *specific* class ("why rotten?"), or leave it `None` to explain the model's own top answer.
- **No `torch.no_grad()`** here: gradients are the whole point.
- The map is fast but **noisy and speckled**. Good for a quick look; do not over-read individual bright pixels.

---

## 12.5 CAM and Grad-CAM: which regions matter

### CAM (Class Activation Map)

**In plain words:** a classifier's final decision is a weighted sum of "feature channels". Redo that sum *but keep the picture grid* instead of averaging it away, and you get a map of where the evidence for the class sits.

It works **only** for networks shaped *convolutional features, then global average pooling, then one `Linear` layer* (ResNet and many others).

```python
def cam(model, x, class_idx=None):
    model.eval()
    store = {}
    h = model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("f", o.detach()))
    with torch.no_grad():
        out = model(x)
    h.remove()
    class_idx = out.argmax(1).item() if class_idx is None else class_idx
    heat = torch.einsum("c,chw->hw", model.fc.weight[class_idx], store["f"][0])   # weight each channel by the class weights
    heat = F.relu(heat)                                                           # keep evidence FOR the class only
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return F.interpolate(heat[None, None], size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
```

> [!WARNING]
> **Watch out: which layer.** Hook the **whole last convolutional stage** (for ResNet, `model.layer4`), which is the real last feature map. Hooking an inner convolution inside the last block captures a value *before* the normalisation, residual addition and ReLU, so it is only a rough approximation of what the classifier actually uses.

### Grad-CAM

> [!TIP]
> **🟢 Extra: the general-purpose upgrade.** Grad-CAM works with **any** architecture, because it uses gradients instead of the last layer's weights: it weights each channel by how strongly the class score depends on it.

```python
def grad_cam(model, x, layer, class_idx=None):
    model.eval()
    st = {}
    def fwd(module, inputs, output):
        st["a"] = output
        output.register_hook(lambda g: st.__setitem__("g", g))     # capture the gradient flowing back
    h = layer.register_forward_hook(fwd)
    try:
        out = model(x)
        class_idx = out.argmax(1).item() if class_idx is None else class_idx
        model.zero_grad()
        out[0, class_idx].backward()
    finally:
        h.remove()
    weights = st["g"].mean(dim=(2, 3), keepdim=True)                # importance of each channel
    heat = F.relu((weights * st["a"]).sum(dim=1, keepdim=True)).detach()
    heat = F.interpolate(heat, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    return (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

heat = grad_cam(model, image, model.layer4, class_idx=1)            # e.g. "why class 1?"
```

**Choosing the layer:** the last convolutional stage is the standard choice: it has the best mix of "knows about objects" and "still has spatial layout". Earlier layers give finer but less meaningful maps.

> [!NOTE]
> **🔵 Advanced alternative: use a maintained library.**
>
> ```python
> # pip install grad-cam
> from pytorch_grad_cam import GradCAM
> cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
> heat = cam(input_tensor=image)[0]                      # [H, W] in 0..1
>
> # pip install captum
> from captum.attr import IntegratedGradients
> attr = IntegratedGradients(model).attribute(image, target=class_idx)
> ```
>
> **Why teams do it:** more methods (Score-CAM, Eigen-CAM, integrated gradients, SHAP-style attributions), tested across many architectures, and you can **cross-check** methods against each other. Agreement between two different methods is much stronger evidence than a single heatmap.

---

## 12.6 Drawing the heatmap

Overlay the map on the original image (undo normalisation first, or use the original file):

```python
import matplotlib.pyplot as plt

def overlay(image_hwc, heat, alpha=0.4, title=""):
    plt.imshow(image_hwc)                                  # image as a [H, W, 3] array in 0..1
    plt.imshow(heat.cpu(), cmap="jet", alpha=alpha)        # red = strong evidence
    plt.title(title); plt.axis("off")
```

To undo normalisation for display: `img = x[0].cpu() * std[:, None, None] + mean[:, None, None]`, then `img.permute(1, 2, 0).clamp(0, 1)`.

---

## 12.7 How to read heatmaps

Make a **grid** of examples rather than looking at one: correct predictions, wrong predictions, and each class.

| What you see | Meaning |
|---|---|
| Heat on the object or the defect (the mould spot, the lesion, the crack) | **Right reason** |
| Heat on the background, a border, a corner, a ruler, a label or watermark | **Shortcut.** The model has learned a cue that will not hold in real use |
| Heat spread over the whole image | Model has no focus: weak features or very small objects |
| Heat on the right object for the *wrong* class | The model sees the object but misjudges it: a data or class-definition problem |
| Different explanation methods disagree | Do not trust either alone. Cross-check |

**When you find a shortcut:** remove or vary it (crop, mask, add images where it is absent or present for the other class), augment against it, then retrain and look again.

> [!TIP]
> **🟢 Extra: a check that your explanation method is not fooling you.** Randomise the model's weights and recompute the heatmap. A trustworthy method must produce a **different, meaningless map**. If the heatmap of a *random* network still looks like a nice object outline, the method is mostly showing image edges, not what the model learned. Do this once when you adopt a method.

**Limits to keep in mind:**
- A heatmap shows where the evidence **is**, not that the model **understands** the object. It is evidence, not proof.
- Maps are coarse (7x7 resized up). They suggest regions, not exact boundaries.
- Be wary of confirmation bias: you will tend to see what you expect. Compare across many images and let a second person look.

---

## 12.8 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Hooks left attached | Slower and slower runs, memory growth | `handle.remove()` in `finally` or a context manager |
> | Model in `train()` mode | Heatmap changes on every call | `model.eval()` first |
> | `no_grad` around saliency or Grad-CAM | `element 0 of tensors does not require grad` error | Gradients must be on for these methods; not for plain CAM |
> | Wrong layer hooked for CAM | Map looks blurry or meaningless | Use the whole last convolutional stage |
> | Heatmap not aligned with the image | Heat offset or stretched | Use the same resized image for model and overlay |
> | Normalised image displayed | Odd colours | Undo normalisation before plotting |
> | Explaining the wrong class | Map seems unrelated to the question | Pass `class_idx` explicitly |
> | In-place ReLU after the hooked layer | Stored activations look odd | Hook a layer followed by pooling or a non-in-place op |
> | One pretty example taken as proof | Cherry-picking | Look at many, including failures |

---

## Key takeaways

1. High accuracy can hide a **shortcut** (a watermark, a background). Interpretability is how you catch it.
2. **Hooks** copy a layer's output. Always remove them.
3. **Saliency** = pixels (noisy). **CAM / Grad-CAM** = regions (the most useful). Use Grad-CAM for any architecture, on the last convolutional stage.
4. Check heatmaps over **many images, including mistakes**. Heat on the object is good; heat on the background or a label is a warning.
5. Cross-check with a second method, and remember a heatmap is evidence, not proof.

**Previous:** [Chapter 11: Embeddings and visual search](11_embeddings.md) &nbsp;|&nbsp; **Next:** [Chapter 13: Shrinking models](13_optimization.md)
