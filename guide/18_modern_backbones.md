# Chapter 18: Modern Backbones

| | |
|---|---|
| **Level** | Intermediate to advanced |
| **You will learn** | What the main families of pretrained vision models are, how to choose between them, and three ways to use one: frozen features, fine-tuning, and zero-shot |
| **Prerequisites** | Chapters 4, 5, 11 |
| **Ready-made code** | `cv_toolkit/models.py` (`build_classifier` accepts any torchvision name) |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

> [!TIP]
> **🟢 Extra: this whole chapter.** Older tutorials use ResNet and MobileNet. The field has moved on, and the pretrained model you start from often matters more than anything you tune afterwards.

> [!NOTE]
> This area changes quickly. The families and ideas below are stable; **specific model names and "best" rankings are not**. Check current benchmarks and model hubs before committing to one.

**In this chapter**
[18.1 The idea](#181-the-idea) · [18.2 The families](#182-the-main-families) · [18.3 Three ways to use one](#183-three-ways-to-use-a-pretrained-model) · [18.4 Frozen features](#184-frozen-features-and-a-linear-probe) · [18.5 Fine-tuning](#185-fine-tuning-transformers) · [18.6 Zero-shot](#186-zero-shot-with-image-text-models) · [18.7 Choosing](#187-choosing) · [18.8 Where to get models](#188-where-to-get-models) · [18.9 Common problems](#189-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 18.1 The idea

A **backbone** is the part of a vision model that turns an image into features. Better backbones, pretrained on more data in smarter ways, give better features. Swapping your backbone is often the cheapest big gain in a project, and Chapter 5's recipe (replace the head, freeze or fine-tune) works unchanged.

---

## 18.2 The main families

| Family | In plain words | Strengths | Watch for |
|---|---|---|---|
| **ResNet** (and similar CNNs) | Stacked convolutions with skip connections | Simple, robust, fast, well supported everywhere | Not the strongest features today |
| **EfficientNet, MobileNet** | CNNs designed for accuracy per compute | Small and fast; good for edge | Fiddly to train from scratch |
| **ConvNeXt** | A CNN modernised with ideas borrowed from transformers | Strong accuracy with CNN simplicity | Heavier than mobile designs |
| **Vision Transformer (ViT)** | Cuts the image into patches (for example 16x16 pixels) and treats them like words in a sentence, using attention | Excellent features when pretrained on large data; scales well | Needs lots of pretraining data; torchvision's version expects a fixed input size; heavier; learning rates must be lower |
| **Swin and other hierarchical transformers** | Transformers with CNN-like multi-scale structure | Strong for detection and segmentation | More complex |
| **CLIP and similar image-text models** | Trained on huge numbers of (image, caption) pairs so images and text share one embedding space | **Zero-shot** classification, strong general features, text search | Biases from web data; check the licence |
| **DINO / DINOv2 and other self-supervised models** | Learn features from images alone, with no labels | **Very strong frozen features** for retrieval, clustering and linear probes | Licence and model size vary |

**The big shift:** large models pretrained on enormous datasets, with or without labels, produce features so good that a **simple classifier on frozen features** often beats a carefully trained smaller network, especially with little data.

---

## 18.3 Three ways to use a pretrained model

| Way | What you do | Data needed | Cost | Typical when |
|---|---|---|---|---|
| **Frozen features + simple classifier** ("linear probe") | Extract embeddings once; train logistic regression or a small head | Tens to thousands of images | Minutes, CPU is fine | **First thing to try.** Great baseline and often enough |
| **Fine-tuning** | Update some or all weights with a small learning rate | Hundreds to many thousands | Hours, GPU | The probe is not accurate enough, and your images differ from the pretraining data |
| **Zero-shot** | Describe the classes in words; the model matches images to text | **None** | Seconds | No labelled data, or a quick first look, or classes that change often |

---

## 18.4 Frozen features and a linear probe

Extract features once, then train a tiny model on them.

```python
backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
backbone.heads = nn.Identity()                 # drop the classifier; output is 768 numbers per image
backbone.eval()

@torch.no_grad()
def extract(loader):
    feats, labels = [], []
    for x, y in loader:
        feats.append(backbone(x.to(device)).cpu()); labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

Xtr, ytr = extract(train_loader)               # use the model's OWN preprocessing (Chapter 5)
Xva, yva = extract(val_loader)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print(clf.score(Xva, yva))
```

**Why this is so good as a first step:**
- **Fast**: the expensive pass over the images happens once, and training the classifier takes seconds. Trying five different backbones is an afternoon.
- **Cannot overfit the backbone**: it is frozen.
- **A strong baseline** to beat. If fine-tuning gains only a point, it may not be worth the complexity.
- No augmentation is possible on cached features, which is the cost; for small datasets this is usually fine.

**Remove the classification head differently per family** (Chapter 5's table): ResNet `fc = nn.Identity()`, ConvNeXt and MobileNet `classifier[-1] = nn.Identity()`, ViT `heads = nn.Identity()`.

---

## 18.5 Fine-tuning transformers

Transformers are more sensitive than CNNs. Rules that help:

| Setting | Guidance |
|---|---|
| **Learning rate** | Much lower than for CNN heads: 1e-5 to 5e-5 for the pretrained layers, 1e-3 for a new head |
| **Layer-wise learning rates** | Smaller rates for early layers, larger for later ones; unfreeze gradually (Chapter 5) |
| **Warmup** | A short linear warmup (Chapter 6) prevents early instability |
| **Optimizer** | `AdamW` with weight decay 0.01 to 0.1 |
| **Input size** | Match what the weights were trained on (torchvision's ViT needs exactly 224x224 unless you change its position embeddings) |
| **Regularisation** | Label smoothing, mixup/CutMix, and stochastic depth are standard in published recipes (Chapter 8) |
| **Memory** | Heavier than CNNs: use mixed precision and gradient accumulation (Chapter 7) |
| **Data** | Transformers are data-hungry from scratch, but fine-tuning from a good pretrained model works with modest data |

> [!WARNING]
> **Watch out: LayerNorm and BatchNorm.** Transformers use LayerNorm, which has no running averages, so the BatchNorm "frozen but still drifting" issue (Chapter 5) does not apply to them. It *does* apply to CNN backbones in the same project.

---

## 18.6 Zero-shot with image-text models

> [!CAUTION]
> **Not tested.** The `open_clip` snippet was written from documentation knowledge and not run. Model names and pretrained tags change between releases.

CLIP-style models put images and text into the same embedding space. To classify with **no training images**, describe each class in words, embed those descriptions, and pick the nearest one to the image.

```python
# pip install open_clip_torch      (one of several libraries that provide CLIP-style models)
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

prompts = ["a photo of a rotten apple", "a photo of a fresh apple"]
with torch.no_grad():
    image_feat = F.normalize(model.encode_image(preprocess(image).unsqueeze(0)), dim=1)
    text_feat = F.normalize(model.encode_text(tokenizer(prompts)), dim=1)
    probs = (100.0 * image_feat @ text_feat.T).softmax(dim=1)         # one probability per prompt
```

**The idea in the last line:** cosine similarity between the image embedding and each text embedding, sharpened by a temperature factor (100), then softmax.

| Tip | Why |
|---|---|
| **Prompt wording matters** | "a photo of a ..." usually beats the bare class name. Try several phrasings and average them |
| **Check it on labelled data** | Zero-shot accuracy is unknown until you measure it (Chapter 10) |
| **Weak on fine, specialised distinctions** | Medical images, defects, rare species. Linear probe or fine-tuning is usually much better there |
| **Use it to bootstrap** | Auto-label a pile of images, then correct by hand, which is far faster than labelling from nothing |

> [!NOTE]
> **🔵 Advanced alternative: image-text embeddings for search.**
>
> **What changes:** the same text and image embeddings power **text-to-image search** ("red car at night") and image-to-image search in one index (Chapter 11).
>
> **Why teams do it:** users type what they want instead of providing an example image, and you need no labels at all.

> [!WARNING]
> **Watch out: bias and licences.** Models trained on web data inherit its biases and gaps; evaluate on *your* population and subgroups (Chapter 10). Pretrained weights come with licences, and some restrict commercial use. Check the licence of **both the code and the weights** before shipping.

---

## 18.7 Choosing

| Situation | Start with |
|---|---|
| Tiny dataset (tens to a few hundred images) | Frozen features from a strong pretrained model, plus logistic regression |
| No labelled data at all | Zero-shot with an image-text model, then label and correct |
| Moderate dataset, photos of everyday things | Fine-tune a ConvNeXt or an efficient CNN; compare with a frozen-feature baseline |
| Large dataset, accuracy is key | Fine-tune a larger transformer or ConvNeXt with a modern recipe |
| Edge device or tight latency | A mobile CNN, possibly distilled from a larger model (Chapter 13) |
| Similarity search or clustering | Frozen embeddings from a self-supervised or image-text model (Chapter 11) |
| Very different domain (medical, satellite, microscopy) | Try domain-specific pretrained models if they exist; otherwise fine-tune and expect to need more care |

**Workflow that wastes the least time:**
1. Frozen features from two or three backbones plus logistic regression (an afternoon).
2. Pick the best; **fine-tune** only if the gap to your target justifies it.
3. Compare cost: accuracy per millisecond and per megabyte matters in production (Chapter 13).

---

## 18.8 Where to get models

> [!CAUTION]
> **Not tested.** The `timm` snippet was not run. Check model names against the library's current list.

| Source | What it offers |
|---|---|
| **`torchvision.models`** | ResNet, EfficientNet, MobileNet, ConvNeXt, ViT, Swin and detection/segmentation models, with a consistent API and weights |
| **`timm` (PyTorch Image Models)** | A very large collection of backbones with pretrained weights and a uniform `create_model` call |
| **`open_clip`** | CLIP-style image-text models |
| **Hugging Face Hub (`transformers`)** | Many vision, image-text and self-supervised models, including DINOv2-style features |

```python
# pip install timm
import timm
model = timm.create_model("convnext_tiny", pretrained=True, num_classes=5)    # head replaced for you
cfg = timm.data.resolve_data_config({}, model=model)                          # the right size and normalisation
transform = timm.data.create_transform(**cfg)
```

> [!WARNING]
> **Watch out: each model has its own preprocessing** (size, mean, std, even channel order). Always use the transform shipped with the weights (`weights.transforms()` in torchvision, the config above in timm, `preprocess` in open_clip). A mismatch fails silently (Chapter 5).

---

## 18.9 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Wrong preprocessing for the chosen model | Weak results from a "strong" model | Use the weights' own transform |
> | ViT given a different image size | Shape error | Resize to the expected size, or adapt position embeddings |
> | Fine-tuning with a CNN-style learning rate | Accuracy collapses in early epochs | 1e-5 to 5e-5, warmup, AdamW |
> | Skipped the frozen-feature baseline | Weeks on fine-tuning for a one-point gain | Always run the linear probe first |
> | Zero-shot trusted without measurement | Confident but wrong on your domain | Evaluate on labelled samples |
> | Features extracted in `train()` mode | Features differ between runs | `model.eval()` and `no_grad` |
> | Licence ignored | Legal problem at launch | Check code and weights licences early |
> | Biases not examined | A subgroup performs far worse | Slice results by subgroup (Chapter 10) |
> | Model too heavy to deploy | Latency or memory targets missed | Smaller backbone, distillation, quantization (Chapter 13) |

---

## Key takeaways

1. Backbone families: **CNNs** (ResNet, EfficientNet, MobileNet, ConvNeXt), **Vision Transformers**, **image-text models** (CLIP), and **self-supervised models** (DINO-style). Names change fast; the ideas are stable.
2. Three uses: **frozen features plus a linear probe** (try first), **fine-tuning**, and **zero-shot**.
3. Frozen features from a strong model often beat a smaller network you train yourself, and let you compare backbones in an afternoon.
4. Transformers need **lower learning rates**, warmup and the right input size.
5. Use each model's **own preprocessing**, check **licences**, and evaluate on your data and subgroups.

**Previous:** [Chapter 17: Beyond classification](17_beyond_classification.md) &nbsp;|&nbsp; **Back to:** [Chapter 0: Project skeleton](00_project_skeleton.md)
