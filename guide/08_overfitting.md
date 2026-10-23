# Chapter 8: Beating Overfitting

| | |
|---|---|
| **Level** | Core |
| **You will learn** | How to recognise overfitting and underfitting, and which fixes to try, in what order |
| **Prerequisites** | Chapters 2, 3, 4, 6 |
| **Ready-made code** | Spread across `data.py` (augmentation), `models.py` (dropout, freezing), `train.py` (early stopping) |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

**In this chapter**
[8.1 The idea](#81-the-idea) · [8.2 Diagnose first](#82-diagnose-first) · [8.3 The fixes, in order](#83-the-fixes-in-the-order-to-try-them) · [8.4 Underfitting](#84-when-the-opposite-happens-underfitting) · [8.5 A strategy](#85-a-strategy-that-works) · [8.6 Common problems](#86-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 8.1 The idea

**Overfitting** is memorising instead of learning: the model gets very good at the training images, including their quirks and noise, and does worse on new ones. Like a student who memorises last year's exam answers instead of understanding the subject.

**Underfitting** is the opposite: the model is too simple, or trained too little, to learn the pattern even on the training images.

This chapter is a *cross-reference*: the tools mostly live in other chapters. Here they are organised by the question "what do I do when validation falls behind training?"

---

## 8.2 Diagnose first

Compare the training and validation numbers at the point where validation is best.

| Training score | Validation score | Diagnosis |
|---|---|---|
| High | **Much lower** (a gap of 10 points or more is a red flag) | **Overfitting** |
| Low | Low | **Underfitting** (or a bug: run the Chapter 7 checks) |
| High | High, close together | Good |
| Lower than validation, persistently | | Check for leakage (Chapter 2) or an unusually easy validation set |

Look at the **loss curves** too (Chapter 9, section 7.3): training loss still falling while validation loss turns upward is the classic overfitting signature, and it tells you *when* it began (that epoch is roughly where early stopping would have saved you).

> [!WARNING]
> **Watch out: a small validation set makes everything noisy.** With 100 validation images, one image is 1 point of accuracy. Do not chase differences smaller than the noise (see Chapter 10 on how large the uncertainty is).

---

## 8.3 The fixes, in the order to try them

Ordered by how much they usually help for the effort. Change **one thing at a time** so you know what worked.

### 1. More and better data

The most reliable cure. Even a few hundred more varied images beat clever tricks.
- Add images of the cases the model gets wrong (Chapter 10's error analysis tells you which).
- Remove wrong labels and near-duplicates (Chapter 2).
- Collect variety (lighting, angles, backgrounds, devices), not more of the same.

### 2. Data augmentation

Show the model many versions of each image so it cannot memorise exact pixels (details in Chapter 3).

| Typical starting set | Notes |
|---|---|
| Random horizontal flip, small rotation (about 10 degrees), brightness and contrast jitter | Only augmentations that keep the label true |
| Random resized crop | Strong, good for object photos |

Stronger augmentation means slower training and a larger training-vs-validation gap *on purpose*: training accuracy drops, validation accuracy often improves.

### 3. Use a pretrained model

A network that has already seen a million photos has much less room to memorise your small dataset (Chapter 5). If you are training from scratch on a small dataset, this is often the single biggest improvement.

### 4. Make the model smaller, or freeze part of it

Fewer free numbers means less capacity to memorise. Reduce channels or the size of the dense layer (Chapter 4), or freeze the pretrained backbone and train only the head (Chapter 5).

### 5. Regularisation: dropout and weight decay

**Dropout** randomly switches off a share of units during training, so the network cannot depend on any one of them.

```python
nn.Dropout(p=0.5)        # in the dense head; p between 0.2 and 0.5
nn.Dropout2d(p=0.1)      # optional, inside conv blocks: drops whole channels; keep p small
```

**Weight decay** gently shrinks the weights toward zero, so the model prefers simple explanations.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)   # try 1e-4 to 1e-1
```

| Setting | If overfitting | If underfitting |
|---|---|---|
| Dropout `p` | Raise (0.3 to 0.5) | Lower or remove |
| Weight decay | Raise (1e-2 to 1e-1 with AdamW) | Lower (1e-4) or zero |

> [!NOTE]
> **🔵 Advanced alternative: no decay on biases and normalisation layers.**
>
> ```python
> decay    = [p for p in model.parameters() if p.ndim > 1]    # weight matrices and conv kernels
> no_decay = [p for p in model.parameters() if p.ndim <= 1]   # biases, BatchNorm scale and shift
> optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-2},
>                                {"params": no_decay, "weight_decay": 0.0}], lr=1e-3)
> ```
>
> **Why teams do it:** shrinking biases and normalisation parameters gives no regularising benefit and can slightly hurt. It is a common refinement in published training recipes.

### 6. Early stopping and picking the best epoch

Stop when validation loss stops improving, and keep the weights from the best epoch (Chapter 8, section 6.3). It costs nothing and is always worth having.

### 7. Label smoothing

> [!TIP]
> **🟢 Extra.** Instead of telling the model "this is 100% a cat", say "90% cat, the rest spread over other classes". It stops the model becoming overconfident and often improves generalisation slightly.

```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)    # 0.05 to 0.1 is typical
```

### 8. Stronger augmentation: mixup, CutMix, random erasing

> [!NOTE]
> **🔵 Advanced alternative: blend or cut-and-paste training images.**
>
> **What changes:** *Mixup* blends two images and their labels. *CutMix* pastes a patch from one image onto another. *Random erasing* blanks a random rectangle. All are available in `torchvision.transforms.v2` (`MixUp`, `CutMix`, `RandomErasing`).
>
> **Why teams do it:** they regularise strongly and are staples of modern training recipes, especially for larger models and longer training.
>
> ```python
> def mixup(x, y, alpha=0.2):
>     lam = np.random.beta(alpha, alpha)
>     idx = torch.randperm(x.size(0))
>     return lam * x + (1 - lam) * x[idx], y, y[idx], lam
>
> xm, ya, yb, lam = mixup(x, y)
> logits = model(xm)
> loss = lam * F.cross_entropy(logits, ya) + (1 - lam) * F.cross_entropy(logits, yb)
> ```
>
> They slow convergence, so they pay off with enough epochs. They are a poor fit when exact fine detail is the label (tiny defects).

### 9. Averaging several predictions

> [!TIP]
> **🟢 Extra: two cheap boosts at prediction time.**
> - **Test-time augmentation (TTA):** predict on the image and a flipped copy, then average the probabilities.
> - **Ensemble:** average the probabilities of several models (different seeds or architectures).
>
> ```python
> @torch.no_grad()
> def predict_tta(model, x):
>     p = model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)
>     return p / 2
>
> probs = torch.stack([m(x).softmax(1) for m in models]).mean(0)      # ensemble of models in eval mode
> ```
>
> Both cost extra compute at prediction time, and only help when each prediction is individually decent and the errors differ.

---

## 8.4 When the opposite happens: underfitting

Training score itself is low. First run the Chapter 7 sanity checks (a bug looks identical). Then:

| Fix | Why |
|---|---|
| Train longer; check the loss was still falling when you stopped | The model simply needs more steps |
| Raise the learning rate, or use a better optimizer (AdamW) | Learning too slowly to get anywhere |
| Bigger model: more channels or layers, or a larger pretrained network | Not enough capacity |
| Remove or lower regularisation (dropout, weight decay, heavy augmentation) | You are holding the model back |
| Unfreeze more of a pretrained backbone (Chapter 5) | The frozen features do not fit your images well |
| Improve the inputs (resolution, preprocessing) | Details that matter may be lost, for example a small defect shrunk away |

---

## 8.5 A strategy that works

1. **Start simple and overfit on purpose.** First get a model that reaches near-perfect *training* accuracy (proves it has the capacity and the code works).
2. **Then close the gap**, in the order of 8.3: more data, augmentation, pretrained weights, regularisation, early stopping.
3. **Change one thing at a time**, and compare on the *same* validation set.
4. **Repeat with two or three seeds** before believing a small gain. A 0.5 point change is often noise.
5. **Look at the errors** (Chapter 10). They tell you which fix to use: confusions between two similar classes need more discriminating data, not more dropout.
6. **Keep the test set sealed** until the end. If you tune on it, your final number is optimistic.

---

## 8.6 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Tuning on the test set | Great test score, disappointing in real use | Tune on validation only; test once |
> | Chasing noise | A 0.5 point gain does not repeat with another seed | Average several seeds; use a larger validation set |
> | Several changes at once | Score improved but you cannot say why | One change per experiment |
> | Too much regularisation | Training and validation scores both low | Lower dropout, weight decay, augmentation |
> | Augmentation breaking the label | Accuracy collapses with stronger augmentation | Review which transforms preserve the label (Chapter 3) |
> | Dropout left on at evaluation | Scores noisy and lower than expected | `model.eval()` (Chapter 6) |
> | Validation set leaked into training | Validation beats training, or near-perfect scores | Check duplicates and groups (Chapter 2) |
> | Overfitting to the validation set | Many rounds of tuning, then a worse test score | Fewer tuning rounds; keep a final untouched test set |

---

## Key takeaways

1. Overfitting shows as **high training score, much lower validation score**. Diagnose with the gap and the loss curves.
2. Order of fixes: **more and better data, augmentation, a pretrained model, a smaller model, regularisation (dropout, weight decay), early stopping**.
3. Underfitting is the opposite: bigger model, longer training, less regularisation. Rule out a bug first.
4. **One change at a time**, same validation set, several seeds.
5. Never tune on the test set.

**Previous:** [Chapter 7: Training health and speed](07_training_health.md) &nbsp;|&nbsp; **Next:** [Chapter 9: Hyperparameter tuning](09_tuning.md)
