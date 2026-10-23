# Chapter 10: Evaluation

| | |
|---|---|
| **Level** | Core |
| **You will learn** | How to measure a model honestly, choose the right metric, find *where and why* it fails, and know how much to trust the number |
| **Prerequisites** | Chapters 2, 6 |
| **Ready-made code** | `cv_toolkit/evaluate.py` |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

**In this chapter**
[10.1 The idea](#101-the-idea) · [10.2 Ground rules](#102-ground-rules) · [10.3 Accuracy](#103-accuracy-and-its-limits) · [10.4 Confusion matrix](#104-the-confusion-matrix) · [10.5 Precision, recall, F1](#105-precision-recall-and-f1) · [10.6 Choosing the metric](#106-choosing-the-metric) · [10.7 Thresholds](#107-thresholds-and-curves) · [10.8 Calibration](#108-calibration-can-you-trust-the-confidence) · [10.9 Error analysis](#109-error-analysis) · [10.10 Uncertainty](#1010-how-much-to-trust-the-number) · [10.11 Common problems](#1011-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 10.1 The idea

Training tells you the model learned *something*. Evaluation tells you **how good it is, for whom, and where it breaks**. One accuracy number is the start of that story, not the end. The most useful work here is looking at the mistakes.

---

## 10.2 Ground rules

1. **Evaluate on data the model never trained on**, and never used to make decisions. That is the **test set**, used once at the end (Chapter 2).
2. **`model.eval()` and `torch.no_grad()`** while evaluating (Chapter 6).
3. **Same preprocessing as training**, with no random augmentation.
4. **Compare against a baseline.** A score means nothing alone. Simplest baselines:
   - **Majority class:** always predict the most common class. If your model barely beats it, it has learned little.
   - **A simple model** (a small CNN, or logistic regression on pretrained features). If a fancy model beats a simple one by a point, the complexity may not be worth it.
   - **Human performance**, if you can measure it. It shows how much headroom exists, and how noisy the labels are.

First, collect all predictions in one pass; every metric below is computed from them.

```python
@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(model(x.to(device)).softmax(dim=1).cpu())
        labels.append(y)
    probs, labels = torch.cat(probs), torch.cat(labels)
    return probs, probs.argmax(dim=1), labels        # probabilities, predictions, true labels
```

---

## 10.3 Accuracy and its limits

**Accuracy** = correct predictions divided by all predictions.

```python
accuracy = (preds == labels).float().mean().item()
```

It is simple and fine when classes are roughly balanced and all mistakes cost the same. It misleads when:
- **Classes are imbalanced.** With 95% "normal", always saying "normal" scores 95%.
- **Mistakes are not equal.** Missing a disease is worse than a false alarm. Accuracy treats them the same.
- **You care about a rare class.** Accuracy is dominated by the common ones.

So always look at **per-class** results too.

---

## 10.4 The confusion matrix

A table: **rows = true class, columns = predicted class**. The diagonal is correct; everything else is a mistake, and its position tells you *which* mistake.

```python
num_classes = 3
cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
for t, p in zip(labels, preds):
    cm[t, p] += 1
```

| | Predicted cat | Predicted dog | Predicted bird |
|---|---|---|---|
| **True cat** | 46 | 3 | 4 |
| **True dog** | 3 | 65 | 7 |
| **True bird** | 6 | 7 | 59 |

Reading it: 7 dogs were called birds and 7 birds were called dogs. That is the pair to investigate. Plot it as a heatmap with class names on the axes, and normalise each row by its total so you see *rates*.

---

## 10.5 Precision, recall and F1

For one class, call it "positive" (say, *disease*):

| Term | Meaning |
|---|---|
| **True positive (TP)** | Said positive, was positive |
| **False positive (FP)** | Said positive, was negative (false alarm) |
| **False negative (FN)** | Said negative, was positive (a miss) |

| Metric | Formula | In plain words | High when |
|---|---|---|---|
| **Precision** | TP / (TP + FP) | "When it says *positive*, how often is it right?" | Few false alarms |
| **Recall** (sensitivity) | TP / (TP + FN) | "Of all the real positives, how many did it find?" | Few misses |
| **F1** | 2 x P x R / (P + R) | One number balancing both (harsh if either is low) | Both are good |

```python
tp = cm.diag().float()
precision = tp / cm.sum(dim=0).clamp(min=1)       # column sums = how many were PREDICTED as each class
recall    = tp / cm.sum(dim=1).clamp(min=1)       # row sums    = how many REALLY are each class
f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
```

**Combining classes into one number:**

| Average | How | Use when |
|---|---|---|
| **Macro** | Plain average over classes | Every class matters equally, including rare ones. **Good default for imbalance** |
| **Weighted** | Average weighted by class size | You care about overall performance on the real mix |
| **Micro** | Pool all decisions (equals accuracy for single-label problems) | Rarely informative on its own |

---

## 10.6 Choosing the metric

Choose by asking **"which mistake costs more?"** *before* you look at results.

| Situation | Care most about | Metric |
|---|---|---|
| Medical screening, safety defects: missing one is costly | Not missing positives | **Recall** (accept some false alarms) |
| Spam-style filters, automatic actions: false alarms are costly | Being right when you say yes | **Precision** |
| Both matter, imbalanced classes | Balance | **F1** or macro F1 |
| Balanced classes, equal costs | Overall correctness | Accuracy |
| Ranking quality, any threshold | Separation of classes | ROC-AUC or average precision (10.7) |

> [!TIP]
> **🟢 Extra: write the metric and the target down first** ("recall of at least 95% on class *defect*, with precision above 80%"). Otherwise it is tempting to pick whichever number came out best afterwards.

---

## 10.7 Thresholds and curves

For two-class problems the model outputs a probability, and you choose the **threshold** above which you say "positive". The default 0.5 is arbitrary.

```python
for threshold in (0.3, 0.5, 0.7):
    pred_pos = probs[:, 1] >= threshold
    tp = (pred_pos & (labels == 1)).sum().item()
    fp = (pred_pos & (labels == 0)).sum().item()
    fn = (~pred_pos & (labels == 1)).sum().item()
    print(threshold, "precision", tp / max(tp + fp, 1), "recall", tp / max(tp + fn, 1))
```

| Threshold | Effect |
|---|---|
| Lower (0.3) | More positives found: **recall up, precision down** |
| Higher (0.7) | Fewer, surer positives: **precision up, recall down** |

You are choosing a point on a trade-off. Pick it on the **validation set** to meet your target (for example "recall at least 95%"), then report the **test** result at that threshold.

**Curves that show every threshold at once:**
- **ROC curve and AUC:** true positive rate against false positive rate. 0.5 is chance, 1.0 is perfect. Can look flattering when positives are rare.
- **Precision-recall curve and average precision:** more honest for rare positives.

```python
from sklearn.metrics import roc_auc_score, average_precision_score
roc_auc_score(labels, probs[:, 1]); average_precision_score(labels, probs[:, 1])
```

---

## 10.8 Calibration: can you trust the confidence?

A model that says "90% sure" should be right about 90% of the time. Many networks are **overconfident**: "99% sure" and right only 85% of the time. This matters when someone acts on the confidence (for example, auto-approve above 95%).

Measure it by grouping predictions into confidence bins and comparing each bin's average confidence to its actual accuracy. The size of the gap, averaged, is the **expected calibration error (ECE)**.

```python
def expected_calibration_error(probs, labels, bins=10):
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    edges, ece = torch.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.float().mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)
```

> [!TIP]
> **🟢 Extra: fixes if it is poor.** *Temperature scaling*: divide the logits by one number, fitted on validation data, before the softmax (it changes confidence but not the predicted class). Label smoothing (Chapter 8) and ensembles also help. If you never act on the confidence value, calibration matters less.

---

## 10.9 Error analysis

> [!TIP]
> **🟢 Extra: the highest-value habit in this chapter.** Numbers say *how much* the model fails. Looking at the failures says *why*, and what to do about it.

Do this after every meaningful model:

**1. Most confused pairs**

```python
off = cm.clone(); off.fill_diagonal_(0)
top = off.flatten().topk(3)
pairs = [(int(i) // num_classes, int(i) % num_classes, int(v)) for v, i in zip(top.values, top.indices)]
# (true class, predicted class, count): translate indices to names with your class list
```

**2. Look at the wrong ones, as pictures.** Show 20 to 30 misclassified images with true and predicted labels. Ask: are they genuinely ambiguous? Mislabelled? Low quality? A systematic pattern?

**3. Look at the most confident wrong ones.** These are the most informative:

```python
conf, _ = probs.max(dim=1)
wrong = (preds != labels).nonzero().flatten()
worst = wrong[conf[wrong].argsort(descending=True)][:20]      # indices of the 20 most confident mistakes
```

Very confident mistakes are often **wrong labels in your data**, not model errors. Fixing labels can beat any model change.

**4. Slice by subgroup.** Compute accuracy separately for each source: camera, site, hospital, lighting, time of day, image size. A good overall score can hide a group where the model fails badly.

```python
for g in groups.unique():
    mask = groups == g
    print(g.item(), (preds[mask] == labels[mask]).float().mean().item(), mask.sum().item())
```

**5. Check it is right for the right reason** (Chapter 12): heatmaps on a few correct predictions show whether the model looks at the object or at a shortcut such as a watermark.

| What you find | What to do |
|---|---|
| Two classes truly look alike | More discriminating data; higher resolution; merge classes if the distinction does not matter |
| Wrong labels | Fix them and retrain |
| Failure on one subgroup | Collect more of that group; check for a systematic difference |
| Poor image quality | Filter or improve capture |
| Model uses a shortcut | Remove or vary the shortcut; augment |

---

## 10.10 How much to trust the number

A test score is an *estimate* with uncertainty. For accuracy `p` measured on `n` test images, a rough 95% margin is:

`margin ≈ 1.96 x sqrt(p x (1 - p) / n)`

| Test images (`n`) | Margin at 85% accuracy |
|---|---|
| 100 | about plus or minus 7 points |
| 400 | about plus or minus 4.5 points |
| 1,000 | about plus or minus 2.2 points |
| 10,000 | about plus or minus 0.7 points |

```python
import math
margin = 1.96 * math.sqrt(accuracy * (1 - accuracy) / len(labels))
```

**What this means:** with 200 test images, a model at 85% and one at 88% are **not distinguishable**. Do not announce an improvement smaller than the margin. For a result not covered by the formula (F1, per-class recall), use a **bootstrap**: resample the test predictions with replacement a thousand times, compute the metric each time, and report the middle 95% of values.

```python
idx = torch.randint(0, n, (1000, n))
scores = (preds[idx] == labels[idx]).float().mean(dim=1)
low, high = scores.quantile(0.025).item(), scores.quantile(0.975).item()
```

The rare classes have the widest uncertainty: 10 test images of a class means one image moves its recall by 10 points.

> [!NOTE]
> **🔵 Advanced alternative: use maintained metric libraries.**
>
> ```python
> # pip install torchmetrics
> from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
> f1 = MulticlassF1Score(num_classes=3, average="macro")(probs, labels)
>
> # pip install scikit-learn
> from sklearn.metrics import classification_report
> print(classification_report(labels, preds, target_names=class_names))
> ```
>
> **Why teams do it:** tested implementations, many more metrics, and consistent edge-case handling (for example a class with no predictions). Writing the maths by hand once is good for understanding; use the library in real work.

---

## 10.11 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Accuracy on imbalanced data | High accuracy, near-zero recall on a rare class | Per-class metrics, macro F1 |
> | Evaluating in train mode | Scores differ between identical runs | `model.eval()` |
> | Augmentation at evaluation | Noisy, lower scores | Evaluation transform only |
> | Test set used for decisions | Optimistic final number | Tune on validation; test once |
> | Chasing tiny differences | Gains vanish with another seed or sample | Compare to the margin (10.10) |
> | No baseline | Number with no meaning | Majority class and a simple model |
> | Metric chosen after seeing results | The "best" metric is quietly picked | Decide the metric and target first |
> | Overall score hides a weak subgroup | Fails in deployment on one site or camera | Slice by subgroup |
> | Overconfident probabilities | "99% sure" predictions are often wrong | Check calibration; temperature scaling |
> | Class order mismatch | Metrics fine, names wrong | Save class names with the model |
> | Never looking at mistakes | Stuck at the same score | Error analysis (10.9) |

---

## Key takeaways

1. **Never trust one number.** Look at the confusion matrix and per-class precision, recall and F1.
2. **Pick the metric by the cost of mistakes**, and write the target down first. Compare to a baseline.
3. For two classes, the **threshold** is a dial between precision and recall: set it on validation data.
4. **Error analysis beats guessing.** Look at the wrong and the confidently wrong images, and slice by subgroup.
5. A test score has **uncertainty**: about plus or minus 4.5 points at 400 test images and 85% accuracy. Do not claim gains smaller than that.
6. Touch the test set once.

**Previous:** [Chapter 9: Hyperparameter tuning](09_tuning.md) &nbsp;|&nbsp; **Next:** [Chapter 11: Embeddings and visual search](11_embeddings.md)
