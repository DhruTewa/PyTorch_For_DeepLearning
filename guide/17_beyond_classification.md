# Chapter 17: Beyond Classification (A Map)

| | |
|---|---|
| **Level** | Orientation |
| **You will learn** | Which computer vision task you actually have, what changes for each, and where to look next |
| **Prerequisites** | Chapters 1 to 10 |
| **Ready-made code** | None: this chapter is a map, not a tutorial |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

> [!TIP]
> **🟢 Extra: this whole chapter.** This guide is mostly about image classification. Many real projects are a different task. This chapter helps you recognise which one you have, and what carries over.

**In this chapter**
[17.1 The idea](#171-the-idea) · [17.2 The task map](#172-the-task-map) · [17.3 What carries over](#173-what-carries-over-from-this-guide) · [17.4 Multi-label](#174-multi-label-classification) · [17.5 Detection](#175-object-detection) · [17.6 Segmentation](#176-segmentation) · [17.7 Other tasks](#177-other-tasks) · [17.8 Choosing](#178-choosing-your-task) · [17.9 Common problems](#179-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 17.1 The idea

"One label per image" is only one question you can ask of a picture. Before building anything, state **exactly what the output should be**: a label, several labels, boxes, outlines, a number? The task decides the data you must collect, the model head, the loss and the metric. Choosing the wrong task is more expensive than choosing the wrong model.

---

## 17.2 The task map

| Task | Question | Output | Label cost |
|---|---|---|---|
| **Classification** | What is this image? | One class | Low (one label per image) |
| **Multi-label classification** | Which of these things are present? | A set of classes | Low to medium |
| **Object detection** | What is where? | Boxes with classes and scores | Medium to high (draw a box per object) |
| **Semantic segmentation** | Which class is each pixel? | A class per pixel | High (outline every region) |
| **Instance segmentation** | Which pixels belong to *each object*? | A mask per object | Highest |
| **Regression on images** | How much? (age, size, count) | A number | Depends |
| **Anomaly detection** | Is this unusual? | A score | Often needs no defect labels |
| **Similarity and retrieval** | What looks like this? | Neighbours | Low (Chapter 11) |
| **Keypoints, pose, OCR, depth, tracking** | Specialised questions | Points, text, depth maps, tracks | Varies |

**Rule of thumb:** use the *simplest* task that answers the business question. If "is there a defect in this image?" is enough, classification beats detection: far cheaper to label and to maintain. Move up only when you need the location or the count.

---

## 17.3 What carries over from this guide

Almost everything:

| From this guide | Still applies |
|---|---|
| Project skeleton, config, seeds, tracking (Chapter 0) | Fully |
| The training recipe and loop (Chapters 1, 6) | Same five steps; only the loss and the batch contents differ |
| Data checks, honest splits, leakage (Chapter 2) | Fully, and grouped splits matter even more |
| Transfer learning (Chapter 5) | The **backbone** is the same; you change the **head** |
| Overfitting, tuning (Chapters 8, 9) | Fully |
| Evaluation mindset, error analysis (Chapter 10) | Same discipline; different metrics (below) |
| Interpretability, shrinking, export, testing, monitoring (Chapters 12 to 16) | Fully, with task-specific details |

**What changes:** the **head**, the **loss**, the **labels format**, the **augmentations** (they must transform boxes and masks along with the image) and the **metrics**.

---

## 17.4 Multi-label classification

An image can contain several things at once (street, car, person). Each class becomes an independent yes/no.

| Change | From single-label | To multi-label |
|---|---|---|
| Last layer | `num_classes` outputs, softmax | `num_classes` outputs, **sigmoid** per class |
| Loss | `CrossEntropyLoss` | `BCEWithLogitsLoss` (targets are 0/1 vectors) |
| Prediction | `argmax` | Threshold each class, for example 0.5 |
| Metrics | Accuracy | Per-class precision, recall, F1; average precision |

```python
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(logits, targets)                 # targets: float tensor of 0s and 1s, shape [batch, num_classes]
preds = (logits.sigmoid() > 0.5).int()          # tune the threshold per class on validation data
```

Imbalance is usual (most labels are 0): use `pos_weight` in the loss and choose thresholds per class.

---

## 17.5 Object detection

**Output:** for each object, a box `(x1, y1, x2, y2)`, a class and a confidence.

| Piece | What to know |
|---|---|
| **Models** | `torchvision` provides Faster R-CNN, RetinaNet and SSD-style detectors with pretrained backbones (`torchvision.models.detection`). Modern popular toolkits (for example Ultralytics YOLO, MMDetection, Detectron2) are commonly used in practice |
| **Labels** | A box and a class per object, in a standard format (COCO JSON, YOLO text files, Pascal VOC XML). Annotation tools: CVAT, Label Studio, Roboflow |
| **Transfer learning** | Start from a detector pretrained on a large set (such as COCO) and replace the classification head for your classes |
| **Augmentation** | Must move the boxes with the image: use `torchvision.transforms.v2`, which handles boxes |
| **Post-processing** | **Non-maximum suppression (NMS)** removes duplicate boxes for the same object |
| **Metric** | **IoU** (overlap between predicted and true box), and **mAP** (mean average precision) across IoU thresholds |

```python
def iou(a, b):                                  # boxes as (x1, y1, x2, y2)
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union
```

A detection counts as correct when its IoU with a true box passes a threshold (0.5 is common) **and** the class matches.

> [!WARNING]
> **Watch out.** Small objects are hard (resolution matters a lot). Box labels must be **consistent**: inconsistent boxes (tight versus loose) cap your achievable score. Evaluate by object size and class, not only overall mAP.

---

## 17.6 Segmentation

**Output:** a class for every pixel (semantic), or a separate mask per object (instance).

| Piece | What to know |
|---|---|
| **Models** | Encoder-decoder networks such as U-Net (very common in medical and industrial work), DeepLab, FCN (`torchvision.models.segmentation` has several) |
| **Labels** | Pixel masks, produced with a polygon or brush tool. Expensive: budget for it |
| **Loss** | Per-pixel cross-entropy, often combined with **Dice loss** when the object of interest is small |
| **Augmentation** | Apply the **same** geometric transform to the image and its mask (and never interpolate mask labels smoothly) |
| **Metric** | **IoU** per class and **mean IoU**, or the **Dice score** |

```python
def mask_iou(pred, true):                       # boolean masks of the same shape
    return ((pred & true).sum() / (pred | true).sum().clamp(min=1)).item()
```

Class imbalance is severe (background dominates), so per-class IoU is essential: pixel accuracy is as misleading as accuracy on imbalanced classes (Chapter 10).

---

## 17.7 Other tasks

| Task | Starting point |
|---|---|
| **Regression** (predict a number from an image) | Same as classification with a single output and `MSELoss` or `L1Loss`; scale the target (Chapter 1); evaluate with mean absolute error |
| **Anomaly detection** | When defects are rare or unknown: learn what normal looks like from normal images only, and flag what differs. Embedding distance (Chapter 11) is a strong simple baseline; dedicated methods such as PatchCore exist |
| **OCR (reading text)** | Use a maintained OCR engine or document model rather than training your own |
| **Keypoints and pose** | Models that output point coordinates or heatmaps; ready-made pose models exist |
| **Depth, optical flow, tracking, video** | Specialised pretrained models and libraries; start from those |
| **Image generation and editing** | Generative models (diffusion): covered briefly in Appendix A |

---

## 17.8 Choosing your task

Ask in this order:

1. **What decision will the output drive?** Write it in one sentence.
2. **Is a whole-image answer enough?** Then classification (or multi-label). Stop here if yes.
3. **Do you need to know *where*, or *how many*?** Detection.
4. **Do you need the exact shape or area?** Segmentation.
5. **Is the thing you want rare or undefined?** Anomaly detection, or collect examples first.
6. **How many labelled examples can you afford?** Labelling cost grows steeply from classification to segmentation; a smaller, cheaper task that works beats an ambitious one that stalls.
7. **Is there a pretrained model close to your need?** Fine-tune it (Chapter 5's idea applies to every task).

---

## 17.9 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Over-ambitious task | Labelling never finishes; model weak | Step down to the simplest task that answers the need |
> | Softmax used for multi-label | Labels compete, only one gets high probability | Sigmoid per class and `BCEWithLogitsLoss` |
> | Augmentation moves the image but not the boxes or masks | Misaligned labels, falling accuracy | Use `transforms.v2`; transform image and target together |
> | Inconsistent annotation | Metric plateaus; errors look arbitrary | Write labelling guidelines; review a sample |
> | Judging detection by overall score | Small or rare objects fail unnoticed | Break down by size and class |
> | Pixel accuracy for segmentation | High score, small objects missed | Per-class IoU or Dice |
> | Training a detector from scratch | Needs huge data | Start from a pretrained detector |
> | Duplicate boxes | Several boxes on one object | Apply NMS and tune its threshold |

---

## Key takeaways

1. **Define the output first**: label, labels, boxes, masks, or a number. The task drives data, model head, loss and metric.
2. Use the **simplest task** that answers the question. Labelling cost rises steeply with complexity.
3. Most of this guide carries over; the **head, loss, label format, augmentation and metrics** change.
4. Multi-label: sigmoid and `BCEWithLogitsLoss`. Detection: boxes, NMS, mAP. Segmentation: pixel masks, IoU and Dice.
5. For any task, start from a **pretrained** model and fine-tune.

**Previous:** [Chapter 16: Monitoring and maintenance](16_monitoring.md) &nbsp;|&nbsp; **Next:** [Chapter 18: Modern backbones](18_modern_backbones.md)
