# Chapter 11: Embeddings and Visual Search

| | |
|---|---|
| **Level** | Intermediate |
| **You will learn** | What an embedding is, three ways to get one, how to train with triplet loss, and how to build and evaluate "find images like this one" |
| **Prerequisites** | Chapters 4, 5, 6 |
| **Ready-made code** | `cv_toolkit/models.py` (`EmbeddingNet`), `cv_toolkit/data.py` (`TripletDataset`), `cv_toolkit/train.py` (`train_triplet`), `cv_toolkit/evaluate.py` (`extract_embeddings`, `find_nearest`) |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

**In this chapter**
[11.1 The idea](#111-the-idea) · [11.2 Where embeddings come from](#112-where-embeddings-come-from) · [11.3 Triplet loss](#113-training-with-triplet-loss) · [11.4 Searching](#114-searching) · [11.5 Evaluating](#115-evaluating-a-search) · [11.6 Scaling up](#116-scaling-up) · [11.7 Common problems](#117-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 11.1 The idea

An **embedding** is a short list of numbers (say 128) that describes an image, arranged so that **similar images get similar lists**. "Similar" then becomes a simple distance between two lists of numbers.

That one idea powers many products:

| Use | How |
|---|---|
| **Visual search** ("more like this") | Find the closest embeddings to the query |
| **Duplicate and near-duplicate detection** | Very small distance |
| **Recommendations** | Items near the ones a user liked |
| **Few-shot classification** | Classify a new image by its nearest labelled examples, with no retraining |
| **Anomaly detection** | An image far from everything normal |
| **Clustering and exploring a dataset** | Group by embedding, plot in 2D |

A classifier answers "which of my fixed classes?" An embedding lets you answer "what is this *like*?", including for classes the model was never trained on.

---

## 11.2 Where embeddings come from

| Source | How | When |
|---|---|---|
| **A pretrained network's features** | Remove the final classification layer; use the output before it | **Always try first.** Strong, needs no training |
| **Your classifier's second-to-last layer** | Same idea, using your fine-tuned model | You have labels and want features tuned to your classes |
| **Trained for similarity** (metric learning) | Train directly so that same-class images are close and different-class images are far (triplet loss, 11.3) | Pretrained features are not good enough for *your* notion of similar |

> [!TIP]
> **🟢 Extra: the pretrained-features baseline.** It takes five lines and is often surprisingly good. Build it before any metric learning, and only train a custom embedding if you can show it beats this baseline.

```python
backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
backbone.fc = nn.Identity()          # drop the classifier: the output is now a 512-number embedding
backbone.eval()
with torch.no_grad():
    embedding = backbone(batch)      # shape: [batch, 512]
```

**Embedding size:** 64 to 512 numbers is typical. Smaller is faster and cheaper to store but loses detail; larger rarely helps beyond a point. Compare on your own search quality (11.5).

**Always normalise** to length 1 (`F.normalize(e, dim=1)`). Then distance depends on direction only and cosine similarity is a plain dot product.

---

## 11.3 Training with triplet loss

Show the network **three images at once**:

| Name | What it is |
|---|---|
| **Anchor** | Any image |
| **Positive** | Another image of the **same** class |
| **Negative** | An image of a **different** class |

The loss pushes the anchor closer to the positive than to the negative by at least a **margin**:

`loss = max(0, distance(anchor, positive) - distance(anchor, negative) + margin)`

When the positive is already clearly closer than the negative, the loss is zero and that triplet teaches nothing more.

**The network.** One network embeds all three images, with **shared weights** (this is what "Siamese" means: the same network applied to each input).

```python
class EmbeddingNet(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(512, dim)               # new head producing the embedding
    def forward(self, x):
        return F.normalize(self.backbone(x), dim=1)          # unit length
```

**The data.** Build triplets on the fly from labelled images:

```python
class TripletDataset(Dataset):
    def __init__(self, base, labels):                        # base returns (image, label)
        self.base, self.labels = base, list(labels)
        self.by_class = {}
        for i, y in enumerate(self.labels):
            self.by_class.setdefault(y, []).append(i)
        self.classes = list(self.by_class)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        anchor, y = self.base[i]
        pos = random.choice([j for j in self.by_class[y] if j != i] or [i])
        neg = random.choice(self.by_class[random.choice([c for c in self.classes if c != y])])
        return anchor, self.base[pos][0], self.base[neg][0]
```

**The loop** is the standard one (Chapter 6) with three inputs:

```python
loss_fn = nn.TripletMarginLoss(margin=0.5)
for a, p, n in loader:
    loss = loss_fn(net(a.to(device)), net(p.to(device)), net(n.to(device)))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

| Choice | Guidance |
|---|---|
| **Margin** | With unit-length embeddings, 0.2 to 0.5 is common. If the loss drops to zero almost immediately, training is too easy: raise the margin or use harder triplets |
| **Starting point** | A pretrained backbone with a small learning rate (1e-4). Training from scratch needs far more data |
| **Labels** | You only need "same class or not". Classes can be products, people, places |

> [!WARNING]
> **Watch out.**
> - **The positive must be a different image of the same class**, not the anchor itself, or the loss is trivially zero. The code above avoids that.
> - **Random negatives get too easy fast.** Most triplets stop teaching anything. The fix is *hard negatives* (below).
> - A **collapsed** embedding (every image mapped to nearly the same point) shows as a loss stuck near the margin. Lower the learning rate and use normalised embeddings.

> [!NOTE]
> **🔵 Advanced alternative: hard-negative mining.**
>
> **What changes:** instead of a random negative, pick the negative that the current model finds *most similar* to the anchor (within the batch, or from a periodically refreshed index).
>
> **Why teams do it:** these are the triplets the model still gets wrong, so every batch carries signal and training converges to much better search quality. The risk: if your labels contain mistakes, the "hardest negatives" are often exactly the mislabelled images, so use *semi-hard* negatives (farther than the positive but inside the margin) when data is noisy.
>
> Libraries such as `pytorch-metric-learning` implement mining and many loss functions (contrastive, ArcFace, NT-Xent) so you do not hand-roll them.

---

## 11.4 Searching

**Step 1: embed the whole gallery once** and keep the result.

```python
@torch.no_grad()
def embed_all(net, loader, device):
    net.eval()                                   # no dropout, BatchNorm in eval mode
    E, Y = [], []
    for x, y in loader:
        E.append(net(x.to(device)).cpu()); Y.append(y)
    return torch.cat(E), torch.cat(Y)
```

**Step 2: embed the query, find the nearest gallery embeddings.**

```python
def search(query, gallery, k=5, exclude_self=False):
    sims = F.normalize(query, dim=1) @ F.normalize(gallery, dim=1).T       # cosine similarity matrix
    scores, idx = sims.topk(k + int(exclude_self), dim=1)
    return (idx[:, 1:], scores[:, 1:]) if exclude_self else (idx, scores)
```

| Distance | When |
|---|---|
| **Cosine similarity** | Default for normalised embeddings. Higher is closer |
| **Euclidean (L2)** | Equivalent ranking for unit-length embeddings; the usual choice in triplet loss |

> [!WARNING]
> **Watch out: `exclude_self`.** If the query image is *in* the gallery, its nearest neighbour is itself at distance zero. Drop it **only** when the query really is in the gallery. Dropping the first result when the query is a new image silently throws away a real match.

Use the **same preprocessing** for gallery and query (Chapter 5), and re-embed the gallery whenever you change the model.

---

## 11.5 Evaluating a search

You need labels (or a human notion of "relevant") and a **separate query set**, not the images the model trained on, or the score is inflated.

**Recall@k:** "for what share of queries is at least one of the top k results from the correct class?"

```python
def recall_at_k(E, L, k=5):
    idx, _ = search(E, E, k=k, exclude_self=True)          # query set = gallery, so exclude itself
    return (L[idx] == L[:, None]).any(dim=1).float().mean().item()
```

| Metric | Question |
|---|---|
| **Recall@k** | Is a right answer somewhere in the top k? |
| **Precision@k** | How many of the top k are right? |
| **Mean average precision (mAP)** | How well are the right ones ranked overall? |

Then **look at the results**: show a query with its top 5 neighbours, for 20 or so queries. It exposes problems no metric does: results matched on background or colour rather than content, or an obvious duplicate ranked below a poor match.

> [!TIP]
> **🟢 Extra: the baseline comparison.** Report recall@k for (a) raw pretrained features and (b) your trained embedding. If (b) is not clearly better, your metric learning added nothing.

---

## 11.6 Scaling up

Brute-force search compares the query with every gallery embedding. That is fine up to roughly hundreds of thousands of images and becomes slow beyond.

> [!NOTE]
> **🔵 Advanced alternative: an approximate nearest-neighbour index (FAISS).**
>
> ```python
> # pip install faiss-cpu
> import faiss
> index = faiss.IndexFlatIP(512)                    # exact inner-product search; use with normalised vectors
> index.add(gallery.numpy().astype("float32"))
> scores, ids = index.search(query.numpy().astype("float32"), k=5)
> ```
>
> **What changes:** a dedicated index structure. `IndexFlatIP` is still exact; for millions of images switch to an approximate index (`IndexIVFFlat`, `IndexHNSWFlat`) that trades a little recall for large speed-ups.
>
> **Why teams do it:** searches over millions of items in milliseconds, and the index can be saved and loaded. Vector databases (managed services) add filtering, updates and scaling on top of the same idea.

---

## 11.7 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Query matches itself | Top result is the query, distance zero | `exclude_self=True` only when the query is in the gallery |
> | Evaluated on training images | Excellent recall that collapses on new images | Hold out a query set |
> | Different preprocessing for query and gallery | Poor matches that look random | One shared preprocessing |
> | Not in `eval()` mode when embedding | Embeddings change between runs | `model.eval()` and `no_grad` |
> | Embeddings not normalised | Distances dominated by vector length | `F.normalize` |
> | Too-easy triplets | Loss near zero, search quality stops improving | Raise margin; hard or semi-hard negatives |
> | Collapsed embedding | Loss stuck near the margin; all results alike | Lower learning rate; normalise; check labels |
> | Search matches background or colour | Neighbours look alike but are different objects | Look at results; crop, augment, or fix the data |
> | Stale index | Results get worse after retraining | Re-embed the gallery whenever the model changes |
> | Noisy labels with hard mining | Training gets worse | Use semi-hard negatives; clean labels |

---

## Key takeaways

1. An **embedding** is a vector where similar images are close. It powers search, deduplication, few-shot classification and anomaly detection.
2. **Start with a pretrained backbone's features.** Train a custom embedding only if you can beat that baseline.
3. **Triplet loss** (anchor, positive, negative, margin) with a shared network trains for similarity. Use hard or semi-hard negatives when random ones get too easy.
4. **Normalise** embeddings, embed the gallery once, search by cosine similarity, and use `exclude_self` correctly.
5. Evaluate with **recall@k on held-out queries** and always look at example results.
6. For millions of images, use an approximate index such as FAISS.

**Previous:** [Chapter 10: Evaluation](10_evaluation.md) &nbsp;|&nbsp; **Next:** [Chapter 12: Interpretability](12_interpretability.md)
