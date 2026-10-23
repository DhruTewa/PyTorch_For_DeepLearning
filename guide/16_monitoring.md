# Chapter 16: Monitoring and Maintenance

| | |
|---|---|
| **Level** | Intermediate |
| **You will learn** | Why deployed models degrade, what to watch, simple drift checks, and when to retrain |
| **Prerequisites** | Chapters 10, 14 |
| **Ready-made code** | Patterns only; adapt to your project |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

> [!TIP]
> **🟢 Extra: this whole chapter.** Many tutorials end before deployment. A model's work starts there.

**In this chapter**
[16.1 The idea](#161-the-idea) · [16.2 Why models degrade](#162-why-models-degrade) · [16.3 What to log](#163-what-to-log) · [16.4 Watching](#164-what-to-watch) · [16.5 Drift checks](#165-simple-drift-checks) · [16.6 Getting labels](#166-getting-labels-back) · [16.7 Retraining](#167-when-and-how-to-retrain) · [16.8 Common problems](#168-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 16.1 The idea

A model is trained on a snapshot of the world, and the world keeps moving. A model that scored 95% on release day can quietly slide to 80% without a single error message, because it keeps returning confident answers.

**Monitoring is how you find out before your users do.**

---

## 16.2 Why models degrade

| Cause | Example |
|---|---|
| **Data drift** (the inputs change) | New camera model, different lighting, a new store layout, a season change, a different image resolution |
| **Concept drift** (the right answer changes) | What counts as a "defect" is redefined; new product versions look different |
| **New classes appear** | A category the model never saw |
| **Upstream changes** | A pipeline bug starts resizing or compressing images differently |
| **Feedback loops** | The model's own decisions change the data it later sees |

The problem: with no labels in production, you cannot compute accuracy directly. So you watch **indirect signals**, and sample for **real labels** (16.6).

---

## 16.3 What to log

Log enough to investigate later, without storing what you should not.

| Log | Why |
|---|---|
| Timestamp and **model version** | Trace a bad result to a model |
| **Predicted class and top probabilities** | Distribution checks (16.5) |
| **Input statistics**: image size, brightness, contrast, file type | Cheap drift signals |
| **Latency** and errors | Operations health |
| A **sample of the actual images** (for example 1% plus all low-confidence ones), where policy and privacy allow | You will need examples to diagnose problems and build the next training set |
| The user's **correction or outcome**, when you can get it | The only source of real accuracy |

```python
import json, time, logging
log = logging.getLogger("predictions")

def log_prediction(image_stats, top, version, latency_ms):
    log.info(json.dumps({
        "t": time.time(), "model": version, "latency_ms": round(latency_ms, 1),
        "top_class": top[0][0], "top_prob": round(top[0][1], 4), **image_stats,
    }))

def image_stats(tensor01):                                       # tensor in 0..1, shape [3, H, W]
    return {"brightness": float(tensor01.mean()), "contrast": float(tensor01.std()),
            "h": tensor01.shape[1], "w": tensor01.shape[2]}
```

> [!WARNING]
> **Watch out: privacy and compliance.** Images can contain people, faces, medical or personal information. Decide *before* logging what you may store, for how long, who can see it, and whether it needs consent or anonymisation. When in doubt, log statistics, not pictures.

---

## 16.4 What to watch

| Signal | How | A worrying change |
|---|---|---|
| **Predicted class mix** | Share of predictions per class over a day or week | A class suddenly doubles or vanishes |
| **Confidence** | Average top probability; share of predictions below a threshold (say 0.6) | Confidence falls: the model is seeing unfamiliar data |
| **Input statistics** | Brightness, contrast, size, file type | A shift away from the training values |
| **Latency and error rate** | Standard service metrics | Slower, or more failures |
| **Accuracy on sampled, labelled data** | Label a small random sample each week (16.6) | A drop. **The only direct measure** |
| **Subgroup results** | Per camera, site, region | One subgroup degrading while the average hides it |

Set **thresholds and alerts** in advance ("average confidence below 0.7 for two days", "a class share changes by more than 50%"), and decide who looks at an alert and what they do.

---

## 16.5 Simple drift checks

Compare a **reference** (your training or validation data, or a good week) with **live** data, on a few simple numbers.

**For a number** (brightness, confidence): a **two-sample test**, or the **Population Stability Index (PSI)**.

```python
from scipy import stats
import numpy as np

p = stats.ks_2samp(reference_values, live_values).pvalue         # small p-value: the distributions differ

def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
```

| PSI | Rule of thumb (a convention, not a law) |
|---|---|
| Below 0.1 | Stable |
| 0.1 to 0.25 | Some shift: look at it |
| Above 0.25 | Large shift: investigate |

**For the class mix:** a chi-square test of the live counts against the counts you would expect from the reference proportions.

```python
expected = ref_counts / ref_counts.sum() * live_counts.sum()
p = stats.chisquare(live_counts, expected).pvalue
```

> [!WARNING]
> **Watch out.**
> - **Statistical significance is not importance.** With huge volumes, tiny harmless differences become "significant". Judge by the **size** of the shift (PSI, change in share), not only the p-value.
> - Drift in the *inputs* does not always hurt accuracy, and accuracy can fall with no visible input drift. Drift checks are early warnings, not proof.
> - Compare like with like: same time of day, same weekday mix, same camera mix.

> [!NOTE]
> **🔵 Advanced alternative: drift on embeddings, and monitoring tools.**
>
> **What changes:** instead of hand-picked statistics, compare the **distribution of embeddings** (Chapter 11) of live images against reference images. A shift in embedding space catches changes you did not think to measure (a new type of background, a different object).
>
> **Why teams do it:** it works across image content, not only brightness and size. Open-source libraries (such as Evidently) and hosted platforms provide ready-made drift reports and dashboards; consider them once you outgrow a few scripts.

---

## 16.6 Getting labels back

Without real labels you cannot know the true accuracy. Ways to get some:

| Source | Notes |
|---|---|
| **Human review of a random sample** | Label 50 to 200 recent predictions each week. Always unbiased: random sampling is key |
| **Review of low-confidence predictions** | Finds the hard cases and gives the best training data, but is *not* a fair estimate of overall accuracy (it is biased toward difficult images) |
| **User corrections or downstream outcomes** | Cheap, but biased toward cases people bother to correct |
| **Ground truth that arrives later** | A repair confirms a defect; a lab result confirms a finding |

Keep the **random** sample for measuring accuracy and the **hard-case** sample for improving the model, separately.

---

## 16.7 When and how to retrain

**Triggers (decide in advance):**
- Accuracy on the labelled sample falls below the target.
- A significant, explained drift in inputs or class mix.
- New classes or new conditions to support.
- A scheduled refresh (for example quarterly) when the world changes steadily.

**Process:**
1. **Add the new data** (especially the failures and hard cases) to the training set, with correct labels.
2. **Retrain** with the same recipe (Chapter 0: config, seed, tracked run).
3. **Evaluate against the old model on the same fixed test set, and on a fresh recent sample** (Chapter 10). The new model must be better on *both*, or at least not worse on the old cases.
4. **Compare per class and per subgroup**, not only overall.
5. **Roll out carefully**: run the new model silently alongside the old one ("shadow mode") and compare predictions, then switch a small share of traffic, and keep the ability to **roll back** instantly (this is why you keep model versions).
6. **Record** the new version, data version and results.

> [!WARNING]
> **Watch out: forgetting.** Retraining only on recent data can make the model worse on older cases. Keep a broad training set (old plus new), and keep a fixed regression test set of the older cases.

---

## 16.8 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | No monitoring at all | You learn of failures from complaints | Log predictions, confidence, input stats |
> | Only watching accuracy | There are no labels, so nothing is watched | Watch indirect signals and sample for labels |
> | Biased label sample | Estimated accuracy far from reality | A random sample for accuracy; a separate hard-case sample |
> | Alerts nobody owns | Alerts ignored | Name an owner and an action per alert |
> | Reacting to every p-value | Constant false alarms with big volumes | Look at effect size (PSI, change in share) |
> | Retrain replaces everything | New model worse on older cases | Keep a regression test set; compare per class |
> | No rollback | A bad model stays live | Keep versions; shadow mode, gradual rollout |
> | Logging images without a policy | Privacy or legal exposure | Decide retention and consent first; log statistics where possible |
> | An upstream pipeline change goes unnoticed | Sudden drop in confidence or a size shift | Alert on input statistics such as image size and file type |

---

## Key takeaways

1. Models degrade silently as the world changes. **Monitor on purpose.**
2. Log the model version, predictions, confidence and input statistics. Mind privacy.
3. Watch **indirect signals** (class mix, confidence, input stats), and **sample for real labels**, since that is the only direct measure of accuracy.
4. Use simple drift checks (PSI, two-sample tests), judge the size of a shift, and treat drift as a warning, not proof.
5. Decide **retraining triggers** in advance. Evaluate the new model against the old one, then roll out gradually with the ability to roll back.

**Previous:** [Chapter 15: Testing and packaging](15_testing_packaging.md) &nbsp;|&nbsp; **Next:** [Chapter 17: Beyond classification](17_beyond_classification.md)
