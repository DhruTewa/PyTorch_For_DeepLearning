# Chapter 9: Hyperparameter Tuning

| | |
|---|---|
| **Level** | Intermediate |
| **You will learn** | What to tune and in what order, how to run an automated search with Optuna, and how to read the results, including the accuracy-versus-size trade-off |
| **Prerequisites** | Chapters 2, 3, 4, 6 |
| **Ready-made code** | `cv_toolkit/tuning.py` |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

**In this chapter**
[9.1 The idea](#91-the-idea) · [9.2 What to tune](#92-what-to-tune-and-in-what-order) · [9.3 Vocabulary](#93-the-vocabulary) · [9.4 The search space](#94-designing-the-search-space) · [9.5 The objective](#95-the-objective-function) · [9.6 Running a study](#96-running-a-study) · [9.7 Reading results](#97-reading-the-results) · [9.8 Accuracy vs size](#98-accuracy-versus-size) · [9.9 Honest tuning](#99-tuning-honestly) · [9.10 Common problems](#910-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 9.1 The idea

A **hyperparameter** is a setting you choose *before* training (learning rate, number of layers, dropout, batch size), as opposed to the weights, which training learns. Instead of guessing and re-running by hand, you let the computer try many combinations, learn which regions work well, and keep the best.

**When to tune:** after you have a working baseline and clean data. Tuning a broken pipeline just finds the least-broken settings. A better dataset usually beats a better search.

---

## 9.2 What to tune, and in what order

Not all settings matter equally. Tune the high-impact ones first.

| Priority | Setting | Notes |
|---|---|---|
| 1 | **Learning rate** | By far the most important. Search on a log scale |
| 2 | **Model size** (layers, channels, or which pretrained network) | Trades accuracy against speed and size |
| 3 | **Regularisation** (dropout, weight decay) | Fights overfitting (Chapter 8) |
| 4 | **Input resolution** | Detail versus cost |
| 5 | **Batch size** | Interacts with the learning rate |
| 6 | **Augmentation strength**, optimizer, scheduler | Usually after the others |

> [!TIP]
> **🟢 Extra: do the cheap manual search first.** Try learning rates 1e-2, 1e-3, 1e-4 by hand for a few epochs each. That alone often finds most of the gain. Use an automated search to refine, not to rescue.

---

## 9.3 The vocabulary

| Term | Meaning |
|---|---|
| **Trial** | One attempt: one set of settings, trained and scored |
| **Study** | The collection of all trials |
| **Search space** | The allowed range for each setting |
| **Objective** | A function: settings in, one score out |
| **Sampler** | The strategy for picking the next settings (Optuna's default, TPE, learns from earlier trials) |
| **Pruner** | Stops hopeless trials early to save time |

---

## 9.4 Designing the search space

Optuna asks for each setting through the `trial` object:

```python
n_layers   = trial.suggest_int("n_layers", 1, 3)
filters    = trial.suggest_int("filters", 8, 64, step=8)
dropout    = trial.suggest_float("dropout", 0.1, 0.5)
lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)        # log scale
batch_size = trial.suggest_categorical("batch_size", [16, 32])
```

| Setting type | Method | Note |
|---|---|---|
| Whole number | `suggest_int(name, low, high, step=...)` | Use `step` to avoid pointless values (8, 16, 24 rather than every integer) |
| Decimal | `suggest_float(name, low, high)` | |
| Rate spanning orders of magnitude | `suggest_float(..., log=True)` | **Always for learning rate and weight decay** |
| A fixed menu | `suggest_categorical(name, [...])` | Batch size, which network, optimizer |

> [!WARNING]
> **Watch out: the log scale.** Sampling a learning rate uniformly between 1e-4 and 1e-2 puts 90% of the tries above 1e-3 and almost never tests the small values. With `log=True`, each order of magnitude gets equal attention.

**Rules for a good space:**
- **Not too wide.** Ranges covering absurd values waste trials. Use your manual search to narrow it.
- **Not too narrow.** If the best result sits at the edge of a range, widen that side and rerun.
- **Few settings.** Four to six is plenty for a first search. Every extra dimension needs more trials.
- **Conditional settings** (a filter count *per layer*) are natural: create them inside a loop over the number of layers, as below.

---

## 9.5 The objective function

The objective builds a model from the trial's settings, trains briefly, and returns a score to maximise (or minimise).

```python
import optuna

def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    filters  = [trial.suggest_int(f"filters_{i}", 8, 32, step=8) for i in range(n_layers)]
    dropout  = trial.suggest_float("dropout", 0.1, 0.5)
    lr       = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    train_loader, val_loader = make_loaders(batch_size)          # you provide this
    model = build_model(n_layers, filters, dropout).to(device)   # and this
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(4):                                       # a SHORT run, not full training
        train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        _, val_acc = evaluate(model, val_loader, loss_fn, device)
        trial.report(val_acc, epoch)                             # tell Optuna the progress
        if trial.should_prune():                                 # hopeless? stop now
            raise optuna.TrialPruned()
    return val_acc
```

**Design points:**
- Trials use **few epochs** (3 to 10) to rank settings quickly. The ranking at 4 epochs usually, not always, matches the ranking at 40. Retrain the winner for the full length afterwards.
- Return **validation** accuracy (or loss). Never the test score.
- Everything that depends on a setting (the data loader for batch size, the model for layers) is built **inside** the objective, fresh for every trial.
- Pass the loaders and device in through a function or closure, so the objective does not depend on global variables.

---

## 9.6 Running a study

```python
study = optuna.create_study(
    study_name="cv_tuning",
    storage="sqlite:///tuning.db",          # saved to a file: stop, resume, inspect later
    load_if_exists=True,
    direction="maximize",                   # higher accuracy is better ("minimize" for loss)
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
)
study.optimize(objective, n_trials=30)

print(study.best_value, study.best_params)
```

| Piece | Why |
|---|---|
| `storage="sqlite:///..."` | Without it, results vanish when Python exits. With it, you can resume and look at old studies |
| `MedianPruner` | Stops a trial whose score at some epoch is below the median of earlier trials at that epoch. Saves a lot of time. The first few trials are never pruned (`n_startup_trials`) |
| `TPESampler(seed=...)` | Reproducible search. TPE learns where good settings lie rather than guessing blindly |
| `n_trials` | 20 to 50 is a reasonable first budget. Stop when the best score stops improving |

**Seed your own starting guess** so the search begins from something you believe in:

```python
study.enqueue_trial({"n_layers": 2, "filters_0": 16, "filters_1": 16, "dropout": 0.3, "lr": 1e-3, "batch_size": 16})
```

---

## 9.7 Reading the results

```python
df = study.trials_dataframe()                           # one row per trial
print(df[df.state == "COMPLETE"].sort_values("value", ascending=False).head(5))
```

- **Look at several top trials, not only the winner.** If the top five all share a learning rate range, that range is solid. If they disagree wildly, the score is noisy or the settings do not matter much.
- **Count pruned trials.** A healthy search prunes a good share of them. Zero pruned means the pruner is not helping; nearly all pruned may mean the space is poor.
- **Which settings mattered?**

```python
print(optuna.importance.get_param_importances(study))   # e.g. learning rate far ahead of the rest
```

If one setting dominates, narrow the others and search *that* one more finely. Plots are available too (`optuna.visualization.plot_param_importances(study)` and `plot_optimization_history(study)`).

> [!WARNING]
> **Watch out: with a small validation set, the "best" trial may just be lucky.** Differences of one or two points are within noise. Before choosing, retrain your top two or three candidates with a few different seeds and compare averages.

---

## 9.8 Accuracy versus size

Often the second-best model is far smaller and almost as accurate, and that is the better choice for deployment. Look at accuracy **and** parameter count (or latency) together.

```python
rows = []
for t in study.trials:
    if t.state.name == "COMPLETE":
        model = build_model(t.params["n_layers"], [t.params[f"filters_{i}"] for i in range(t.params["n_layers"])], t.params["dropout"])
        rows.append({"trial": t.number, "acc": t.value, "params": sum(p.numel() for p in model.parameters())})
print(pd.DataFrame(rows).sort_values("acc", ascending=False).head(5))
```

> [!NOTE]
> **🔵 Advanced alternative: optimise accuracy and size at the same time.**
>
> **What changes:** a *multi-objective* study returns a set of best trade-offs (the "Pareto front") instead of one winner.
>
> **Why teams do it:** in production you almost always have a size or latency budget. The result is a menu: "this much accuracy for this size".
>
> ```python
> study = optuna.create_study(directions=["maximize", "minimize"])
> def objective(trial):
>     ...                                     # build and train as before
>     return val_acc, num_parameters
> study.optimize(objective, n_trials=50)
> study.best_trials                           # the Pareto-optimal trials
> ```
>
> **Also useful:** `study.optimize(objective, n_trials=100, n_jobs=4)` runs trials in parallel (best for CPU-bound work; on one GPU parallel trials compete for the same device). For very large searches, look at Ray Tune or a shared database server.

---

## 9.9 Tuning honestly

1. **Tune on validation, report on test.** The test set is touched once, after the final choice.
2. **After the search, retrain** the best settings for the full number of epochs, ideally with a few seeds.
3. **Optional:** retrain on train plus validation once settings are final, then evaluate on the test set. Only do this if you will not need the validation set to decide when to stop; otherwise keep it separate.
4. **A search can overfit the validation set.** Run hundreds of trials on a small validation set and the winner partly reflects luck. Keep the test set sealed and be suspicious of a large validation-to-test drop.
5. **Write down the search space and seed** with the results, so the experiment can be repeated (Chapter 0).

---

## 9.10 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Learning rate sampled linearly | Winning rates are all at the top of the range | `log=True` |
> | Best value at the edge of a range | The winner uses the maximum or minimum allowed | Widen that range and rerun |
> | Results lost when the script ends | Cannot resume or inspect | Use `storage="sqlite:///..."` |
> | Trials too short or too long | Ranking at 3 epochs differs from ranking at full training | Check the top trials with longer runs |
> | Noisy validation | Top trials differ by about one point | More validation images; retrain candidates over several seeds |
> | Tuning on the test set | Great test score, poor in real use | Validation only |
> | Too many settings | Search never improves | Fix the unimportant ones; tune four to six |
> | A crash in one trial stops the study | Study halts on an out-of-memory or shape error | Keep the space within what fits in memory; Optuna can skip failed trials with `catch=(RuntimeError,)` in `optimize` |
> | Tuning a broken pipeline | All trials score poorly | Run the Chapter 7 sanity checks first |

---

## Key takeaways

1. **Learning rate first**, then model size, then regularisation. A quick manual search often gets most of the benefit.
2. Search the learning rate on a **log scale**.
3. The objective is a function from settings to a **validation** score. Use short runs, report progress, and let the **pruner** drop bad trials.
4. Save the study to a file so you can resume it, and **look at several top trials**, not one winner.
5. Weigh accuracy against size. The second-best model is often the better one to ship.
6. Retrain the winner properly, with a few seeds, and use the test set once.

**Previous:** [Chapter 8: Beating overfitting](08_overfitting.md) &nbsp;|&nbsp; **Next:** [Chapter 10: Evaluation](10_evaluation.md)
