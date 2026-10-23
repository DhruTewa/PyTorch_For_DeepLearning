# CV Pocket Guide

A practical, project-ordered guide to computer vision projects in PyTorch, with a small toolkit of reusable helper modules.

## The project route

![The computer vision project route: set up, data, choose a model, train, evaluate, improve, tune, then shrink, serve, test and monitor if it goes to production](CV_Project.png)

Each step names the guide chapters to read. Steps 8 and 9 apply only if the model goes to production. For the clickable version (chapter links, a highlight checklist and an animated walkthrough), download `cv_guide_roadmap.html` and open it in a browser. GitHub shows only its source code.

```
guide/        19 chapters (00 to 18), appendices and an index: start at guide/README.md
cv_toolkit/   generic, reusable modules (copy these into a new project)
  config.py        seeds, device, logging, one config object
  data.py          datasets, transforms, splits, loaders, data checks
  models.py        CNNs, transfer learning, embeddings
  train.py         optimizer/scheduler factories, training loops, checkpoints
  evaluate.py      metrics, error analysis, inference, retrieval
  viz.py           plots
  interpret.py     hooks, saliency, CAM, Grad-CAM
  optimize.py      pruning, quantization, QAT, size/latency
  tuning.py        Optuna search
  lightning_tools.py  PyTorch Lightning version of the pipeline
  deploy.py        bundle, predictor, ONNX export
```

## Using it
```bash
pip install torch torchvision matplotlib pandas        # core
pip install optuna lightning torchmetrics torchinfo     # optional, per module
```
Import with `from cv_toolkit import data, models, train, evaluate`. Copy the folder into your project, or install it as a package (see Chapter 15).

## How to read the code
* Every function starts with "In plain words".
* `MODERNIZE` marks a spot where a common older approach is out of date, with the reason.
* `ADVANCED (production alternative)` blocks are commented code. Each has a "What changes" and a "Why teams do it" header. Nothing in them runs.

## What was and was not tested
Tested on CPU (torch 2.14.1) with synthetic data: the modules import and their main functions run, metrics match torchmetrics, ONNX export matches PyTorch outputs, and the model-shrinking pipeline runs.

Not tested: real datasets and real accuracy numbers, downloading pretrained weights, GPU and mixed-precision paths, the image-generation code, the FastAPI sketch in `deploy.py`, and the commented ADVANCED blocks. Library APIs move, so check your installed version's documentation.
