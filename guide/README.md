# Computer Vision Pocket Guide

A practical, project-ordered guide to building computer vision projects with PyTorch. Read it top to bottom once, then use it as a reference.

**Colour key used in every chapter**

> 🟢 **Extra**: goes beyond the core recipe
> 🟡 **Watch out**: a common problem and how to spot it
> 🟣 **Outdated vs current**: older practice and what replaced it
> 🔵 **Advanced**: a production alternative, shown as code you can adopt later

(In viewers that support GitHub-style callouts, these appear as coloured boxes.)

## Chapters

| # | Chapter | You will learn |
|---|---|---|
| 0 | [Project skeleton](00_project_skeleton.md) | Folder layout, config, seeds, tracking, the whole lifecycle |
| 1 | [The training recipe](01_training_recipe.md) | The five steps behind every network |
| 2 | [Data: inspecting and splitting](02_data_inspection_splitting.md) | Looking at data, quality checks, honest splits |
| 3 | [Data: transforms, loaders and imbalance](03_data_pipeline.md) | Augmentation, datasets and loaders, rare classes |
| 4 | [Designing models](04_models.md) | Layers, CNN blocks, choosing settings, efficient designs |
| 5 | [Transfer learning](05_transfer_learning.md) | Reusing pretrained networks |
| 6 | [Training the model](06_training_loop.md) | The loop, optimizers, schedules, checkpoints |
| 7 | [Training health and speed](07_training_health.md) | Sanity checks, reading curves, faster training |
| 8 | [Beating overfitting](08_overfitting.md) | Diagnose and fix overfitting and underfitting |
| 9 | [Hyperparameter tuning](09_tuning.md) | Automated search with Optuna |
| 10 | [Evaluation](10_evaluation.md) | Metrics, error analysis, calibration, uncertainty |
| 11 | [Embeddings and visual search](11_embeddings.md) | Similarity, triplet loss, retrieval |
| 12 | [Interpretability](12_interpretability.md) | Heatmaps, Grad-CAM, right for the right reason |
| 13 | [Shrinking models](13_optimization.md) | Quantization, pruning, distillation |
| 14 | [Export and serving](14_deployment.md) | Bundles, predictors, ONNX, APIs |
| 15 | [Testing and packaging](15_testing_packaging.md) | Tests, packages, automation |
| 16 | [Monitoring and maintenance](16_monitoring.md) | Drift, labels, retraining |
| 17 | [Beyond classification](17_beyond_classification.md) | Detection, segmentation and other tasks (a map) |
| 18 | [Modern backbones](18_modern_backbones.md) | ViT, ConvNeXt, CLIP, DINO and how to use them |

## Appendices

| | |
|---|---|
| [A: Generating synthetic images](A_generative_images.md) | Diffusion models for extra training data |
| [B: Outdated vs current practice](B_modernization_index.md) | Every purple box in one table |

## Code that goes with the guide

`../cv_toolkit/` holds the reusable modules. Each chapter's header names the module it matches.

## Status of the code

Most snippets and every `cv_toolkit` module were run on CPU with synthetic data (torch 2.14.1). These parts were **not** tested, and are marked with a warning box where they appear: GPU paths (float16 mixed precision, DDP, `torch.compile` speed) in Chapter 7; the `open_clip` and `timm` snippets in Chapter 18; the Dockerfile and GitHub Actions file in Chapter 15; the image-generation code in Appendix A. Library APIs move, so check your installed version's documentation.
