"""
cv_toolkit: a reusable computer-vision project toolkit
=========================================================================
Generic, config-driven modules with plain-English notes.

  config           Chapter 0   settings, seeds, device, logging
  data             Chapters 2-3  inspect, splits, transforms, datasets, loaders
  models           Chapters 4-5  MLP, CNN, MobileNet-style, pretrained, embeddings
  train            Chapters 6-7  loops, early stopping, checkpoints, sanity checks
  lightning_tools  Chapter 6    Lightning DataModule / Module / Trainer
  tuning           Chapter 9   Optuna
  evaluate         Chapters 10-11  metrics, error analysis, embeddings search
  interpret        Chapter 12  feature maps, saliency, CAM, Grad-CAM
  optimize         Chapter 13  pruning, quantization
  deploy           Chapter 14  bundle, predictor, ONNX
  viz              plotting for all of the above

Import the module you need; nothing heavy loads at `import cv_toolkit`.
    from cv_toolkit import data, models, train, evaluate
"""
