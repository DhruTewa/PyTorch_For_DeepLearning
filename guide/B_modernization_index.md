# Appendix B: Outdated vs Current Practice (Index)

Every purple box in the guide, in one place. Each is a habit you may see in older tutorials, notebooks or courses, with what replaced it.

| Topic | Older practice | Current practice | Chapter |
|---|---|---|---|
| Progress output | `print()` only | `logging` to screen and file | [0.6](00_project_skeleton.md#06-device-and-logging) |
| Normalisation statistics | Computed on all data | Computed on training data only | [1.4](01_training_recipe.md#14-preparing-data-honestly) |
| Train/validation split | Unseeded random split | Seeded, and the list of files saved | [2.5](02_data_inspection_splitting.md#25-splitting-honestly) |
| Augmentation API | `transforms.ToTensor()` and hand-picked flips | `transforms.v2` (`ToImage`, `ToDtype`) and automatic policies | [3.2](03_data_pipeline.md#32-transforms-and-augmentation) |
| Flatten size | Lazy `Linear` created inside `forward` | Compute the size once in `__init__` | [4.4](04_models.md#44-building-a-cnn-step-by-step) |
| Loading pretrained weights | `pretrained=True` | `weights=...` (name the exact set) | [5.3](05_transfer_learning.md#53-steps) |
| Loading a saved file | `strict=False`, whole-model pickles, `weights_only=False` | Strict loading, `state_dict`, `weights_only=True` | [5.8](05_transfer_learning.md#58-reloading-your-own-saved-model) |
| Optimizer | `Adam` with `weight_decay` | `AdamW` | [6.4](06_training_loop.md#64-optimizer-and-learning-rate) |
| Quantization | `torch.ao.quantization` | Its deprecation notice points to `torchao` and export-based flows: check current docs for new work | [13.4](13_optimization.md#134-quantization) |
| Model export | TorchScript (`torch.jit.trace`) | ONNX or `torch.export` | [14.5](14_deployment.md#145-export-formats) |
| ONNX export arguments | `dynamic_axes`, opset 17 | `dynamic_shapes`, opset 18 or higher (version dependent) | [14.6](14_deployment.md#146-exporting-to-onnx) |

> [!NOTE]
> "Current" reflects what was true when this guide was written. Library APIs move: when something here conflicts with your installed version's documentation, trust the documentation.
