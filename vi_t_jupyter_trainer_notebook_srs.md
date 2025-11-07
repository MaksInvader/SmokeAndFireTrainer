# Vision Transformer (ViT) Jupyter Trainer Notebook — Software Requirements Specification (SRS)

## 1. Purpose
This SRS defines the requirements for a Jupyter Notebook based trainer for Vision Transformer (ViT) models to localize **Fire and Smoke** from images using a YOLO-style labeled dataset. The notebook is intended to be modular and interactive so that users can **select optional "blocks" (features/modules)** via simple toggles (e.g., ipywidgets) and immediately understand how each selection affects model behavior, training stability, accuracy, inference speed and resource usage.

## 2. Scope
- A single-file Jupyter Notebook (or small set of companion files) that implements data loading, model definition, training loop, evaluation, and inference for ViT-based object localization.
- Support for YOLO-style bounding-box labels (class + x_center,y_center,w,h normalized or pixel coordinates — user-configurable).
- Interactive controls (checkboxes/sliders) to enable/disable optional blocks and hyperparameters.
- Clear documentation and textual explanation of how each block affects the final model.
- Experiment tracking, checkpointing, logging and reproducible configurations (YAML/JSON export).

## 3. Definitions and Abbreviations
- ViT — Vision Transformer.
- YOLO-style dataset — bounding boxes in format commonly used by YOLO models (class, x_center, y_center, width, height).
- FPN — Feature Pyramid Network.
- GIoU/DIoU/CIoU — bounding-box loss variants.
- NMS — Non-Maximum Suppression.

## 4. Stakeholders
- ML Practitioners building Fire & Smoke localization models.
- Dataset curators labeling images using YOLO format.
- Engineers integrating model into edge or cloud inference pipelines.

## 5. High-Level Requirements
### 5.1 Functional Requirements
FR-1: Notebook must load dataset from a user-specified directory (train/val/test), read YOLO-style labels, and create PyTorch/TensorFlow data pipelines.

FR-2: Notebook must provide an interactive control panel (ipywidgets) or configuration cell where user can enable/disable blocks and tune hyperparameters.

FR-3: Notebook must allow multiple ViT backbone configurations (patch size, depth, heads, embedding dim) and optionally load ImageNet-pretrained weights.

FR-4: Notebook must offer optional detection/localization heads and corresponding loss functions: YOLO-style head, anchor-free regression head, FPN+head for multi-scale.

FR-5: Notebook must include data augmentation options (Resize/Crop, RandAugment, MixUp, CutMix, Mosaic), and allow toggling.

FR-6: Notebook must support training features: mixed precision (AMP), gradient accumulation, distributed (torch.distributed/or single-GPU), checkpointing, resume training.

FR-7: Notebook must compute and log metrics: mAP@0.5, mAP@[.5:.95], precision/recall, per-class AP, inference FPS, and confusion matrices for classification of fire vs smoke.

FR-8: Notebook must export final model (TorchScript/ONNX) and provide example inference cells.

FR-9: Notebook must provide explanations (inline) for each block: expected effect on accuracy, computational cost, and recommended use cases.

### 5.2 Non-Functional Requirements
NFR-1: Should run on standard research GPU (e.g., single NVIDIA GPU with 12GB VRAM) but also include memory-efficient options.

NFR-2: Notebook should be modular and readable; sections split into clearly labeled cells.

NFR-3: Provide deterministic seeding option for reproducibility.

NFR-4: Minimal external dependencies; prefer PyTorch + torchvision + timm + pytorch-lightning or a plain PyTorch implementation. Optionally support HuggingFace Transformers.

## 6. Data and I/O
- Input: image files + YOLO-style text labels (one file per image). Config param to select normalization type (normalized vs absolute coords).
- Output: checkpoint files, logs (TensorBoard/Weights & Biases optional), final exported model file (.pt/.onnx), evaluation CSVs.

## 7. Component Design (Notebook Structure)
1. Header / Setup
   - Environment checks, imports, device selection, reproducibility seeds.
   - Install commands (optional) guarded by `if`.

2. Configuration Cell (UI)
   - ipywidgets panel (Checkboxes, Sliders, Dropdowns) to choose blocks and hyperparameters.
   - A single `config` dict is produced from UI state, and displayed as YAML.

3. Data Loader
   - Parser for YOLO label format.
   - Augmentation pipeline (torchvision or albumentations) with toggleable augmentations.

4. Model Builder
   - ViT backbone factory (timm/HF) with parameters: patch_size, embed_dim, depth, num_heads, mlp_ratio, drop_rate, attn_drop, use_pretrained.
   - Positional embedding options: learned, sinusoidal, relative.
   - Optional blocks inserted here (see Section 8).

5. Detection Head(s)
   - YOLO-style head mapping tokens/patch embeddings to bbox and class outputs.
   - FPN option to supply multi-scale features.
   - Anchor-free head option (center-ness + regression).

6. Loss & Metrics
   - Classification loss (CE), localization loss (L1/GIoU/DIoU), objectness loss.
   - NMS/Post-processing parameters.

7. Training Loop
   - Mixed precision support, gradient accumulation, LR scheduler.
   - Checkpointing and logging.

8. Evaluation & Visualization
   - Compute mAP, plot PR curves, visualize predictions on sample images.

9. Export / Inference
   - Export to TorchScript/ONNX and run sample inference.

## 8. Optional Blocks (Selectable) — Description + Expected Impact
Below are modular blocks the user can enable/disable. For each block the notebook includes: 1) toggle in UI, 2) code snippet implementing it, 3) short note describing tradeoffs.

### 8.1 Backbone Hyperparameters
- **Patch Size (e.g., 8, 16, 32)**
  - Effect: Smaller patches -> finer spatial resolution → better localization at cost of higher memory and compute. Larger patches -> faster but coarser localization.

- **Depth / Number of Transformer Layers**
  - Effect: Deeper model -> higher capacity, can fit complex patterns (e.g., smoke textures). More prone to overfitting and slower training.

- **Number of Attention Heads**
  - Effect: More heads -> potentially better multi-scale/feature decomposition; increases compute and GPU memory.

- **Embedding Dimension / MLP Ratio**
  - Effect: Larger dims -> richer features; increases memory and inference latency.

### 8.2 Positional Embedding Types
- **Learned absolute embeddings**
  - Effect: Simpler; may generalize less to varying resolutions or input sizes.

- **Sinusoidal (fixed)**
  - Effect: Better generalization to unseen resolutions; no extra params.

- **Relative positional biases / rotary**
  - Effect: Improved attention locality and generalization, small compute overhead.

### 8.3 Pretraining
- **Load ImageNet / MAE pre-trained weights**
  - Effect: Faster convergence and significantly better accuracy with limited labeled data. If dataset is quite different (infrared, drone), may need more fine-tuning.

- **Self-supervised pretraining (MAE snippet)**
  - Effect: Good when labeled data is scarce; adds extra pretraining step but improves robustness.

### 8.4 Multi-scale Features
- **FPN (Feature Pyramid Network)**
  - Effect: Improves detection of small objects (small fire spots / smoke plumes) by combining features at multiple scales. Increases model complexity and inference time.

- **Pyramid ViT approaches (Per-stage patch merging)**
  - Effect: Native multi-scale ViT backbones provide more efficient multi-scale representation.

### 8.5 Detection Head Variants
- **YOLO-style single-shot head**
  - Effect: Fast inference, straightforward training. Good for real-time.

- **Anchor-free center-based head**
  - Effect: Often simpler label assignment, fewer hyperparameters, potentially better small-object localization.

- **Two-stage head (R-CNN style)**
  - Effect: Higher accuracy for challenging boxes, but much slower.

### 8.6 Data Augmentation Blocks
- **Mosaic**
  - Effect: Improved detection of small objects and context; changes label statistics — helps generalization.

- **MixUp / CutMix**
  - Effect: Regularizes classifier, sometimes hurts localization if not carefully applied (use cautiously for bbox tasks).

- **Color jitter / Random brightness / Smoke-specific augmentations (e.g., haze overlays)**
  - Effect: Better robustness to lighting and smoky conditions.

### 8.7 Regularization & Optimization
- **Stochastic Depth**
  - Effect: Regularizes deep ViT and reduces overfitting.

- **Dropout, LayerNorm epsilon tuning**
  - Effect: Small stabilizing effect.

- **Optimizer choices (AdamW vs SGD)**
  - Effect: AdamW often faster convergence for transformers; SGD with momentum may be preferable for very large datasets.

- **LR Scheduler (Cosine annealing, OneCycle)**
  - Effect: Cosine + warm-up tends to work well with transformers.

### 8.8 Training Performance & Stability
- **Mixed Precision (AMP)**
  - Effect: Lower memory usage, faster training with minimal to no accuracy loss.

- **Gradient Accumulation**
  - Effect: Enables larger effective batch size when VRAM is limited; can change optimization dynamics.

- **Gradient Clipping**
  - Effect: Prevents exploding gradients; stabilizes training for large LR.

### 8.9 Post-processing & Inference
- **NMS variants (standard, soft-NMS)**
  - Effect: soft-NMS can improve AP in crowded scenes; standard NMS is faster.

- **Quantization (Post-training / QAT)**
  - Effect: Reduced model size and inference latency; may reduce accuracy.

### 8.10 Monitoring & Explainability
- **Attention Map Visualization block**
  - Effect: Helps debug where model looks when predicting — useful for trust and diagnosing failure modes.

- **Grad-CAM / Saliency**
  - Effect: Helps explain predictions at the image region level.

## 9. Impact Matrix (Quick Reference)
| Block | Accuracy Impact | Speed (infer) | Memory | Use-case notes |
|---|---:|---:|---:|---|
| Smaller patch size | + (better loc.) | - | - | Use if small fires/smoke need detection |
| Deeper backbone | + | - | - | Use for complex datasets, watch overfitting |
| Pretraining (ImageNet/MAE) | ++ | ~ | ~ | Highly recommended if labels limited |
| FPN | + (small obj) | - | - | Use when objects at multiple scales |
| Mosaic | + | ~ | ~ | Helpful for small objects, needs careful tuning |
| Mixed Precision | ~ | + | + | Always enable if GPU supports it |
| Anchor-free head | +/~ | ~ | ~ | Simpler label assignment, fewer anchors |
| Soft-NMS | + (crowded) | - | ~ | Use if many overlapping boxes |

## 10. UI/Interaction Design
- The notebook will present a single **Configuration Panel** at the top. When the user toggles options and clicks **Apply**, the `config` dict is printed and the notebook will use that configuration for subsequent cells.
- The panel will be implemented with `ipywidgets` (Checkboxes, Dropdowns, IntSliders) and will include an **Explain** button next to each major block that pops up a brief markdown explanation.

## 11. Acceptance Criteria
- AC-1: Notebook runs end-to-end on a sample dataset and completes one training epoch without errors.
- AC-2: UI panel updates the `config` object and model/data loader respect toggled options.
- AC-3: Enabling pretraining loads the weights and shows measurable faster convergence (e.g. loss curve) vs cold-start on same settings.
- AC-4: FPN enabled vs disabled shows improved AP on small objects in an evaluation run (quantified in report).
- AC-5: Exported model loads and runs inference in the notebook with matching post-processing.

## 12. Test Plan
- Unit tests (or simple run cells) for label parsing, augmentation pipeline, and model forward pass.
- Integration test: train for N steps and ensure metrics are computed and logged.
- Ablation tests: toggle blocks (pretraining, FPN, mosaic) and compare metric delta.

## 13. Risks and Mitigations
- **Large memory footprint**: Provide low-memory configs (larger patch size, reduce batch size, use gradient accumulation, mixed precision).
- **Misuse of augmentations harming bbox labels**: Provide clear warnings and default-safe augmentation sets for bbox tasks.
- **Long training times for ablations**: Recommend small-scale experiments and checkpointing.

## 14. Deliverables
- `ViT_trainer.ipynb` — annotated and modular Jupyter notebook.
- `config_schema.yaml` — available config template.
- Example dataset folder and a small synthetic dataset for smoke/fire.
- Quickstart README cell in the notebook.

## 15. Appendix
### Example UI widgets (conceptual)
```python
import ipywidgets as widgets
patch_size = widgets.Dropdown(options=[8,16,32], value=16, description='Patch')
use_pretrained = widgets.Checkbox(value=True, description='Use pretrained')
use_fpn = widgets.Checkbox(value=False, description='Enable FPN')
mosaic = widgets.Checkbox(value=True, description='Mosaic Aug')
apply_btn = widgets.Button(description='Apply')
# on click: collect values -> config dict
```

### Minimal config schema (YAML)
```yaml
model:
  backbone: vit_base
  patch_size: 16
  pretrained: true
  depth: 12
  num_heads: 12
train:
  batch_size: 16
  epochs: 50
  optimizer: AdamW
augmentations:
  mosaic: true
  mixup: false
  randaugment: true
detection:
  head: yolov1
  loss: giou
```

---

*End of SRS.*

*Notes:* The notebook will intentionally separate configuration from implementation so users can toggle blocks without editing core cells. Each selectable block is accompanied by a short explanation of tradeoffs and recommended defaults for Fire/Smoke localization.

