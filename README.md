# SmokeAndFireTrainer

Vision Transformer (ViT) Jupyter Trainer for Fire & Smoke localization.

This repository contains a single interactive Jupyter notebook, `ViT_trainer.ipynb`, which is a modular trainer and demo for detecting and localizing fire and smoke using a ViT backbone and YOLO-style detection head. The notebook follows a Software Requirements Specification (SRS) and is designed for experimentation and education.

## Contents
- `ViT_trainer.ipynb` — Main notebook: setup, interactive configuration (ipywidgets), data loader (YOLO-style parsing + synthetic dataset), ViT backbone (timm), simple detection head, training loop (AMP support), export helpers, and a smoke-test that runs one epoch.
- `config_schema.yaml` — Minimal configuration template matching the notebook's `config` structure.

## Quick start

1. Create and activate a Python environment (recommended):

	```powershell
	python -m venv .venv; .\.venv\Scripts\Activate.ps1
	```

2. Install dependencies (adjust versions as needed):

	```powershell
	pip install --upgrade pip
	pip install torch torchvision timm pyyaml ipywidgets pillow numpy
	```

	- If you do not have a CUDA-capable GPU or do not need `timm`, you can still run the notebook using CPU only; `timm` is optional but recommended for ViT backbones.

3. Launch Jupyter and open the notebook:

	```powershell
	jupyter notebook ViT_trainer.ipynb
	```

4. (Optional) Edit `config_schema.yaml` then run the first configuration cell to load your settings. Use the UI's Apply button to set `config` for the notebook run.

5. Run the smoke-test cell (near the bottom of the notebook) to verify a one-epoch run on a small synthetic dataset.

## Notes and recommendations

- The notebook is intentionally lightweight and educational. The detection head, loss (`dummy_loss`), and evaluation are placeholders — replace them with production-quality implementations (GIoU/DIoU/CIoU, objectness, classification losses, mAP evaluation) before use on real datasets.
- Use `config_schema.yaml` to store reproducible experiment settings. The notebook will load this YAML automatically if present.
- For large-scale training, consider adding: dataset pipeline for real YOLO-format datasets, advanced augmentations (Mosaic, MixUp), checkpointing, and integration with experiment logging (TensorBoard / Weights & Biases).

## Troubleshooting

- If `timm` import fails, install it with `pip install timm` or change the notebook to use a simple PyTorch backbone fallback.
- If Jupyter widgets don't render, enable widgets extension:

  ```powershell
  pip install ipywidgets
  jupyter nbextension enable --py widgetsnbextension
  ```

## Next steps I can help with

- Implement a full YOLO-style detection head and matching/assignment logic.
- Replace `dummy_loss` with GIoU + classification + objectness losses.
- Add a simple mAP@0.5 evaluator (or integrate COCO metrics via pycocotools).
- Create a small synthetic dataset on disk and CI-style smoke tests.

If you'd like any of the above, tell me which one to implement next and I'll update the notebook and tests.
