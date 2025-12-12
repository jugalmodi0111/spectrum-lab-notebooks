# Spectrum Lab — Notebooks (Q1–Q5)

A small collection of Jupyter notebooks covering dataset preparation, retinal fundus analysis with deep learning (IDRiD), secure model-weight loading, and deployment.

## Notebooks

- **Q1.ipynb — CIFAR-100 → ImageFolder Extraction**  
  Converts CIFAR-100 (python format) into an ImageFolder layout (`train/<class>/`, `test/<class>/`) for PyTorch.

- **Q2.ipynb — IDRiD Optic Disc Segmentation (U-Net)**  
  End-to-end optic disc segmentation with a U-Net-style model: data story, training, evaluation, and visualizations.

- **Q3.ipynb — IDRiD Fovea Center Detection (Heatmap Regression)**  
  Keypoint regression using heatmap supervision and distance-based evaluation.

- **Q4.ipynb — Secure, Non-Executable Weight Loading**  
  Demonstrates a safe loading approach without `pickle`/`torch.load` on untrusted data (integrity checks + strict validation).

- **Q5.ipynb — Multi-Task Deployment with Gradio**  
  Runs a Gradio demo for optic disc segmentation + fovea detection (requires a trained checkpoint).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## IDRiD dataset (Q2/Q3/Q5)

The IDRiD dataset is **not** included in this repo.

Default expected structure:

```
data/IDRiD/
  A. Segmentation/A. Segmentation/
  C. Localization/C. Localization/
```

You can override paths using environment variables:

- `IDRID_ROOT` — base folder containing `A. Segmentation/` and `C. Localization/`
- `IDRID_SEG_ROOT` — explicit path to `A. Segmentation/A. Segmentation` (Q2)
- `IDRID_LOC_ROOT` — explicit path to `C. Localization/C. Localization` (Q3)

Example:

```bash
export IDRID_ROOT="/path/to/IDRiD"
```

## Model checkpoint (Q5)

Q5 expects a trained multi-task checkpoint.

- Default: `models/best_multitask_model.pth`
- Override: `MODEL_PATH=/path/to/checkpoint.pth`

## Notes

- The notebooks are written to run on CPU/CUDA/Apple Silicon (MPS) where available.
- Outputs (`outputs/`, `results/`) are ignored by git.
