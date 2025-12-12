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

## Project card (optional)

If you want the GitHub repo to look great at a glance, add a couple of screenshots:

1. Run the notebook cells to produce plots/results.
2. Export a notebook to HTML and screenshot the key figures.

Example export:

```bash
jupyter nbconvert --to html Q2.ipynb
jupyter nbconvert --to html Q3.ipynb
```

Suggested screenshots to include (if available):

- **Q2:** training curves + segmentation overlay predictions
- **Q3:** distance histogram/SDR plot + qualitative fovea localization examples
- **Q5:** Gradio UI showing a sample prediction

Place images under `docs/` (e.g., `docs/q2_results.png`) and link them here once added.

Template (uncomment after you add the files):

<!--
### Screenshots

![Q2 training curves](docs/q2_training_curves.png)
![Q2 predictions](docs/q2_predictions.png)

![Q3 SDR curve](docs/q3_sdr_curve.png)
![Q3 examples](docs/q3_examples.png)

![Q5 Gradio UI](docs/q5_gradio_ui.png)
-->
