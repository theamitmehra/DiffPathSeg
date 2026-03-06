# Diffusion-Augmented Segmentation for Rare Medical Pathologies

This project is a runnable starter application for a DiffAug-style pipeline:

- Generates synthetic pathology examples with mask-guided inpainting.
- Validates generated samples with a latent-space quality network.
- Curates high-quality samples into a training-ready synthetic dataset.

## Features

- Generator backends:
  - `mock`: zero-dependency synthetic inpainting
  - `ldm`: `diffusers` inpainting backend (lazy-loaded)
- QC backends:
  - `heuristic`: zero-dependency latent validator
  - `torch`: trainable latent segmentation validator
- Longitudinal cohort hooks:
  - severity progression simulation across timesteps

## Quickstart (Zero-Dependency)

```bash
python -m app.main run --config configs/default.json
```

## Advanced Run (LDM + Torch)

Install dependencies in your main Python environment:

```bash
pip install torch numpy pillow diffusers transformers accelerate safetensors
```

Run with:

```bash
python -m app.main run --config configs/ldm_torch.example.json
```

## Outputs

- `outputs/synthetic`: all generated image/mask pairs (`.pgm`)
- `outputs/curated`: accepted pairs (`.pgm`)
- `outputs/metrics.json`: run summary

## Project Layout

```text
app/
  main.py
  config.py
  pipeline.py
  qc.py
  generation.py
  dataset.py
  utils.py
configs/
  default.json
  ldm_torch.example.json
```
