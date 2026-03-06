# Diffusion-Augmented Segmentation for Rare Medical Pathologies

This project ships as a deployable API + browser UI.

## What Is Productionized

- FastAPI service endpoints:
  - `GET /` (web dashboard)
  - `GET /healthz`
  - `POST /v1/jobs`
  - `GET /v1/jobs/{job_id}`
  - `GET /v1/jobs/{job_id}/validation`
  - `GET /v1/jobs/{job_id}/train_eval`
  - `GET /v1/jobs/{job_id}/artifacts`
  - `GET /v1/jobs/{job_id}/artifacts/{bucket}/{filename}`
- Background job execution with bounded concurrency.
- Per-job isolated output directories (`outputs/<job_id>/...`).
- Optional API key protection (`APP_API_KEY` via `x-api-key`).
- Dockerized runtime for cloud deployment.

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start server:

```bash
python -m app.main serve --host 0.0.0.0 --port 8000
```

Open UI:

- `http://localhost:8000/`

## Validation and Artifacts

After a job completes, click `Validate Job` for QC, `Train Eval` for baseline-vs-augmented Dice/IoU, then `Load Artifacts` to browse files.
The UI shows clickable links for generated `.png` previews (and `.pgm` originals) from:

- `synthetic`
- `curated`

## Free Deployment (Render)

This repo includes `render.yaml` and `Dockerfile`.

1. Push latest code to GitHub.
2. In Render, create a Blueprint from this repo.
3. Render deploys `diffpathseg-api`.
4. Open `<render-url>/` for UI and `<render-url>/healthz` for health.

Environment variables:

- `APP_API_KEY`
- `APP_DEFAULT_CONFIG` (default `configs/default.json`)
- `MAX_CONCURRENT_JOBS`
- `LOG_LEVEL`

## Important Free-Tier Limits

- Free instances can sleep and cold-start.
- Disk is ephemeral; outputs are not durable long-term.
- For durable production use object storage (S3/GCS) and always-on compute.

## Optional Advanced Backends

To use diffusion (`backend=ldm`) and torch QC (`qc.backend=torch`), install:

```bash
pip install torch numpy pillow diffusers transformers accelerate safetensors
```

Use `configs/ldm_torch.example.json` or send `config_override` in the UI/API.

