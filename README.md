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
- Optional durable artifact persistence to S3.
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

## Workflow In UI

After a job completes:

- `Validate Job`: QC report
- `Train Eval`: baseline-vs-augmented Dice/IoU
- `Load Artifacts`: image links (`.png` and `.pgm`)

## Durable Storage (S3)

By default, artifacts are local (`ARTIFACT_STORE=local`).
To persist artifacts after Render restarts, set these env vars:

- `ARTIFACT_STORE=s3`
- `S3_BUCKET=<your-bucket>`
- `S3_REGION=<aws-region>`
- `S3_PREFIX=diffpathseg` (optional)
- `S3_PRESIGN_EXPIRY_SECONDS=3600` (optional)
- `S3_PUBLIC_BASE_URL=https://<bucket-host>` (optional; if set, uses direct URLs)

When S3 is enabled, `/v1/jobs/{job_id}/artifacts` returns remote URLs.

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
- S3 variables above (optional)

## Important Free-Tier Limits

- Free instances can sleep and cold-start.
- Local disk is ephemeral; use S3 mode for durable artifacts.

## Optional Advanced Backends

To use diffusion (`backend=ldm`) and torch QC (`qc.backend=torch`), install:

```bash
pip install torch numpy pillow diffusers transformers accelerate safetensors
```

Use `configs/ldm_torch.example.json` or send `config_override` in the UI/API.
