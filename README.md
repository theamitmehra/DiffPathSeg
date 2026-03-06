# Diffusion-Augmented Segmentation for Rare Medical Pathologies

This project now ships as a deployable API service with asynchronous job execution.

## What Is Productionized

- API server with FastAPI endpoints:
  - `GET /healthz`
  - `POST /v1/jobs`
  - `GET /v1/jobs/{job_id}`
- Background job execution with bounded concurrency.
- Per-job isolated output directories (`outputs/<job_id>/...`).
- API key support (`APP_API_KEY` via `x-api-key` header).
- Dockerized runtime for cloud deployment.

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run API server:

```bash
python -m app.main serve --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/healthz
```

Create job:

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H "Content-Type: application/json" \
  -d "{}"
```

Get job status:

```bash
curl http://localhost:8000/v1/jobs/<job_id>
```

## Free Deployment (Render)

This repo includes `render.yaml` and `Dockerfile` for one-click deployment from GitHub.

1. Push latest code to GitHub.
2. In Render, create a new Blueprint and select this repository.
3. Render picks `render.yaml` and deploys `diffpathseg-api`.
4. After deploy, call `<render-url>/healthz`.

Environment variables you can set:

- `APP_API_KEY`: protect API endpoints.
- `APP_DEFAULT_CONFIG`: default config path (default `configs/default.json`).
- `MAX_CONCURRENT_JOBS`: worker slot count.
- `LOG_LEVEL`: logging level.

## Important Free-Tier Limits

- Free instances can sleep and have cold starts.
- Ephemeral disk means generated outputs may not persist long-term.
- For durable production, use object storage (S3/GCS) and a paid always-on plan.

## Optional Advanced Backends

To use diffusion (`backend=ldm`) and torch QC (`qc.backend=torch`), install:

```bash
pip install torch numpy pillow diffusers transformers accelerate safetensors
```

Then run with `configs/ldm_torch.example.json` or pass override in the job request.
