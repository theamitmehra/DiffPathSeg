import copy
import logging
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .config import AppConfig, config_to_dict, load_config, load_config_dict
from .pipeline import run_pipeline
from .validation import build_validation_report


def _setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


_setup_logging()
logger = logging.getLogger("diffpathseg.api")


class JobRequest(BaseModel):
    config_path: Optional[str] = Field(default=None, description="Path to JSON config file")
    config_override: Dict[str, Any] = Field(default_factory=dict, description="Partial config override")


class JobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    metrics: Optional[Dict[str, Any]]
    error: Optional[str]


@dataclass
class JobRecord:
    job_id: str
    status: str = "queued"
    created_at: float = field(default_factory=lambda: time.time())
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str) -> JobRecord:
        with self._lock:
            job = JobRecord(job_id=job_id)
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


app = FastAPI(title="DiffPathSeg API", version="1.2.0")
jobs = JobStore()
_max_concurrent_jobs = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "2")))
_worker_semaphore = threading.BoundedSemaphore(_max_concurrent_jobs)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DiffPathSeg Console</title>
  <style>
    :root {
      --bg-1: #0f172a;
      --bg-2: #1e293b;
      --card: #0b1220;
      --card-border: #1f2a44;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #22d3ee;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--text);
      background: radial-gradient(circle at 10% 10%, #1d4ed8 0%, transparent 40%),
                  radial-gradient(circle at 90% 20%, #0f766e 0%, transparent 45%),
                  linear-gradient(145deg, var(--bg-1), var(--bg-2));
      display: grid;
      place-items: center;
      padding: 20px;
    }
    .panel {
      width: min(940px, 100%);
      background: linear-gradient(180deg, rgba(11,18,32,0.92), rgba(11,18,32,0.82));
      border: 1px solid var(--card-border);
      border-radius: 18px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.45);
      overflow: hidden;
    }
    .head {
      padding: 22px 24px;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
    }
    .title { margin: 0; font-size: 1.45rem; }
    .sub { margin: 0; color: var(--muted); font-size: 0.92rem; }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      padding: 20px 24px;
    }
    @media (max-width: 780px) { .grid { grid-template-columns: 1fr; } }
    .group { display: flex; flex-direction: column; gap: 6px; }
    label {
      font-size: 0.82rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }
    input, textarea {
      width: 100%;
      border-radius: 10px;
      border: 1px solid #334155;
      background: #0b1325;
      color: var(--text);
      padding: 10px 12px;
      font-size: 0.95rem;
    }
    textarea {
      min-height: 120px;
      resize: vertical;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .actions {
      display: flex;
      gap: 10px;
      padding: 0 24px 20px;
      flex-wrap: wrap;
    }
    button {
      border: none;
      border-radius: 10px;
      padding: 11px 16px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
    }
    .btn-primary {
      background: linear-gradient(90deg, var(--accent), #0ea5e9);
      color: #001018;
    }
    .btn-secondary {
      background: #1e293b;
      color: var(--text);
      border: 1px solid #334155;
    }
    .status {
      margin: 0 24px 12px;
      padding: 10px 12px;
      border-radius: 10px;
      background: #0f172a;
      border: 1px solid #334155;
      color: var(--muted);
      min-height: 42px;
    }
    .status.ok { border-color: #14532d; color: #bbf7d0; }
    .status.err { border-color: #7f1d1d; color: #fecaca; }
    .output {
      margin: 0 24px 14px;
      border-radius: 12px;
      border: 1px solid #334155;
      background: #020817;
      padding: 14px;
      overflow: auto;
      min-height: 130px;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.86rem;
      line-height: 1.45;
      color: #cbd5e1;
    }
    .artifacts {
      margin: 0 24px 24px;
      border-radius: 12px;
      border: 1px solid #334155;
      background: #020817;
      padding: 14px;
      min-height: 80px;
    }
    .artifacts a {
      color: #7dd3fc;
      text-decoration: none;
      margin-right: 12px;
      display: inline-block;
      margin-bottom: 8px;
    }
  </style>
</head>
<body>
  <main class="panel">
    <section class="head">
      <div>
        <h1 class="title">DiffPathSeg Console</h1>
        <p class="sub">Run synthetic generation jobs and monitor status</p>
      </div>
      <p class="sub" id="health">Checking health...</p>
    </section>

    <section class="grid">
      <div class="group">
        <label for="configPath">Config Path</label>
        <input id="configPath" value="configs/default.json" />
      </div>
      <div class="group">
        <label for="apiKey">API Key (Optional)</label>
        <input id="apiKey" type="password" placeholder="x-api-key header" />
      </div>
      <div class="group" style="grid-column: 1 / -1;">
        <label for="override">Config Override JSON (Optional)</label>
        <textarea id="override" placeholder='{"num_generate_attempts": 40, "max_curated_samples": 20}'></textarea>
      </div>
      <div class="group" style="grid-column: 1 / -1;">
        <label for="jobId">Track Existing Job ID (Optional)</label>
        <input id="jobId" placeholder="Paste job_id to fetch status" />
      </div>
    </section>

    <section class="actions">
      <button class="btn-primary" id="runBtn">Create Job</button>
      <button class="btn-secondary" id="checkBtn">Check Job</button>
      <button class="btn-secondary" id="valBtn">Validate Job</button>
      <button class="btn-secondary" id="artBtn">Load Artifacts</button>
    </section>

    <p class="status" id="status">Ready.</p>
    <pre class="output" id="output">No job yet.</pre>
    <div class="artifacts" id="artifacts">Artifacts will appear here after completion.</div>
  </main>

  <script>
    const statusEl = document.getElementById("status");
    const outputEl = document.getElementById("output");
    const healthEl = document.getElementById("health");
    const runBtn = document.getElementById("runBtn");
    const checkBtn = document.getElementById("checkBtn");
    const valBtn = document.getElementById("valBtn");
    const artBtn = document.getElementById("artBtn");
    const artifactsEl = document.getElementById("artifacts");

    function setStatus(msg, kind = "") {
      statusEl.textContent = msg;
      statusEl.className = kind ? `status ${kind}` : "status";
    }

    function getHeaders() {
      const headers = { "Content-Type": "application/json" };
      const apiKey = document.getElementById("apiKey").value.trim();
      if (apiKey) headers["x-api-key"] = apiKey;
      return headers;
    }

    function apiHeadersOnly() {
      const headers = {};
      const apiKey = document.getElementById("apiKey").value.trim();
      if (apiKey) headers["x-api-key"] = apiKey;
      return headers;
    }

    async function checkHealth() {
      try {
        const res = await fetch("/healthz");
        if (!res.ok) throw new Error("health endpoint failed");
        const data = await res.json();
        healthEl.textContent = `Service: ${data.status}`;
      } catch (e) {
        healthEl.textContent = "Service: unavailable";
      }
    }

    function renderArtifacts(payload) {
      const files = payload.files || [];
      if (!files.length) {
        artifactsEl.textContent = "No artifacts found for this job yet.";
        return;
      }
      const browserFiles = files.filter((f) => f.name.endsWith(".png"));
      const chosen = browserFiles.length ? browserFiles : files;
      const html = chosen.map((f) => `<a href="${f.url}" target="_blank" rel="noopener">${f.kind}: ${f.name}</a>`).join(" ");
      artifactsEl.innerHTML = html;
    }

    async function loadArtifacts(jobId) {
      if (!jobId) {
        setStatus("Enter a job_id first.", "err");
        return;
      }
      const res = await fetch(`/v1/jobs/${jobId}/artifacts`, { headers: apiHeadersOnly() });
      const text = await res.text();
      let payload = {};
      try { payload = JSON.parse(text); } catch { payload = { raw: text }; }

      if (!res.ok) {
        setStatus(`Artifacts fetch failed (${res.status}).`, "err");
        artifactsEl.textContent = JSON.stringify(payload, null, 2);
        return;
      }
      renderArtifacts(payload);
      setStatus(`Artifacts loaded for ${jobId}.`, "ok");
    }


    async function validateJob(jobId) {
      if (!jobId) {
        setStatus("Enter a job_id first.", "err");
        return;
      }
      const res = await fetch(`/v1/jobs/${jobId}/validation`, { headers: apiHeadersOnly() });
      const text = await res.text();
      let payload = {};
      try { payload = JSON.parse(text); } catch { payload = { raw: text }; }

      if (!res.ok) {
        setStatus(`Validation failed (${res.status}).`, "err");
        outputEl.textContent = JSON.stringify(payload, null, 2);
        return;
      }

      outputEl.textContent = JSON.stringify(payload, null, 2);
      const invalid = payload.summary ? payload.summary.invalid_samples : 0;
      const kind = invalid > 0 ? "err" : "ok";
      setStatus(`Validation complete for ${jobId}. invalid=${invalid}`, kind);
    }
    async function fetchJob(jobId, silent = false) {
      if (!jobId) {
        if (!silent) setStatus("Enter a job_id first.", "err");
        return;
      }

      const res = await fetch(`/v1/jobs/${jobId}`, { headers: apiHeadersOnly() });
      const text = await res.text();
      let payload = {};
      try { payload = JSON.parse(text); } catch { payload = { raw: text }; }

      if (!res.ok) {
        setStatus(`Job lookup failed (${res.status}).`, "err");
        outputEl.textContent = JSON.stringify(payload, null, 2);
        return;
      }

      outputEl.textContent = JSON.stringify(payload, null, 2);
      setStatus(`Job ${payload.job_id} is ${payload.status}.`, payload.status === "failed" ? "err" : "ok");
      if (payload.status === "completed") {
        await loadArtifacts(payload.job_id);
      }
      return payload;
    }

    async function createJob() {
      runBtn.disabled = true;
      setStatus("Creating job...", "");
      artifactsEl.textContent = "Waiting for new job artifacts...";

      let override = {};
      const rawOverride = document.getElementById("override").value.trim();
      if (rawOverride) {
        try {
          override = JSON.parse(rawOverride);
        } catch (e) {
          setStatus("Override JSON is invalid.", "err");
          runBtn.disabled = false;
          return;
        }
      }

      const payload = {
        config_path: document.getElementById("configPath").value.trim() || null,
        config_override: override,
      };

      try {
        const res = await fetch("/v1/jobs", {
          method: "POST",
          headers: getHeaders(),
          body: JSON.stringify(payload),
        });

        const text = await res.text();
        let data = {};
        try { data = JSON.parse(text); } catch { data = { raw: text }; }

        if (!res.ok) {
          setStatus(`Job creation failed (${res.status}).`, "err");
          outputEl.textContent = JSON.stringify(data, null, 2);
          return;
        }

        const jobId = data.job_id;
        document.getElementById("jobId").value = jobId;
        setStatus(`Job ${jobId} queued. Polling status...`, "ok");
        outputEl.textContent = JSON.stringify(data, null, 2);

        const timer = setInterval(async () => {
          const job = await fetchJob(jobId, true);
          if (!job) return;
          if (job.status === "completed" || job.status === "failed") {
            clearInterval(timer);
          }
        }, 2500);
      } catch (e) {
        setStatus(`Request failed: ${e.message}`, "err");
      } finally {
        runBtn.disabled = false;
      }
    }

    runBtn.addEventListener("click", createJob);
    checkBtn.addEventListener("click", async () => {
      const jobId = document.getElementById("jobId").value.trim();
      try { await fetchJob(jobId); } catch (e) { setStatus(`Request failed: ${e.message}`, "err"); }
    });
    valBtn.addEventListener("click", async () => {
      const jobId = document.getElementById("jobId").value.trim();
      try { await validateJob(jobId); } catch (e) { setStatus(`Request failed: ${e.message}`, "err"); }
    });
    artBtn.addEventListener("click", async () => {
      const jobId = document.getElementById("jobId").value.trim();
      try { await loadArtifacts(jobId); } catch (e) { setStatus(`Request failed: ${e.message}`, "err"); }
    });

    checkHealth();
  </script>
</body>
</html>
"""


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "diffpathseg",
        "max_concurrent_jobs": _max_concurrent_jobs,
        "time": time.time(),
    }


def _assert_auth(api_key: Optional[str]) -> None:
    expected = os.getenv("APP_API_KEY", "").strip()
    if not expected:
        return
    if api_key != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


def _run_job(job_id: str, cfg: AppConfig) -> None:
    got_slot = _worker_semaphore.acquire(blocking=True)
    if not got_slot:
        jobs.update(job_id, status="failed", error="could not acquire worker slot", finished_at=time.time())
        return

    try:
        jobs.update(job_id, status="running", started_at=time.time())
        logger.info("job started", extra={"job_id": job_id})
        metrics = run_pipeline(cfg, run_id=job_id)
        jobs.update(job_id, status="completed", finished_at=time.time(), metrics=metrics)
        logger.info("job completed", extra={"job_id": job_id})
    except Exception as exc:  # pragma: no cover
        err = f"{exc}\n{traceback.format_exc()}"
        jobs.update(job_id, status="failed", finished_at=time.time(), error=err)
        logger.exception("job failed", extra={"job_id": job_id})
    finally:
        _worker_semaphore.release()


def _job_output_root(job_id: str) -> Path:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if not job.metrics or not job.metrics.get("output_dir"):
        raise HTTPException(status_code=409, detail="job artifacts not ready")
    return Path(str(job.metrics["output_dir"]))


@app.post("/v1/jobs", response_model=JobResponse)
def create_job(payload: JobRequest, x_api_key: Optional[str] = Header(default=None)) -> JobResponse:
    _assert_auth(x_api_key)

    cfg_path = payload.config_path or os.getenv("APP_DEFAULT_CONFIG", "configs/default.json")
    base_cfg = load_config(cfg_path)
    merged_cfg = _deep_merge(config_to_dict(base_cfg), payload.config_override)
    cfg = load_config_dict(merged_cfg)

    job_id = str(uuid.uuid4())
    jobs.create(job_id)

    thread = threading.Thread(target=_run_job, args=(job_id, cfg), daemon=True)
    thread.start()
    return JobResponse(job_id=job_id, status="queued")


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, x_api_key: Optional[str] = Header(default=None)) -> JobStatusResponse:
    _assert_auth(x_api_key)

    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        metrics=job.metrics,
        error=job.error,
    )



@app.get("/v1/jobs/{job_id}/validation")
def validate_job(job_id: str, x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _assert_auth(x_api_key)

    root = _job_output_root(job_id)
    return build_validation_report(job_id=job_id, output_root=root)
@app.get("/v1/jobs/{job_id}/artifacts")
def list_artifacts(job_id: str, x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _assert_auth(x_api_key)

    root = _job_output_root(job_id)
    files: List[Dict[str, Any]] = []
    for kind in ["synthetic", "curated"]:
        d = root / kind
        if not d.exists():
            continue
        for pattern in ["*.png", "*.pgm"]:
            for p in sorted(d.glob(pattern)):
                files.append(
                    {
                        "kind": kind,
                        "name": p.name,
                        "size_bytes": p.stat().st_size,
                        "media_type": "image/png" if p.suffix.lower() == ".png" else "image/x-portable-graymap",
                        "url": f"/v1/jobs/{job_id}/artifacts/{kind}/{p.name}",
                    }
                )

    return {"job_id": job_id, "count": len(files), "files": files}


@app.get("/v1/jobs/{job_id}/artifacts/{bucket}/{filename}")
def download_artifact(job_id: str, bucket: str, filename: str, x_api_key: Optional[str] = Header(default=None)) -> FileResponse:
    _assert_auth(x_api_key)

    if bucket not in {"synthetic", "curated"}:
        raise HTTPException(status_code=400, detail="invalid artifact bucket")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid filename")

    root = _job_output_root(job_id)
    path = (root / bucket / filename).resolve()
    expected_parent = (root / bucket).resolve()
    if expected_parent not in path.parents:
        raise HTTPException(status_code=400, detail="invalid artifact path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")

    suffix = path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/x-portable-graymap"
    return FileResponse(path=str(path), filename=filename, media_type=media_type)














