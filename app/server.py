import copy
import logging
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import AppConfig, config_to_dict, load_config, load_config_dict
from .pipeline import run_pipeline


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


app = FastAPI(title="DiffPathSeg API", version="1.0.0")
jobs = JobStore()
_max_concurrent_jobs = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "2")))
_worker_semaphore = threading.BoundedSemaphore(_max_concurrent_jobs)


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
