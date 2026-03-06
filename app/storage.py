import os
from pathlib import Path
from typing import Any, Dict, List


class ArtifactPersistenceError(RuntimeError):
    pass


def _collect_local_files(output_root: Path) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    for kind in ["synthetic", "curated"]:
        directory = output_root / kind
        if not directory.exists():
            continue
        for pattern in ["*.png", "*.pgm"]:
            for p in sorted(directory.glob(pattern)):
                files.append(
                    {
                        "kind": kind,
                        "name": p.name,
                        "size_bytes": p.stat().st_size,
                        "path": p,
                    }
                )
    return files


def _mime_for_name(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".pgm"):
        return "image/x-portable-graymap"
    return "application/octet-stream"


def _load_s3_client(region: str):
    try:
        import boto3
    except Exception as exc:  # pragma: no cover
        raise ArtifactPersistenceError("boto3 is required for ARTIFACT_STORE=s3") from exc

    kwargs: Dict[str, Any] = {}
    if region:
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


def persist_job_artifacts(job_id: str, output_root: Path) -> Dict[str, Any]:
    mode = os.getenv("ARTIFACT_STORE", "local").strip().lower()
    local_files = _collect_local_files(output_root)

    if mode == "local":
        manifest: List[Dict[str, Any]] = []
        for f in local_files:
            manifest.append(
                {
                    "kind": f["kind"],
                    "name": f["name"],
                    "size_bytes": f["size_bytes"],
                    "storage": "local",
                    "media_type": _mime_for_name(f["name"]),
                }
            )
        return {"mode": "local", "count": len(manifest), "files": manifest}

    if mode != "s3":
        raise ArtifactPersistenceError(f"unsupported ARTIFACT_STORE '{mode}'")

    bucket = os.getenv("S3_BUCKET", "").strip()
    if not bucket:
        raise ArtifactPersistenceError("S3_BUCKET must be set for ARTIFACT_STORE=s3")

    region = os.getenv("S3_REGION", "").strip()
    prefix = os.getenv("S3_PREFIX", "diffpathseg").strip().strip("/")
    public_base = os.getenv("S3_PUBLIC_BASE_URL", "").strip().rstrip("/")
    expiry = int(os.getenv("S3_PRESIGN_EXPIRY_SECONDS", "3600"))

    client = _load_s3_client(region)

    manifest: List[Dict[str, Any]] = []
    for f in local_files:
        key = f"{prefix}/{job_id}/{f['kind']}/{f['name']}"
        client.upload_file(
            Filename=str(f["path"]),
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ContentType": _mime_for_name(f["name"])},
        )

        if public_base:
            remote_url = f"{public_base}/{key}"
        else:
            remote_url = client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expiry,
            )

        manifest.append(
            {
                "kind": f["kind"],
                "name": f["name"],
                "size_bytes": f["size_bytes"],
                "storage": "s3",
                "bucket": bucket,
                "key": key,
                "media_type": _mime_for_name(f["name"]),
                "remote_url": remote_url,
            }
        )

    return {"mode": "s3", "bucket": bucket, "count": len(manifest), "files": manifest}


def refresh_remote_urls(manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not manifest:
        return []

    mode = (manifest[0].get("storage") or "").lower()
    if mode != "s3":
        return manifest

    public_base = os.getenv("S3_PUBLIC_BASE_URL", "").strip().rstrip("/")
    expiry = int(os.getenv("S3_PRESIGN_EXPIRY_SECONDS", "3600"))
    region = os.getenv("S3_REGION", "").strip()

    if public_base:
        refreshed: List[Dict[str, Any]] = []
        for item in manifest:
            out = dict(item)
            key = out.get("key", "")
            out["remote_url"] = f"{public_base}/{key}"
            refreshed.append(out)
        return refreshed

    client = _load_s3_client(region)
    refreshed = []
    for item in manifest:
        out = dict(item)
        bucket = out.get("bucket")
        key = out.get("key")
        if bucket and key:
            out["remote_url"] = client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expiry,
            )
        refreshed.append(out)
    return refreshed
