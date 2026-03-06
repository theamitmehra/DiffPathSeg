from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_pgm(path: Path) -> Tuple[int, int, bytes]:
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"{path.name}: unsupported PGM magic")

        dims_line = f.readline().strip()
        while dims_line.startswith(b"#"):
            dims_line = f.readline().strip()
        parts = dims_line.split()
        if len(parts) != 2:
            raise ValueError(f"{path.name}: invalid dimensions")

        width = int(parts[0])
        height = int(parts[1])

        maxval_line = f.readline().strip()
        while maxval_line.startswith(b"#"):
            maxval_line = f.readline().strip()
        maxval = int(maxval_line)
        if maxval != 255:
            raise ValueError(f"{path.name}: maxval must be 255")

        data = f.read()
        expected = width * height
        if len(data) != expected:
            raise ValueError(f"{path.name}: pixel length mismatch")
        return width, height, data


def _mask_area_ratio(mask_bytes: bytes) -> float:
    total = len(mask_bytes)
    if total == 0:
        return 0.0
    positives = sum(1 for b in mask_bytes if b > 127)
    return positives / total


def _mask_is_binary(mask_bytes: bytes) -> bool:
    for b in mask_bytes:
        if b not in (0, 255):
            return False
    return True


def _validate_directory(directory: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "directory": str(directory),
        "samples": 0,
        "valid": 0,
        "invalid": 0,
        "issues": [],
    }

    if not directory.exists():
        result["issues"].append({"type": "missing_directory", "message": f"{directory} does not exist"})
        return result

    mask_files = sorted(directory.glob("*_mask.pgm"))
    result["samples"] = len(mask_files)
    for mask_file in mask_files:
        sample_id = mask_file.name.replace("_mask.pgm", "")
        image_file = directory / f"{sample_id}.pgm"
        png_image_file = directory / f"{sample_id}.png"
        png_mask_file = directory / f"{sample_id}_mask.png"

        sample_issues: List[str] = []
        if not image_file.exists():
            sample_issues.append("missing_image_pgm")
        if not png_image_file.exists() or not png_mask_file.exists():
            sample_issues.append("missing_png_preview")

        if image_file.exists():
            try:
                iw, ih, image_bytes = _read_pgm(image_file)
            except Exception as exc:
                sample_issues.append(f"image_read_error:{exc}")
                iw, ih, image_bytes = 0, 0, b""
        else:
            iw, ih, image_bytes = 0, 0, b""

        try:
            mw, mh, mask_bytes = _read_pgm(mask_file)
        except Exception as exc:
            sample_issues.append(f"mask_read_error:{exc}")
            mw, mh, mask_bytes = 0, 0, b""

        if iw != mw or ih != mh:
            sample_issues.append("dimension_mismatch")

        if mask_bytes:
            area_ratio = _mask_area_ratio(mask_bytes)
            if area_ratio <= 0.0:
                sample_issues.append("empty_mask")
            if area_ratio >= 0.95:
                sample_issues.append("oversized_mask")
            if not _mask_is_binary(mask_bytes):
                sample_issues.append("non_binary_mask")

        if image_bytes and mask_bytes:
            fg = [image_bytes[i] for i, m in enumerate(mask_bytes) if m > 127]
            bg = [image_bytes[i] for i, m in enumerate(mask_bytes) if m <= 127]
            if fg and bg:
                fg_mean = sum(fg) / len(fg)
                bg_mean = sum(bg) / len(bg)
                if abs(fg_mean - bg_mean) < 1.0:
                    sample_issues.append("weak_foreground_contrast")

        if sample_issues:
            result["invalid"] += 1
            if len(result["issues"]) < 200:
                result["issues"].append(
                    {
                        "sample_id": sample_id,
                        "file": mask_file.name,
                        "problems": sample_issues,
                    }
                )
        else:
            result["valid"] += 1

    return result


def build_validation_report(job_id: str, output_root: Path) -> Dict[str, Any]:
    synthetic = _validate_directory(output_root / "synthetic")
    curated = _validate_directory(output_root / "curated")

    total = synthetic["samples"] + curated["samples"]
    valid = synthetic["valid"] + curated["valid"]
    invalid = synthetic["invalid"] + curated["invalid"]

    return {
        "job_id": job_id,
        "output_dir": str(output_root),
        "summary": {
            "total_samples": total,
            "valid_samples": valid,
            "invalid_samples": invalid,
            "pass_rate": (valid / total) if total else 0.0,
        },
        "synthetic": synthetic,
        "curated": curated,
    }
