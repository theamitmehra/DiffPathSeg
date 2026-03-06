import json
import random
from pathlib import Path
from typing import Any, Dict, List

Image2D = List[List[float]]


def seed_everything(seed: int) -> None:
    random.seed(seed)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _float_to_byte(v: float) -> int:
    if v < 0.0:
        return 0
    if v > 1.0:
        return 255
    return int(v * 255.0)


def save_grayscale_image(arr: Image2D, path: Path) -> None:
    h = len(arr)
    w = len(arr[0]) if h else 0
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        row_bytes = bytearray()
        for row in arr:
            row_bytes.extend(_float_to_byte(v) for v in row)
        f.write(row_bytes)


def save_binary_mask(mask: Image2D, path: Path) -> None:
    binary = [[1.0 if v > 0.5 else 0.0 for v in row] for row in mask]
    save_grayscale_image(binary, path)


def write_json(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
