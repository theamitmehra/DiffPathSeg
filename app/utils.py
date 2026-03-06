import json
import random
import struct
import zlib
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


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack("!I", len(data))
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return length + chunk_type + data + struct.pack("!I", crc)


def save_grayscale_png(arr: Image2D, path: Path) -> None:
    h = len(arr)
    w = len(arr[0]) if h else 0

    raw = bytearray()
    for row in arr:
        raw.append(0)
        raw.extend(_float_to_byte(v) for v in row)

    ihdr = struct.pack("!IIBBBBB", w, h, 8, 0, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=9)

    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    png.extend(_png_chunk(b"IHDR", ihdr))
    png.extend(_png_chunk(b"IDAT", idat))
    png.extend(_png_chunk(b"IEND", b""))

    with open(path, "wb") as f:
        f.write(png)


def save_binary_mask(mask: Image2D, path: Path) -> None:
    binary = [[1.0 if v > 0.5 else 0.0 for v in row] for row in mask]
    save_grayscale_image(binary, path)


def save_binary_mask_png(mask: Image2D, path: Path) -> None:
    binary = [[1.0 if v > 0.5 else 0.0 for v in row] for row in mask]
    save_grayscale_png(binary, path)


def write_json(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
