from dataclasses import dataclass
from typing import List, Tuple
import math
import random

Image2D = List[List[float]]


@dataclass
class SampleSpec:
    prompt: str
    severity: float


def generate_normal_images(num_images: int, image_size: int, seed: int) -> List[Image2D]:
    random.seed(seed)
    images: List[Image2D] = []
    for _ in range(num_images):
        img: Image2D = []
        for y in range(image_size):
            row: List[float] = []
            for x in range(image_size):
                fx = x / max(1, image_size - 1)
                fy = y / max(1, image_size - 1)
                base = 0.2 + 0.5 * fx + 0.2 * math.sin(fy * math.pi)
                noise = random.gauss(0.0, 0.03)
                row.append(min(1.0, max(0.0, base + noise)))
            img.append(row)
        images.append(img)
    return images


def sample_longitudinal_specs(
    prompts: List[str],
    enable: bool,
    timesteps: int,
    severity_min: float,
    severity_max: float,
) -> List[SampleSpec]:
    specs: List[SampleSpec] = []
    if not enable:
        for p in prompts:
            specs.append(SampleSpec(prompt=p, severity=0.5))
        return specs

    for p in prompts:
        for t in range(max(1, timesteps)):
            alpha = t / max(1, timesteps - 1)
            severity = severity_min + alpha * (severity_max - severity_min)
            specs.append(SampleSpec(prompt=p, severity=float(severity)))
    return specs


def build_ellipse_mask(image_size: int, severity: float, seed: int) -> Image2D:
    random.seed(seed)
    h = w = image_size
    cy = random.randint(int(0.3 * h), int(0.7 * h))
    cx = random.randint(int(0.3 * w), int(0.7 * w))
    ry = int((0.04 + 0.15 * severity) * h)
    rx = int((0.04 + 0.15 * severity) * w)

    mask: Image2D = []
    for y in range(h):
        row: List[float] = []
        for x in range(w):
            nx = ((x - cx) ** 2) / max(1, rx * rx)
            ny = ((y - cy) ** 2) / max(1, ry * ry)
            inside = 1.0 if (nx + ny) <= 1.0 else 0.0
            jitter = random.gauss(0.0, 0.1)
            soft = min(1.0, max(0.0, inside + jitter * inside))
            row.append(1.0 if soft > 0.4 else 0.0)
        mask.append(row)
    return mask


def pathology_signature(prompt: str) -> Tuple[float, float]:
    p = prompt.lower()
    if "lung" in p:
        return 0.9, 0.15
    if "liver" in p:
        return 0.75, 0.1
    if "white matter" in p or "brain" in p:
        return 0.6, 0.08
    return 0.7, 0.12
