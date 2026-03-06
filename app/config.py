from dataclasses import dataclass
from pathlib import Path
from typing import List
import json


@dataclass
class QualityThresholds:
    min_iou: float
    min_area_ratio: float
    max_area_ratio: float


@dataclass
class LongitudinalConfig:
    enable: bool
    timesteps: int
    severity_min: float
    severity_max: float


@dataclass
class LDMConfig:
    model_id: str
    device: str
    guidance_scale: float
    num_inference_steps: int
    strength: float


@dataclass
class QCConfig:
    backend: str
    latent_size: int


@dataclass
class AppConfig:
    seed: int
    backend: str
    output_dir: Path
    image_size: int
    num_normal_images: int
    num_generate_attempts: int
    max_curated_samples: int
    batch_size: int
    quality_thresholds: QualityThresholds
    longitudinal: LongitudinalConfig
    ldm: LDMConfig
    qc: QCConfig
    text_prompts: List[str]


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    thresholds = QualityThresholds(**raw["quality_thresholds"])
    longitudinal = LongitudinalConfig(**raw["longitudinal"])

    ldm_raw = raw.get("ldm", {})
    ldm = LDMConfig(
        model_id=str(ldm_raw.get("model_id", "runwayml/stable-diffusion-inpainting")),
        device=str(ldm_raw.get("device", "cpu")),
        guidance_scale=float(ldm_raw.get("guidance_scale", 7.5)),
        num_inference_steps=int(ldm_raw.get("num_inference_steps", 30)),
        strength=float(ldm_raw.get("strength", 0.9)),
    )

    qc_raw = raw.get("qc", {})
    qc = QCConfig(
        backend=str(qc_raw.get("backend", "heuristic")),
        latent_size=int(qc_raw.get("latent_size", 32)),
    )

    return AppConfig(
        seed=int(raw["seed"]),
        backend=str(raw["backend"]),
        output_dir=Path(raw["output_dir"]),
        image_size=int(raw["image_size"]),
        num_normal_images=int(raw["num_normal_images"]),
        num_generate_attempts=int(raw["num_generate_attempts"]),
        max_curated_samples=int(raw["max_curated_samples"]),
        batch_size=int(raw["batch_size"]),
        quality_thresholds=thresholds,
        longitudinal=longitudinal,
        ldm=ldm,
        qc=qc,
        text_prompts=list(raw["text_prompts"]),
    )
