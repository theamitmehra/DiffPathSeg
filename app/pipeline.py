from typing import Dict, List

from .config import AppConfig
from .dataset import build_ellipse_mask, generate_normal_images, sample_longitudinal_specs
from .generation import build_generator
from .qc import QualityValidator
from .utils import ensure_dirs, save_binary_mask, save_grayscale_image, write_json


def run_pipeline(cfg: AppConfig) -> Dict[str, float]:
    out_root = cfg.output_dir
    synthetic_dir = out_root / "synthetic"
    curated_dir = out_root / "curated"
    ensure_dirs(out_root, synthetic_dir, curated_dir)

    normals = generate_normal_images(cfg.num_normal_images, cfg.image_size, cfg.seed)
    specs = sample_longitudinal_specs(
        cfg.text_prompts,
        cfg.longitudinal.enable,
        cfg.longitudinal.timesteps,
        cfg.longitudinal.severity_min,
        cfg.longitudinal.severity_max,
    )

    generator = build_generator(cfg.backend, cfg.seed, cfg.ldm)
    validator = QualityValidator(backend=cfg.qc.backend, latent_size=cfg.qc.latent_size)

    accepted_count = 0
    attempts = 0
    ious: List[float] = []

    for idx in range(cfg.num_generate_attempts):
        if accepted_count >= cfg.max_curated_samples:
            break

        spec = specs[idx % len(specs)]
        normal = normals[idx % len(normals)]
        mask = build_ellipse_mask(cfg.image_size, spec.severity, seed=cfg.seed + idx)
        synth = generator.generate(normal, mask, spec.prompt, spec.severity)

        _ = validator.train_step(synth, mask)
        qc = validator.evaluate(
            synth,
            mask,
            min_iou=cfg.quality_thresholds.min_iou,
            min_area=cfg.quality_thresholds.min_area_ratio,
            max_area=cfg.quality_thresholds.max_area_ratio,
        )

        sample_id = f"sample_{idx:04d}"
        save_grayscale_image(synth, synthetic_dir / f"{sample_id}.pgm")
        save_binary_mask(mask, synthetic_dir / f"{sample_id}_mask.pgm")

        attempts += 1
        ious.append(qc.iou)

        if qc.accepted:
            save_grayscale_image(synth, curated_dir / f"{sample_id}.pgm")
            save_binary_mask(mask, curated_dir / f"{sample_id}_mask.pgm")
            accepted_count += 1

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    metrics = {
        "attempts": attempts,
        "accepted": accepted_count,
        "acceptance_rate": (accepted_count / attempts) if attempts else 0.0,
        "mean_iou": mean_iou,
        "generator_backend": cfg.backend,
        "qc_backend": cfg.qc.backend,
    }
    write_json(metrics, out_root / "metrics.json")
    return metrics
