from abc import ABC, abstractmethod
from typing import List, Optional
import random

from .config import LDMConfig
from .dataset import pathology_signature

Image2D = List[List[float]]


class InpaintGenerator(ABC):
    @abstractmethod
    def generate(self, normal_image: Image2D, mask: Image2D, prompt: str, severity: float) -> Image2D:
        raise NotImplementedError


class MockInpaintGenerator(InpaintGenerator):
    def __init__(self, seed: int) -> None:
        random.seed(seed)

    def generate(self, normal_image: Image2D, mask: Image2D, prompt: str, severity: float) -> Image2D:
        intensity, noise_scale = pathology_signature(prompt)
        out: Image2D = []
        for y, row in enumerate(normal_image):
            out_row: List[float] = []
            for x, pix in enumerate(row):
                lesion = min(1.0, max(0.0, intensity * severity + random.gauss(0.0, noise_scale)))
                m = mask[y][x]
                value = pix * (1.0 - m) + lesion * m
                out_row.append(min(1.0, max(0.0, value)))
            out.append(out_row)
        return out


class LDMInpaintGenerator(InpaintGenerator):
    """
    Optional diffusers backend.

    Requires external dependencies:
    - torch
    - numpy
    - pillow
    - diffusers
    """

    def __init__(self, ldm_cfg: LDMConfig) -> None:
        self.cfg = ldm_cfg
        self._pipe: Optional[object] = None

    def _lazy_init(self) -> None:
        if self._pipe is not None:
            return

        try:
            import torch
            from diffusers import AutoPipelineForInpainting
        except Exception as exc:
            raise RuntimeError(
                "LDM backend needs torch + diffusers installed. "
                "Set backend='mock' or install required packages."
            ) from exc

        dtype = torch.float16 if self.cfg.device.startswith("cuda") else torch.float32
        pipe = AutoPipelineForInpainting.from_pretrained(self.cfg.model_id, torch_dtype=dtype)
        pipe = pipe.to(self.cfg.device)
        self._pipe = pipe

    def generate(self, normal_image: Image2D, mask: Image2D, prompt: str, severity: float) -> Image2D:
        self._lazy_init()

        try:
            import numpy as np
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("LDM backend needs numpy + pillow installed.") from exc

        img = np.array(normal_image, dtype=np.float32)
        msk = np.array(mask, dtype=np.float32)

        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        msk_u8 = (msk > 0.5).astype(np.uint8) * 255

        image_pil = Image.fromarray(img_u8, mode="L").convert("RGB")
        mask_pil = Image.fromarray(msk_u8, mode="L")

        prompt_full = f"{prompt}, severity {severity:.2f}, medical imaging style"

        result = self._pipe(
            prompt=prompt_full,
            image=image_pil,
            mask_image=mask_pil,
            guidance_scale=self.cfg.guidance_scale,
            num_inference_steps=self.cfg.num_inference_steps,
            strength=self.cfg.strength,
        )

        out = result.images[0].convert("L")
        out_np = np.array(out, dtype=np.float32) / 255.0
        return out_np.tolist()


def build_generator(backend: str, seed: int, ldm_cfg: LDMConfig) -> InpaintGenerator:
    b = backend.lower().strip()
    if b == "mock":
        return MockInpaintGenerator(seed=seed)
    if b == "ldm":
        return LDMInpaintGenerator(ldm_cfg=ldm_cfg)
    raise ValueError(f"Unsupported backend '{backend}'. Use 'mock' or 'ldm'.")
