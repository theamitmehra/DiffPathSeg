from dataclasses import dataclass
from typing import List, Tuple

Image2D = List[List[float]]


@dataclass
class QCResult:
    iou: float
    area_ratio: float
    accepted: bool


class HeuristicLatentValidator:
    def __init__(self, latent_size: int = 32) -> None:
        self.latent_size = latent_size
        self.threshold = 0.6

    def _downsample(self, image: Image2D) -> Image2D:
        h = len(image)
        w = len(image[0]) if h else 0
        lh = self.latent_size
        lw = self.latent_size
        out: Image2D = [[0.0 for _ in range(lw)] for _ in range(lh)]

        for ly in range(lh):
            y0 = int(ly * h / lh)
            y1 = int((ly + 1) * h / lh)
            for lx in range(lw):
                x0 = int(lx * w / lw)
                x1 = int((lx + 1) * w / lw)
                total = 0.0
                cnt = 0
                for y in range(y0, max(y0 + 1, y1)):
                    for x in range(x0, max(x0 + 1, x1)):
                        total += image[y][x]
                        cnt += 1
                out[ly][lx] = total / max(1, cnt)
        return out

    def _upsample_binary(self, latent_mask: Image2D, h: int, w: int) -> Image2D:
        lh = len(latent_mask)
        lw = len(latent_mask[0]) if lh else 0
        out: Image2D = [[0.0 for _ in range(w)] for _ in range(h)]
        for y in range(h):
            ly = min(lh - 1, int(y * lh / h))
            for x in range(w):
                lx = min(lw - 1, int(x * lw / w))
                out[y][x] = 1.0 if latent_mask[ly][lx] > 0.5 else 0.0
        return out

    def _mean_inside_outside(self, image: Image2D, mask: Image2D) -> Tuple[float, float]:
        in_total, in_cnt = 0.0, 0
        out_total, out_cnt = 0.0, 0
        for y in range(len(image)):
            for x in range(len(image[0])):
                if mask[y][x] > 0.5:
                    in_total += image[y][x]
                    in_cnt += 1
                else:
                    out_total += image[y][x]
                    out_cnt += 1
        mean_in = in_total / max(1, in_cnt)
        mean_out = out_total / max(1, out_cnt)
        return mean_in, mean_out

    def train_step(self, image: Image2D, mask: Image2D) -> float:
        mean_in, mean_out = self._mean_inside_outside(image, mask)
        target_threshold = (mean_in + mean_out) * 0.5
        self.threshold = 0.85 * self.threshold + 0.15 * target_threshold
        return abs(mean_in - mean_out)

    def predict_mask(self, image: Image2D) -> Image2D:
        latent = self._downsample(image)
        latent_pred = [[1.0 if v >= self.threshold else 0.0 for v in row] for row in latent]
        h = len(image)
        w = len(image[0]) if h else 0
        return self._upsample_binary(latent_pred, h, w)


class TorchLatentValidator:
    def __init__(self, latent_size: int = 32) -> None:
        try:
            import numpy as np
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except Exception as exc:
            raise RuntimeError(
                "qc.backend='torch' requires torch and numpy installed. Use qc.backend='heuristic' instead."
            ) from exc

        self.np = np
        self.torch = torch
        self.nn = nn

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(64, 32, 2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(32, 16, 2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 1, 1),
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _Net().to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.latent_size = latent_size

    def train_step(self, image: Image2D, mask: Image2D) -> float:
        x = self.torch.tensor(image, dtype=self.torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        y = self.torch.tensor(mask, dtype=self.torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def predict_mask(self, image: Image2D) -> Image2D:
        with self.torch.no_grad():
            self.model.eval()
            x = self.torch.tensor(image, dtype=self.torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            pred = self.torch.sigmoid(self.model(x))[0, 0].detach().cpu().numpy()
        return (pred > 0.5).astype(self.np.float32).tolist()


class QualityValidator:
    def __init__(self, backend: str = "heuristic", latent_size: int = 32) -> None:
        b = backend.lower().strip()
        if b == "heuristic":
            self.impl = HeuristicLatentValidator(latent_size=latent_size)
        elif b == "torch":
            self.impl = TorchLatentValidator(latent_size=latent_size)
        else:
            raise ValueError("Unsupported qc backend. Use 'heuristic' or 'torch'.")

    def train_step(self, image: Image2D, mask: Image2D) -> float:
        return self.impl.train_step(image, mask)

    def evaluate(self, image: Image2D, target_mask: Image2D, min_iou: float, min_area: float, max_area: float) -> QCResult:
        pred = self.impl.predict_mask(image)

        h = len(target_mask)
        w = len(target_mask[0]) if h else 0
        inter = 0.0
        union = 0.0
        area = 0.0
        total = float(h * w) if h and w else 1.0
        for y in range(h):
            for x in range(w):
                p = 1.0 if pred[y][x] > 0.5 else 0.0
                t = 1.0 if target_mask[y][x] > 0.5 else 0.0
                if p > 0.5 and t > 0.5:
                    inter += 1.0
                if p > 0.5 or t > 0.5:
                    union += 1.0
                area += t

        iou = 0.0 if union == 0 else inter / union
        area_ratio = area / total
        accepted = iou >= min_iou and min_area <= area_ratio <= max_area
        return QCResult(iou=iou, area_ratio=area_ratio, accepted=accepted)
