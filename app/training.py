from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_pgm(path: Path) -> Tuple[int, int, bytes]:
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"{path.name}: unsupported PGM format")

        dims = f.readline().strip()
        while dims.startswith(b"#"):
            dims = f.readline().strip()
        w_h = dims.split()
        if len(w_h) != 2:
            raise ValueError(f"{path.name}: invalid dimensions")
        width = int(w_h[0])
        height = int(w_h[1])

        maxval = f.readline().strip()
        while maxval.startswith(b"#"):
            maxval = f.readline().strip()
        if int(maxval) != 255:
            raise ValueError(f"{path.name}: unsupported max value")

        data = f.read()
        if len(data) != width * height:
            raise ValueError(f"{path.name}: data length mismatch")
        return width, height, data


def _load_pairs(directory: Path) -> List[Tuple[str, bytes, bytes]]:
    pairs: List[Tuple[str, bytes, bytes]] = []
    for mask_path in sorted(directory.glob("*_mask.pgm")):
        sample_id = mask_path.name.replace("_mask.pgm", "")
        image_path = directory / f"{sample_id}.pgm"
        if not image_path.exists():
            continue

        iw, ih, image_bytes = _read_pgm(image_path)
        mw, mh, mask_bytes = _read_pgm(mask_path)
        if iw != mw or ih != mh:
            continue
        pairs.append((sample_id, image_bytes, mask_bytes))
    return pairs


def _train_threshold_model(train_pairs: List[Tuple[str, bytes, bytes]]) -> Dict[str, float]:
    fg_sum = 0.0
    fg_n = 0
    bg_sum = 0.0
    bg_n = 0

    for _, image_bytes, mask_bytes in train_pairs:
        for idx, pix in enumerate(image_bytes):
            if mask_bytes[idx] > 127:
                fg_sum += pix
                fg_n += 1
            else:
                bg_sum += pix
                bg_n += 1

    if fg_n == 0 or bg_n == 0:
        raise ValueError("insufficient foreground/background pixels for training")

    fg_mean = fg_sum / fg_n
    bg_mean = bg_sum / bg_n
    threshold = 0.5 * (fg_mean + bg_mean)
    fg_brighter = fg_mean >= bg_mean

    return {
        "threshold": threshold,
        "fg_mean": fg_mean,
        "bg_mean": bg_mean,
        "fg_brighter": 1.0 if fg_brighter else 0.0,
    }


def _predict_mask(image_bytes: bytes, model: Dict[str, float]) -> List[int]:
    threshold = model["threshold"]
    fg_brighter = model["fg_brighter"] >= 0.5

    pred: List[int] = []
    if fg_brighter:
        for pix in image_bytes:
            pred.append(1 if pix >= threshold else 0)
    else:
        for pix in image_bytes:
            pred.append(1 if pix <= threshold else 0)
    return pred


def _dice_iou(pred: List[int], mask_bytes: bytes) -> Tuple[float, float]:
    tp = 0
    fp = 0
    fn = 0
    for i, p in enumerate(pred):
        t = 1 if mask_bytes[i] > 127 else 0
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 1:
            fn += 1

    denom_dice = (2 * tp + fp + fn)
    dice = (2 * tp / denom_dice) if denom_dice else 1.0
    denom_iou = (tp + fp + fn)
    iou = (tp / denom_iou) if denom_iou else 1.0
    return dice, iou


def _evaluate(pairs: List[Tuple[str, bytes, bytes]], model: Dict[str, float]) -> Dict[str, float]:
    if not pairs:
        return {"dice": 0.0, "iou": 0.0}

    dice_sum = 0.0
    iou_sum = 0.0
    for _, image_bytes, mask_bytes in pairs:
        pred = _predict_mask(image_bytes, model)
        dice, iou = _dice_iou(pred, mask_bytes)
        dice_sum += dice
        iou_sum += iou

    n = len(pairs)
    return {"dice": dice_sum / n, "iou": iou_sum / n}


def run_training_experiment(job_id: str, output_root: Path, baseline_fraction: float = 0.25) -> Dict[str, Any]:
    curated_dir = output_root / "curated"
    pairs = _load_pairs(curated_dir)
    if len(pairs) < 6:
        raise ValueError("need at least 6 curated samples for train/eval")

    n = len(pairs)
    val_count = max(1, int(0.2 * n))
    train_pairs = pairs[:-val_count]
    val_pairs = pairs[-val_count:]

    if len(train_pairs) < 2:
        raise ValueError("not enough training samples after split")

    base_train_count = max(1, int(len(train_pairs) * baseline_fraction))
    base_train_count = min(base_train_count, len(train_pairs) - 1)
    baseline_train = train_pairs[:base_train_count]
    augmented_train = train_pairs

    baseline_model = _train_threshold_model(baseline_train)
    augmented_model = _train_threshold_model(augmented_train)

    baseline_metrics = _evaluate(val_pairs, baseline_model)
    augmented_metrics = _evaluate(val_pairs, augmented_model)

    return {
        "job_id": job_id,
        "output_dir": str(output_root),
        "split": {
            "curated_total": n,
            "train": len(train_pairs),
            "validation": len(val_pairs),
            "baseline_train": len(baseline_train),
            "augmented_train": len(augmented_train),
        },
        "baseline": baseline_metrics,
        "augmented": augmented_metrics,
        "lift": {
            "dice": augmented_metrics["dice"] - baseline_metrics["dice"],
            "iou": augmented_metrics["iou"] - baseline_metrics["iou"],
        },
        "note": "baseline uses smaller training subset; augmented uses full curated train split",
    }
