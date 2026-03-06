import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List


def _log_path() -> Path:
    configured = os.getenv("EXPERIMENTS_LOG_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path("outputs") / "experiments.jsonl"


def append_experiment(record: Dict[str, Any]) -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(record)
    payload.setdefault("recorded_at", time.time())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def list_experiments(limit: int = 30) -> List[Dict[str, Any]]:
    path = _log_path()
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    rows.sort(key=lambda x: float(x.get("recorded_at", 0.0)), reverse=True)
    return rows[: max(1, limit)]


def summarize_experiments(limit: int = 30) -> Dict[str, Any]:
    runs = list_experiments(limit=limit)
    best: Dict[str, Any] = {}
    best_lift = None
    for run in runs:
        lift = float(run.get("dice_lift", 0.0))
        if best_lift is None or lift > best_lift:
            best_lift = lift
            best = run

    return {
        "count": len(runs),
        "runs": runs,
        "best_by_dice_lift": best,
    }
