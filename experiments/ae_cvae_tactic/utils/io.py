from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def make_run_dir(base_dir: str | Path) -> Path:
    base = Path(base_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    candidate = base / timestamp
    counter = 1
    while candidate.exists():
        candidate = base / f"{timestamp}_{counter:02d}"
        counter += 1
    for name in ("logs", "checkpoints", "scalers", "latent", "metrics", "plots", "reports", "embeddings"):
        (candidate / name).mkdir(parents=True, exist_ok=True)
    return candidate


def write_json(data: Any, path: str | Path) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, Path):
            return str(value)
        return value

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(convert(data), handle, indent=2, ensure_ascii=False, allow_nan=False)


def capture_environment(path: str | Path) -> None:
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.splitlines()
    except Exception as exc:  # pragma: no cover - diagnostic only
        freeze = [f"pip freeze unavailable: {exc}"]
    write_json(
        {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": freeze,
        },
        path,
    )
