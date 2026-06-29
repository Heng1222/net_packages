from __future__ import annotations

import json
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PATH_FIELDS = (
    ("data", "input_path"),
    ("data", "prepared_dir"),
    ("conditions", "path"),
    ("output", "base_dir"),
)


def _nested_get(config: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def load_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    config = deepcopy(raw)
    root = Path(project_root).expanduser().resolve()
    for keys in PATH_FIELDS:
        value = _nested_get(config, keys)
        if value is None:
            continue
        resolved = Path(str(value)).expanduser()
        if not resolved.is_absolute():
            resolved = root / resolved
        target = config
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = str(resolved.resolve())
    _validate_config(config)
    config["_meta"] = {"project_root": str(root), "config_path": str(config_path)}
    return config


def _validate_config(config: dict[str, Any]) -> None:
    data = config.get("data", {})
    if not data.get("input_path"):
        raise ValueError("data.input_path is required.")
    split = data.get("split", {})
    ratios = [float(split.get(key, 0.0)) for key in ("train_ratio", "val_ratio", "test_ratio")]
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must be non-negative and sum to 1.0: {ratios}")
    if split.get("strategy") != "time":
        raise ValueError("This Step1 experiment currently supports data.split.strategy='time'.")
    model = config.get("model", {})
    if int(model.get("condition_dim", 0)) != 768:
        raise ValueError("model.condition_dim must be 768 for ModernBERT condition alignment.")
    if int(model.get("input_dim", 0)) != 768:
        raise ValueError("model.input_dim must be 768 for ModernBERT payload embeddings.")


def save_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = deepcopy(config)
    serializable.pop("_meta", None)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)


def make_run_dir(base_dir: str | Path) -> Path:
    base = Path(base_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    candidate = base / timestamp
    counter = 1
    while candidate.exists():
        candidate = base / f"{timestamp}_{counter:02d}"
        counter += 1
    for name in ("logs", "checkpoints", "scalers", "metrics", "plots", "reports", "embeddings"):
        (candidate / name).mkdir(parents=True, exist_ok=True)
    return candidate


def write_json(data: Any, path: str | Path) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
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
    except Exception as exc:
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


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)
