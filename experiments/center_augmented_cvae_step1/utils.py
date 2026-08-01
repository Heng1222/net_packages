from __future__ import annotations

import json
import logging
import platform
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PATH_FIELDS = (
    ("data", "input_path"), ("data", "prepared_dir"),
    ("conditions", "path"), ("conditions", "cache_dir"),
    ("evaluation", "golden_path"), ("output", "base_dir"),
)


def load_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")
    config = deepcopy(raw)
    root = Path(project_root).expanduser().resolve()
    for keys in PATH_FIELDS:
        value: Any = config
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if value is None:
            continue
        resolved = Path(str(value)).expanduser()
        if not resolved.is_absolute():
            resolved = root / resolved
        target = config
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = str(resolved.resolve())
    validate_config(config)
    config["_meta"] = {"project_root": str(root), "config_path": str(config_path)}
    return config


def validate_config(config: dict[str, Any]) -> None:
    split = config.get("data", {}).get("split", {})
    ratios = [float(split.get(key, 0.0)) for key in ("train_ratio", "val_ratio", "test_ratio")]
    if split.get("strategy") != "time" or any(x < 0 for x in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("A non-negative time split summing to one is required.")
    if int(config.get("model", {}).get("input_dim", 0)) <= 0:
        raise ValueError("model.input_dim must be positive.")
    if float(config.get("model", {}).get("gate_temperature", 0.0)) <= 0:
        raise ValueError("model.gate_temperature must be positive.")
    if not bool(config.get("data", {}).get("embedder", {}).get("normalize", False)):
        raise ValueError("Payload embeddings must be L2-normalized for this experiment.")


def save_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = deepcopy(config)
    serializable.pop("_meta", None)
    Path(path).write_text(yaml.safe_dump(serializable, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_run_dir(base_dir: str | Path) -> Path:
    base = Path(base_dir)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    candidate = base / stamp
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stamp}_{suffix:02d}"
        suffix += 1
    for name in ("logs", "checkpoints", "metrics", "plots", "reports", "embeddings"):
        (candidate / name).mkdir(parents=True, exist_ok=True)
    return candidate


def configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("center_augmented_cvae_step1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_json(data: Any, path: str | Path) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, dict): return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [convert(v) for v in value]
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, Path): return str(value)
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    Path(path).write_text(json.dumps(convert(data), indent=2, ensure_ascii=False), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def capture_environment(path: Path) -> None:
    write_json({"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "numpy": np.__version__, "cuda_available": torch.cuda.is_available()}, path)
