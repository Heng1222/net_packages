from __future__ import annotations

import json
import logging
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
    ("data", "input_path"), ("data", "prepared_dir"),
    ("conditions", "path"), ("conditions", "cache_dir"),
    ("evaluation", "golden_path"), ("output", "base_dir"),
)


def load_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping.")
    config = deepcopy(config); root = Path(project_root).resolve()
    for keys in PATH_FIELDS:
        target: Any = config
        for key in keys[:-1]: target = target.get(key, {})
        value = target.get(keys[-1]) if isinstance(target, dict) else None
        if value is not None:
            resolved = Path(str(value)).expanduser()
            target[keys[-1]] = str((resolved if resolved.is_absolute() else root / resolved).resolve())
    validate_config(config)
    config["_meta"] = {"project_root": str(root), "config_path": str(config_path)}
    return config


def validate_config(config: dict[str, Any]) -> None:
    split = config["data"]["split"]
    ratios = [float(split[key]) for key in ("train_ratio", "val_ratio", "test_ratio")]
    if split.get("strategy") != "time" or abs(sum(ratios) - 1.0) > 1e-8 or min(ratios) < 0:
        raise ValueError("Time split ratios must be non-negative and sum to one.")
    if int(config["model"]["residual_dim"]) != 16:
        raise ValueError("UCD-CVAE v2.1 requires model.residual_dim=16.")
    if config["model"]["geometry_variant"] not in {"full_orthogonal", "common_removal_only"}:
        raise ValueError("Unsupported geometry variant.")
    benign = float(config["evaluation"]["benign_threshold"])
    block = float(config["evaluation"]["block_threshold"])
    if not 0 <= benign < block <= 1:
        raise ValueError("Thresholds must satisfy 0 <= benign < block <= 1.")


def make_run_dir(base_dir: str | Path) -> Path:
    base = Path(base_dir); stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    candidate = base / stamp; suffix = 1
    while candidate.exists():
        candidate = base / f"{stamp}_{suffix:02d}"; suffix += 1
    for name in ("logs", "checkpoints", "metrics", "plots", "reports", "embeddings"):
        (candidate / name).mkdir(parents=True, exist_ok=True)
    return candidate


def configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("ucd_cvae_v2_1"); logger.setLevel(logging.INFO); logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def write_json(value: Any, path: str | Path) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, dict): return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)): return [convert(v) for v in item]
        if isinstance(item, (np.ndarray,)): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, Path): return str(item)
        if isinstance(item, float) and not np.isfinite(item): return None
        return item
    Path(path).write_text(json.dumps(convert(value), indent=2, ensure_ascii=False), encoding="utf-8")


def save_config(config: dict[str, Any], path: Path) -> None:
    body = deepcopy(config); body.pop("_meta", None)
    path.write_text(yaml.safe_dump(body, sort_keys=False, allow_unicode=True), encoding="utf-8")


def seed_everything(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def capture_environment(path: Path) -> None:
    try:
        packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                  text=True, timeout=60, check=False).stdout.splitlines()
    except Exception as exc:
        packages = [f"unavailable: {exc}"]
    write_json({"python": sys.version, "platform": platform.platform(), "packages": packages}, path)
