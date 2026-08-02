from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from experiments.disentangled_cvae_step1.utils import (
    capture_environment,
    make_run_dir,
    resolve_device,
    save_config,
    write_json,
)
from experiments.disentangled_cvae_uwf_zeekdata24.download import SOURCE_CATEGORIES


PATH_FIELDS = (
    ("data", "raw_dir"),
    ("data", "prepared_dir"),
    ("conditions", "path"),
    ("output", "base_dir"),
)


def load_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    config = deepcopy(raw)
    root = Path(project_root).expanduser().resolve()
    for section, key in PATH_FIELDS:
        value = config.get(section, {}).get(key)
        if value is None:
            continue
        resolved = Path(str(value)).expanduser()
        if not resolved.is_absolute():
            resolved = root / resolved
        config[section][key] = str(resolved.resolve())
    validate_config(config)
    config["_meta"] = {"project_root": str(root), "config_path": str(config_path)}
    return config


def validate_config(config: dict[str, Any]) -> None:
    data = config.get("data", {})
    for key in ("raw_dir", "prepared_dir", "source_base_url"):
        if not data.get(key):
            raise ValueError(f"data.{key} is required.")
    categories = data.get("source_categories", [])
    if tuple(map(str, categories)) != SOURCE_CATEGORIES:
        raise ValueError(
            "data.source_categories must list the eight official UWF CSV categories in canonical order."
        )
    ratios = [float(data.get("split", {}).get(key, 0.0)) for key in ("train_ratio", "val_ratio", "test_ratio")]
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must be non-negative and sum to 1.0: {ratios}")
    if data.get("split", {}).get("strategy") != "stratified_technique":
        raise ValueError("Only data.split.strategy='stratified_technique' is supported.")
    model = config.get("model", {})
    if int(model.get("input_dim", 0)) != 768 or int(model.get("condition_dim", 0)) != 768:
        raise ValueError("ModernBERT input_dim and condition_dim must both be 768.")
    controls = config.get("controls")
    if controls is not None:
        allowed = {
            "semantic", "random_gaussian", "random_orthogonal", "semantic_label_shuffle"
        }
        variants = list(map(str, controls.get("variants", [])))
        if "semantic" not in variants or not set(variants).issubset(allowed) or len(set(variants)) != len(variants):
            raise ValueError("controls.variants must contain semantic and unique supported controls.")
        seeds = list(map(int, controls.get("seeds", [])))
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("controls.seeds must contain at least one unique seed.")
        if int(controls.get("bootstrap_repeats", 0)) <= 0:
            raise ValueError("controls.bootstrap_repeats must be positive.")
        fraction = float(controls.get("minimum_seed_fraction", 0.8))
        if not 0.0 < fraction <= 1.0:
            raise ValueError("controls.minimum_seed_fraction must be in (0, 1].")


__all__ = [
    "capture_environment",
    "load_config",
    "make_run_dir",
    "resolve_device",
    "save_config",
    "write_json",
]
