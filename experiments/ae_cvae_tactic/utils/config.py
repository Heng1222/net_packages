from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PATH_FIELDS = (
    ("data", "input_path"),
    ("data", "metadata_path"),
    ("data", "embedder", "cache_dir"),
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


def _nested_set(config: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    target = config
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def load_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    config = deepcopy(raw)
    root = Path(project_root).expanduser().resolve()
    for keys in PATH_FIELDS:
        value = _nested_get(config, keys)
        if value is not None:
            resolved = Path(str(value)).expanduser()
            if not resolved.is_absolute():
                resolved = root / resolved
            _nested_set(config, keys, str(resolved.resolve()))
    validate_config(config)
    config["_meta"] = {"config_path": str(config_path), "project_root": str(root)}
    return config


def validate_config(config: dict[str, Any]) -> None:
    data = config.get("data", {})
    if not data.get("input_path"):
        raise ValueError("data.input_path is required.")
    split = data.get("split", {})
    ratios = [float(split.get(name, 0.0)) for name in ("train_ratio", "val_ratio", "test_ratio")]
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must be non-negative and sum to 1.0; got {ratios}")
    if split.get("strategy", "stratified") not in {"stratified", "random", "time"}:
        raise ValueError("data.split.strategy must be stratified, random, or time.")
    if config.get("preprocessing", {}).get("normalization", "none") not in {
        "standard", "minmax", "l2", "none"
    }:
        raise ValueError("preprocessing.normalization must be standard, minmax, l2, or none.")
    embedder = data.get("embedder", {})
    if data.get("payload_text_col") and embedder.get("backend") != "sentence_transformers":
        raise ValueError("Text payload input requires data.embedder.backend='sentence_transformers'.")
    if int(embedder.get("max_length", 1)) <= 0:
        raise ValueError("data.embedder.max_length must be positive.")
    if embedder.get("overflow_strategy", "error") not in {"error", "truncate"}:
        raise ValueError("data.embedder.overflow_strategy must be error or truncate.")
    cvae = config.get("model", {}).get("cvae", {})
    if cvae.get("latent_representation", "mu") not in {"mu", "z"}:
        raise ValueError("model.cvae.latent_representation must be 'mu' or 'z'.")
    if cvae.get("objective", "elbo") != "elbo":
        raise ValueError("model.cvae.objective must be 'elbo'.")
    if cvae.get("likelihood", "gaussian") != "gaussian":
        raise ValueError("model.cvae.likelihood must be 'gaussian'.")
    if cvae.get("reconstruction_loss", "mse") != "mse":
        raise ValueError("Gaussian ELBO requires model.cvae.reconstruction_loss='mse'.")
    if float(cvae.get("observation_variance", 1.0)) <= 0:
        raise ValueError("model.cvae.observation_variance must be positive.")
    if "beta" in cvae:
        raise ValueError("model.cvae.beta is no longer supported; standard ELBO uses KL weight 1.")
    score = config.get("evaluation", {}).get("compatibility_score", "reconstruction")
    if score != "reconstruction":
        raise ValueError("This experiment specification supports evaluation.compatibility_score='reconstruction'.")


def save_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = deepcopy(config)
    serializable.pop("_meta", None)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)
