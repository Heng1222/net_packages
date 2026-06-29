from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .embedders import build_text_embedder


@dataclass(slots=True)
class ConditionEmbeddings:
    labels: list[str]
    matrix: np.ndarray
    metadata: dict[str, Any]

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[1])


def _read_condition_records(path: Path, fmt: str | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    actual = (fmt or path.suffix.lstrip(".")).lower()
    if actual in {"yaml", "yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif actual == "json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif actual == "csv":
        return {}, pd.read_csv(path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported condition format: {actual}")
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    body = data.get("tactics", data) if isinstance(data, dict) else data
    records: list[dict[str, Any]] = []
    if isinstance(body, dict):
        for key, value in body.items():
            record = dict(value)
            record.setdefault("_key", str(key))
            records.append(record)
    elif isinstance(body, list):
        records = [dict(item) for item in body]
    else:
        raise ValueError("Condition file must contain a mapping or list.")
    return metadata, records


def _cache_key(config: dict[str, Any], labels: list[str], texts: list[str]) -> str:
    digest = hashlib.sha256()
    for label, text in zip(labels, texts, strict=True):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    for key in (
        "embedder_backend",
        "embedder_model_name",
        "embedder_model_revision",
        "text_field",
        "normalize",
    ):
        digest.update(f"{key}={config.get(key)}".encode("utf-8"))
    return digest.hexdigest()[:24]


def load_condition_embeddings(
    config: dict[str, Any],
    observed_labels: list[str] | None,
    device: torch.device,
    cache_dir: Path,
) -> ConditionEmbeddings:
    path = Path(config["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Condition file not found: {path}")
    metadata, records = _read_condition_records(path, config.get("format"))
    label_field = config.get("label_field", "label")
    text_field = config.get("text_field", "description_full")
    exclude = set(map(str, config.get("exclude_labels", []) or []))
    by_label = {str(record.get(label_field, record.get("_key", ""))): record for record in records}
    if observed_labels is None:
        requested = sorted(label for label in by_label if label and label not in exclude)
    else:
        requested = sorted(label for label in set(map(str, observed_labels)) if label and label not in exclude)
    missing = [label for label in requested if label not in by_label]
    if missing:
        raise KeyError(f"Condition file is missing labels required by Step1: {missing}")

    texts = [str(by_label[label].get(text_field, "")) for label in requested]
    empty = [label for label, text in zip(requested, texts, strict=True) if not text]
    if empty:
        raise KeyError(f"Condition text field '{text_field}' is empty for: {empty}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(config, requested, texts)
    cache_path = cache_dir / f"conditions_{key}.npz"
    meta_path = cache_dir / f"conditions_{key}.json"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            matrix = archive["matrix"].astype(np.float32)
            labels = archive["labels"].astype(str).tolist()
        return ConditionEmbeddings(labels, matrix, {"cache_hit": True, "cache_path": str(cache_path), **metadata})

    embedder_config = {
        "backend": config.get("embedder_backend", "sentence_transformers"),
        "model_name": config.get("embedder_model_name"),
        "model_revision": config.get("embedder_model_revision"),
        "batch_size": config.get("batch_size", 4),
        "normalize": config.get("normalize", True),
        "output_dim": config.get("output_dim", 768),
    }
    embedder = build_text_embedder(embedder_config, device)
    matrix = embedder.encode(texts).astype(np.float32)
    np.savez_compressed(cache_path, labels=np.asarray(requested, dtype=str), matrix=matrix)
    meta_path.write_text(
        json.dumps(
            {
                "labels": requested,
                "cache_path": str(cache_path),
                "embedding_dim": int(matrix.shape[1]),
                "condition_file_metadata": metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return ConditionEmbeddings(requested, matrix, {"cache_hit": False, "cache_path": str(cache_path), **metadata})


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    normalized = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    return normalized @ normalized.T
