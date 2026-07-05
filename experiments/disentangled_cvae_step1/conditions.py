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


def _record_value(record: dict[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set)):
        parts: list[str] = []
        for item in value:
            parts.extend(_flatten_text(item))
        return parts
    stripped = str(value).strip()
    return [stripped] if stripped else []


def _condition_text_fields(config: dict[str, Any]) -> list[str]:
    fields = config.get("text_fields")
    if fields is None:
        fields = [config.get("text_field", "description_full")]
    elif isinstance(fields, str):
        fields = [fields]
    fields = [str(field) for field in fields if str(field).strip()]
    if not fields:
        raise ValueError("conditions.text_fields must contain at least one field.")
    return fields


def _condition_text(record: dict[str, Any], fields: list[str]) -> tuple[str, list[str]]:
    terms: list[str] = []
    missing: list[str] = []
    for field in fields:
        value = _record_value(record, field)
        if value is None:
            missing.append(field)
            continue
        terms.extend(_flatten_text(value))
    return ", ".join(terms), missing


def _cache_key(config: dict[str, Any], labels: list[str], texts: list[str], text_fields: list[str]) -> str:
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
        "normalize",
    ):
        digest.update(f"{key}={config.get(key)}".encode("utf-8"))
    digest.update(json.dumps({"text_fields": text_fields}, sort_keys=True).encode("utf-8"))
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
    text_fields = _condition_text_fields(config)
    exclude = set(map(str, config.get("exclude_labels", []) or []))
    by_label = {str(record.get(label_field, record.get("_key", ""))): record for record in records}
    if observed_labels is None:
        requested = sorted(label for label in by_label if label and label not in exclude)
    else:
        requested = sorted(label for label in set(map(str, observed_labels)) if label and label not in exclude)
    missing = [label for label in requested if label not in by_label]
    if missing:
        raise KeyError(f"Condition file is missing labels required by Step1: {missing}")

    texts: list[str] = []
    missing_fields: dict[str, list[str]] = {}
    for label in requested:
        text, missing = _condition_text(by_label[label], text_fields)
        texts.append(text)
        if missing:
            missing_fields[label] = missing
    if missing_fields:
        raise KeyError(f"Condition text field(s) are missing: {missing_fields}")
    empty = [label for label, text in zip(requested, texts, strict=True) if not text]
    if empty:
        raise KeyError(f"Condition text fields {text_fields} are empty for: {empty}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(config, requested, texts, text_fields)
    cache_path = cache_dir / f"conditions_{key}.npz"
    meta_path = cache_dir / f"conditions_{key}.json"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            matrix = archive["matrix"].astype(np.float32)
            labels = archive["labels"].astype(str).tolist()
        return ConditionEmbeddings(
            labels,
            matrix,
            {"cache_hit": True, "cache_path": str(cache_path), "text_fields": text_fields, **metadata},
        )

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
                "text_fields": text_fields,
                "text_preview": dict(zip(requested, texts, strict=True)),
                "condition_file_metadata": metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return ConditionEmbeddings(
        requested,
        matrix,
        {"cache_hit": False, "cache_path": str(cache_path), "text_fields": text_fields, **metadata},
    )


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    normalized = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    return normalized @ normalized.T
