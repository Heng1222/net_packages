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
    tactic_labels: list[str]
    matrix: np.ndarray
    metadata: dict[str, Any]
    raw_matrix: np.ndarray | None = None

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


def _normalize_rows(matrix: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.clip(norms, 1e-12, None)
    if fallback is not None:
        small = norms[:, 0] < 1e-12
        if np.any(small):
            normalized[small] = _normalize_rows(fallback)[small]
    return normalized.astype(np.float32)


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    normalized = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    return normalized @ normalized.T


def cosine_similarity_summary(matrix: np.ndarray) -> dict[str, float]:
    similarity = cosine_similarity_matrix(np.asarray(matrix, dtype=np.float32))
    if similarity.shape[0] < 2:
        return {
            "offdiag_min": 0.0,
            "offdiag_p25": 0.0,
            "offdiag_mean": 0.0,
            "offdiag_median": 0.0,
            "offdiag_p75": 0.0,
            "offdiag_max": 0.0,
        }
    offdiag = similarity[~np.eye(similarity.shape[0], dtype=bool)]
    return {
        "offdiag_min": float(np.min(offdiag)),
        "offdiag_p25": float(np.quantile(offdiag, 0.25)),
        "offdiag_mean": float(np.mean(offdiag)),
        "offdiag_median": float(np.median(offdiag)),
        "offdiag_p75": float(np.quantile(offdiag, 0.75)),
        "offdiag_max": float(np.max(offdiag)),
    }


def _condition_geometry_config(config: dict[str, Any]) -> dict[str, Any]:
    geometry = config.get("geometry", config.get("postprocess", {}))
    if geometry is None or geometry is False:
        return {"method": "none"}
    if isinstance(geometry, str):
        return {"method": geometry}
    if not isinstance(geometry, dict):
        raise ValueError("conditions.geometry must be a mapping, string, false, or null.")
    return dict(geometry)


def apply_condition_geometry(matrix: np.ndarray, config: dict[str, Any] | None) -> tuple[np.ndarray, dict[str, Any]]:
    geometry = dict(config or {})
    method = str(geometry.get("method", "none")).lower()
    append_common = bool(geometry.get("append_common_condition", False))
    raw = np.asarray(matrix, dtype=np.float32)
    raw_normalized = _normalize_rows(raw)
    raw_summary = cosine_similarity_summary(raw_normalized)

    if method in {"none", "raw"}:
        if append_common:
            raise ValueError(
                "conditions.geometry.append_common_condition requires centering to define "
                "the deducted common vector."
            )
        return raw.astype(np.float32), {
            "condition_geometry": {"method": "none"},
            "raw_condition_cosine": raw_summary,
            "transformed_condition_cosine": raw_summary,
        }

    if method not in {"center", "common_component_removal"}:
        raise ValueError(
            "conditions.geometry.method must be one of: none, center, common_component_removal."
        )

    centered = bool(geometry.get("center", True))
    normalize = bool(geometry.get("normalize", True))
    strength = float(geometry.get("strength", 1.0))
    if not 0.0 <= strength <= 1.0:
        raise ValueError("conditions.geometry.strength must be between 0.0 and 1.0.")

    transformed = np.asarray(raw, dtype=np.float64)
    common_vector: np.ndarray | None = None
    if centered:
        common_vector = transformed.mean(axis=0, keepdims=True)
        transformed = transformed - common_vector
    elif append_common:
        raise ValueError(
            "conditions.geometry.append_common_condition requires conditions.geometry.center=true."
        )

    remove_top_components = 0
    if method == "common_component_removal":
        remove_top_components = int(geometry.get("remove_top_components", 1))
        if remove_top_components < 0:
            raise ValueError("conditions.geometry.remove_top_components must be non-negative.")
        max_components = max(0, min(transformed.shape[0] - 1, transformed.shape[1]))
        remove_top_components = min(remove_top_components, max_components)
        if remove_top_components:
            _, _, vh = np.linalg.svd(transformed, full_matrices=False)
            components = vh[:remove_top_components]
            transformed = transformed - transformed @ components.T @ components

    if normalize:
        transformed = _normalize_rows(transformed, fallback=raw)
    else:
        transformed = transformed.astype(np.float32)

    if strength < 1.0:
        transformed = _normalize_rows(
            (1.0 - strength) * raw_normalized + strength * _normalize_rows(transformed),
            fallback=raw,
        )

    common_metadata: dict[str, Any] = {"appended": False}
    if append_common:
        assert common_vector is not None
        if float(np.linalg.norm(common_vector)) < 1e-12:
            raise ValueError("The deducted common condition vector is zero and cannot be appended.")
        # Preserve the exact vector deducted during centering. Gate cosine
        # computation normalizes conditions inside the model when needed.
        transformed = np.vstack((transformed, common_vector.astype(np.float32)))
        common_metadata = {
            "appended": True,
            "index": int(raw.shape[0]),
            "label": str(geometry.get("common_label", "Common Tactic Component")),
            "norm_before_normalization": float(np.linalg.norm(common_vector)),
        }

    transformed = transformed.astype(np.float32)
    return transformed, {
        "condition_geometry": {
            "method": method,
            "center": centered,
            "normalize": normalize,
            "remove_top_components": remove_top_components,
            "strength": strength,
            "common_condition": common_metadata,
        },
        "raw_condition_cosine": cosine_similarity_summary(
            np.vstack((raw, common_vector)) if append_common else raw_normalized
        ),
        "transformed_condition_cosine": cosine_similarity_summary(transformed),
    }


def _condition_output_labels(
    tactic_labels: list[str],
    geometry_metadata: dict[str, Any],
) -> list[str]:
    common = geometry_metadata.get("condition_geometry", {}).get("common_condition", {})
    if not common.get("appended", False):
        return list(tactic_labels)
    common_label = str(common["label"])
    if common_label in tactic_labels:
        raise ValueError(f"Common condition label duplicates a tactic label: {common_label}")
    return [*tactic_labels, common_label]


def _raw_output_matrix(
    raw_tactics: np.ndarray,
    geometry_metadata: dict[str, Any],
) -> np.ndarray:
    common = geometry_metadata.get("condition_geometry", {}).get("common_condition", {})
    if not common.get("appended", False):
        return np.asarray(raw_tactics, dtype=np.float32)
    centroid = np.asarray(raw_tactics, dtype=np.float32).mean(axis=0, keepdims=True)
    return np.vstack((raw_tactics, centroid)).astype(np.float32)


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
            raw_matrix = archive["matrix"].astype(np.float32)
            labels = archive["labels"].astype(str).tolist()
        matrix, geometry_metadata = apply_condition_geometry(raw_matrix, _condition_geometry_config(config))
        output_labels = _condition_output_labels(labels, geometry_metadata)
        return ConditionEmbeddings(
            output_labels,
            labels,
            matrix,
            {
                "cache_hit": True,
                "cache_path": str(cache_path),
                "text_fields": text_fields,
                **geometry_metadata,
                **metadata,
            },
            _raw_output_matrix(raw_matrix, geometry_metadata),
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
    raw_matrix = embedder.encode(texts).astype(np.float32)
    matrix, geometry_metadata = apply_condition_geometry(raw_matrix, _condition_geometry_config(config))
    output_labels = _condition_output_labels(requested, geometry_metadata)
    np.savez_compressed(cache_path, labels=np.asarray(requested, dtype=str), matrix=raw_matrix)
    meta_path.write_text(
        json.dumps(
            {
                "labels": output_labels,
                "tactic_labels": requested,
                "cache_path": str(cache_path),
                "embedding_dim": int(matrix.shape[1]),
                "text_fields": text_fields,
                "text_preview": dict(zip(requested, texts, strict=True)),
                "condition_file_metadata": metadata,
                **geometry_metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return ConditionEmbeddings(
        output_labels,
        requested,
        matrix,
        {
            "cache_hit": False,
            "cache_path": str(cache_path),
            "text_fields": text_fields,
            **geometry_metadata,
            **metadata,
        },
        _raw_output_matrix(raw_matrix, geometry_metadata),
    )
