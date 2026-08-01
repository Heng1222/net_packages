from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .embedders import build_text_embedder, normalize_rows


@dataclass(slots=True)
class ConditionSet:
    labels: list[str]
    tactic_labels: list[str]
    raw_tactics: np.ndarray
    centroid: np.ndarray
    centered_tactics: np.ndarray
    decode_matrix: np.ndarray
    gate_matrix: np.ndarray
    metadata: dict[str, Any]

    @property
    def dimension(self) -> int:
        return int(self.decode_matrix.shape[1])


def centroid_decomposition(raw_tactics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw_tactics, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] < 2:
        raise ValueError("raw_tactics must be a 2D matrix with at least two rows.")
    centroid = raw.mean(axis=0).astype(np.float32)
    if float(np.linalg.norm(centroid)) < 1e-12:
        raise ValueError("Condition centroid is zero and cannot define the common condition.")
    centered = (raw - centroid[None, :]).astype(np.float32)
    if np.any(np.linalg.norm(centered, axis=1) < 1e-12):
        raise ValueError("At least one tactic equals the centroid and cannot define a gate direction.")
    decode = np.vstack((centroid[None, :], centered)).astype(np.float32)
    gate = normalize_rows(decode)
    return centroid, centered, decode, gate


def cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    normalized = normalize_rows(matrix)
    return (normalized @ normalized.T).astype(np.float32)


def _flatten(value: Any) -> list[str]:
    if value is None: return []
    if isinstance(value, str): return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value: result.extend(_flatten(item))
        return result
    return [str(value)]


def _records(path: Path, fields: list[str]) -> tuple[list[str], list[str], dict[str, Any]]:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict) or not isinstance(body.get("tactics"), dict):
        raise ValueError("Condition YAML must contain a tactics mapping.")
    labels, texts = [], []
    for key, value in body["tactics"].items():
        record = dict(value)
        label = str(record.get("label", key))
        terms: list[str] = []
        for field in fields: terms.extend(_flatten(record.get(field)))
        if not terms: raise ValueError(f"Condition {label} has no text in fields {fields}.")
        labels.append(label); texts.append(", ".join(terms))
    if len(labels) != 13:
        raise ValueError(f"Exactly 13 tactic conditions are required; got {len(labels)}.")
    return labels, texts, dict(body.get("metadata", {}))


def load_condition_set(config: dict[str, Any], device: torch.device) -> ConditionSet:
    path = Path(config["path"])
    if not path.is_file(): raise FileNotFoundError(f"Condition YAML not found: {path}")
    fields = list(map(str, config.get("text_fields", ["keywords", "techniques"])))
    labels, texts, source_metadata = _records(path, fields)
    cache_dir = Path(config["cache_dir"]); cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(path.read_bytes())
    digest.update(json.dumps(config["embedder"], sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(fields).encode("utf-8"))
    key = digest.hexdigest()[:24]
    cache_path = cache_dir / f"raw_conditions_{key}.npz"
    cache_hit = cache_path.is_file()
    if cache_hit:
        with np.load(cache_path, allow_pickle=False) as archive:
            raw = archive["raw"].astype(np.float32)
            cached_labels = archive["labels"].astype(str).tolist()
        if cached_labels != labels: raise ValueError("Cached condition label order differs from YAML.")
    else:
        embedder = build_text_embedder(config["embedder"], device)
        raw = normalize_rows(embedder.encode(texts))
        np.savez_compressed(cache_path, raw=raw, labels=np.asarray(labels, dtype=str))
    centroid, centered, decode, gate = centroid_decomposition(raw)
    common = str(config.get("common_label", "Common Tactic Component"))
    metadata = {
        "method": "centroid_decomposition", "common_label": common,
        "equation": "raw_tactic_i = centroid + centered_tactic_i",
        "centering_error": float(np.max(np.abs(centered.mean(axis=0)))),
        "recomposition_error": float(np.max(np.abs(raw - (centroid[None, :] + centered)))),
        "cache_hit": cache_hit, "cache_path": str(cache_path), "text_fields": fields,
        "source": source_metadata,
    }
    return ConditionSet([common, *labels], labels, raw, centroid, centered, decode, gate, metadata)


def save_condition_set(conditions: ConditionSet, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "condition_embeddings.npz", labels=np.asarray(conditions.labels, dtype=str),
        tactic_labels=np.asarray(conditions.tactic_labels, dtype=str), raw_tactics=conditions.raw_tactics,
        centroid=conditions.centroid, centered_tactics=conditions.centered_tactics,
        decode_matrix=conditions.decode_matrix, gate_matrix=conditions.gate_matrix,
    )
    (directory / "condition_geometry.json").write_text(
        json.dumps(conditions.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
