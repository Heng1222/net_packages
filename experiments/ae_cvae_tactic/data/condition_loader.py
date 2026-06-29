from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .embedders import SentenceTransformerEmbedder, TextEmbedder


TACTIC_ID_RE = re.compile(r"TA\d{4}", re.IGNORECASE)


@dataclass(slots=True)
class ConditionSet:
    labels: list[str]
    matrix: np.ndarray
    mode: str
    mapping: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float32)
        if self.matrix.ndim != 2 or len(self.labels) != len(self.matrix):
            raise ValueError("Condition labels and matrix shape do not match.")

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[1])

    def for_keys(self, keys: np.ndarray) -> np.ndarray:
        lookup = {label: index for index, label in enumerate(self.labels)}
        missing = sorted({str(key) for key in keys if str(key) not in lookup})
        if missing:
            raise KeyError(f"Condition vectors are missing labels: {missing}")
        return np.vstack([self.matrix[lookup[str(key)]] for key in keys]).astype(np.float32)


def _read_condition_file(path: Path, fmt: str | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Condition file not found: {path}")
    actual = (fmt or path.suffix.lstrip(".")).lower()
    if actual in {"yaml", "yml"}:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    elif actual == "json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif actual == "csv":
        return {}, pd.read_csv(path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported condition format: {actual}")
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    body = data.get("tactics", data) if isinstance(data, dict) else data
    if isinstance(body, dict):
        records = []
        for key, value in body.items():
            record = dict(value) if isinstance(value, dict) else {"description_full": value}
            record.setdefault("_key", str(key))
            records.append(record)
        return metadata, records
    if isinstance(body, list):
        return metadata, [dict(item) for item in body]
    raise ValueError("Condition file must contain a mapping or list of records.")


def _tactic_id(value: str) -> str | None:
    match = TACTIC_ID_RE.search(value)
    return match.group(0).upper() if match else None


def _match_records(
    labels: list[str], records: list[dict[str, Any]], label_field: str, id_field: str
) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        names = [str(record.get(field, "")) for field in (label_field, "_key", "name")]
        for name in names:
            if name:
                by_label[name.casefold()] = record
        record_id = str(record.get(id_field, "")) or next((_tactic_id(name) or "" for name in names), "")
        if record_id:
            by_id[record_id.upper()] = record
    matched: list[dict[str, Any]] = []
    missing: list[str] = []
    for label in labels:
        record = by_label.get(label.casefold())
        if record is None and _tactic_id(label):
            record = by_id.get(_tactic_id(label) or "")
        if record is None:
            missing.append(label)
        else:
            matched.append(record)
    if missing:
        raise KeyError(f"Condition file is missing tactic labels: {missing}")
    return matched


def _record_value(record: dict[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _derangement(size: int, seed: int) -> np.ndarray:
    if size < 2:
        raise ValueError("wrong condition mode requires at least two tactics.")
    rng = np.random.default_rng(seed)
    original = np.arange(size)
    for _ in range(1000):
        candidate = rng.permutation(size)
        if np.all(candidate != original):
            return candidate
    return np.roll(original, 1)


def load_condition_set(
    config: dict[str, Any],
    labels: list[str],
    mode: str,
    seed: int,
    embedder: TextEmbedder | None = None,
) -> ConditionSet:
    labels = sorted(str(label) for label in labels)
    random_dim = int(config.get("random_dim", 768))
    if mode == "random":
        rng = np.random.default_rng(seed)
        matrix = rng.normal(size=(len(labels), random_dim)).astype(np.float32)
        matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
        return ConditionSet(labels, matrix, mode, {label: "seeded_random" for label in labels})
    if mode == "none":
        return ConditionSet(labels, np.zeros((len(labels), random_dim), dtype=np.float32), mode)

    path = config.get("path")
    if not path:
        raise ValueError(f"conditions.path is required for condition mode '{mode}'.")
    metadata, records = _read_condition_file(Path(path), config.get("format"))
    matched = _match_records(labels, records, config.get("label_field", "label"), config.get("id_field", "tactic_id"))

    base_mode = config.get("wrong_base_mode", "full") if mode == "wrong" else mode
    field_by_mode = {
        "full": config.get("text_field_full", "description_full"),
        "short": config.get("text_field_short", "description_short"),
        "keywords": config.get("text_field_keywords", "keywords"),
    }
    if base_mode not in field_by_mode:
        raise ValueError(f"Unknown condition mode: {mode}")

    embedding_field = config.get("embedding_field")
    if embedding_field:
        vectors = [_record_value(record, embedding_field) for record in matched]
        if any(vector is None for vector in vectors):
            missing = [labels[index] for index, vector in enumerate(vectors) if vector is None]
            raise KeyError(f"Precomputed condition embedding field '{embedding_field}' is missing for: {missing}")
        matrix = np.asarray(vectors, dtype=np.float32)
    else:
        texts: list[str] = []
        field = str(field_by_mode[base_mode])
        for label, record in zip(labels, matched, strict=True):
            value = _record_value(record, field)
            if value is None:
                raise KeyError(f"Condition text field '{field}' is missing for '{label}'.")
            texts.append(", ".join(map(str, value)) if isinstance(value, list) else str(value))
        if embedder is None:
            raise RuntimeError(
                "No text embedding backend available. Please provide precomputed condition embeddings or install sentence-transformers."
            )
        matrix = embedder.encode(texts)

    mapping = {label: label for label in labels}
    if mode == "wrong":
        permutation = _derangement(len(labels), seed)
        matrix = matrix[permutation]
        mapping = {labels[index]: labels[int(permutation[index])] for index in range(len(labels))}
    return ConditionSet(labels, matrix, mode, mapping, metadata)


def make_condition_embedder(config: dict[str, Any], device: Any) -> TextEmbedder | None:
    backend = config.get("embedder_backend", "sentence_transformers")
    if backend in {None, "none"}:
        return None
    if backend != "sentence_transformers":
        raise ValueError(f"Unsupported condition embedder backend: {backend}")
    return SentenceTransformerEmbedder(
        config["embedder_model_name"],
        config.get("embedder_model_revision"),
        device,
        config.get("normalize", True),
        config.get("batch_size", 32),
    )
