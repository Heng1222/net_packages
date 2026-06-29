from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .dataset import DatasetBundle, LoadedData


class DataAdapter(Protocol):
    def load(self, config: dict[str, Any]) -> LoadedData: ...


def payload_to_text(value: Any, parser: str = "auto") -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple, np.ndarray)):
        return "\n[PACKET]\n".join(payload_to_text(item, "none") for item in value if item is not None)
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value)
    if parser in {"auto", "python_literal_list"} and text.lstrip().startswith(("[", "(")):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return payload_to_text(parsed, "none")
        except (ValueError, SyntaxError):
            if parser == "python_literal_list":
                logging.getLogger("ae_cvae_tactic").warning("Could not parse one payload list; using its raw string.")
    return text


def _parse_embedding(value: Any, row: int, column: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        result = value
    elif isinstance(value, (list, tuple)):
        result = np.asarray(value)
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Cannot parse embedding at row {row}, column '{column}'.") from exc
        result = np.asarray(parsed)
    else:
        raise ValueError(f"Unsupported embedding value at row {row}, column '{column}': {type(value).__name__}")
    if result.ndim != 1:
        raise ValueError(f"Embedding at row {row}, column '{column}' is not one-dimensional.")
    return result.astype(np.float32)


def _read_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported metadata format: {suffix}")


class TabularAdapter:
    def __init__(self, fmt: str) -> None:
        self.fmt = fmt

    def load(self, config: dict[str, Any]) -> LoadedData:
        path = Path(config["input_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Input data file not found: {path}")
        if self.fmt == "csv":
            frame = pd.read_csv(path)
        elif self.fmt == "jsonl":
            frame = pd.read_json(path, lines=True)
        else:  # pragma: no cover - guarded by factory
            raise ValueError(self.fmt)

        def optional_column(name: str) -> np.ndarray | None:
            column = config.get(name)
            if not column:
                return None
            if column not in frame.columns:
                raise KeyError(f"Configured {name} column not found: '{column}'")
            values = frame[column]
            if values.isna().any():
                count = int(values.isna().sum())
                raise ValueError(f"Column '{column}' contains {count} missing values.")
            return values.astype(str).to_numpy(dtype=str)

        sample_col = config.get("sample_id_col")
        if sample_col:
            if sample_col not in frame.columns:
                raise KeyError(f"Configured sample_id_col not found: '{sample_col}'")
            sample_ids = frame[sample_col].astype(str).to_numpy(dtype=str)
        else:
            sample_ids = np.arange(len(frame)).astype(str)

        labels = optional_column("label_col")
        condition_keys = optional_column("condition_key_col")
        metadata_cols = config.get("metadata_cols", []) or []
        missing_metadata = sorted(set(metadata_cols).difference(frame.columns))
        if missing_metadata:
            raise KeyError(f"Configured metadata columns not found: {missing_metadata}")
        metadata = frame.loc[:, metadata_cols].copy() if metadata_cols else pd.DataFrame(index=frame.index)

        embedding_col = config.get("embedding_col")
        embedding_prefix = config.get("embedding_prefix")
        text_col = config.get("payload_text_col")
        selected = sum(bool(value) for value in (embedding_col, embedding_prefix, text_col))
        if selected != 1:
            raise ValueError("Configure exactly one of data.embedding_col, data.embedding_prefix, or data.payload_text_col.")

        features: np.ndarray | None = None
        texts: list[str] | None = None
        if embedding_col:
            if embedding_col not in frame.columns:
                raise KeyError(f"Configured embedding_col not found: '{embedding_col}'")
            vectors = [_parse_embedding(value, row, embedding_col) for row, value in enumerate(frame[embedding_col])]
            dimensions = {len(vector) for vector in vectors}
            if len(dimensions) != 1:
                raise ValueError(f"Embedding column '{embedding_col}' has inconsistent dimensions: {sorted(dimensions)}")
            features = np.vstack(vectors)
        elif embedding_prefix:
            columns = [column for column in frame.columns if str(column).startswith(str(embedding_prefix))]
            if not columns:
                raise KeyError(f"No columns start with embedding_prefix '{embedding_prefix}'.")
            features = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        else:
            if text_col not in frame.columns:
                raise KeyError(f"Configured payload_text_col not found: '{text_col}'")
            parser = str(config.get("payload_parser", "auto"))
            texts = [payload_to_text(value, parser) for value in frame[text_col]]

        return LoadedData(sample_ids, features, texts, labels, condition_keys, metadata)


class ArrayAdapter:
    def load(self, config: dict[str, Any]) -> LoadedData:
        path = Path(config["input_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Input data file not found: {path}")
        if path.suffix.lower() == ".npy":
            features = np.load(path, allow_pickle=False)
        else:
            archive = np.load(path, allow_pickle=False)
            key = str(config.get("array_key", "x"))
            if key not in archive:
                raise KeyError(f"NPZ array key '{key}' not found. Available: {archive.files}")
            features = archive[key]
        if np.asarray(features).ndim != 2:
            raise ValueError(f"NPY/NPZ features must be 2D; got {np.asarray(features).shape}")

        metadata_path = config.get("metadata_path")
        if metadata_path:
            frame = _read_metadata(Path(metadata_path))
            if len(frame) != len(features):
                raise ValueError("Metadata and feature row counts differ.")
        else:
            frame = pd.DataFrame(index=np.arange(len(features)))

        def values(column_key: str, fallback: np.ndarray | None = None) -> np.ndarray | None:
            column = config.get(column_key)
            if not column:
                return fallback
            if column not in frame.columns:
                raise KeyError(f"Configured {column_key} not found in metadata: '{column}'")
            return frame[column].astype(str).to_numpy(dtype=str)

        sample_ids = values("sample_id_col", np.arange(len(features)).astype(str))
        labels = values("label_col")
        condition_keys = values("condition_key_col")
        metadata_cols = config.get("metadata_cols", []) or []
        metadata = frame.loc[:, metadata_cols].copy() if metadata_cols else pd.DataFrame(index=frame.index)
        return LoadedData(sample_ids, np.asarray(features, dtype=np.float32), None, labels, condition_keys, metadata)


def infer_format(path: str | Path, configured: str | None) -> str:
    if configured:
        return configured.lower()
    suffix = Path(path).suffix.lower()
    mapping = {".csv": "csv", ".jsonl": "jsonl", ".ndjson": "jsonl", ".npy": "npy", ".npz": "npz"}
    if suffix not in mapping:
        raise ValueError(f"Cannot infer input format from suffix '{suffix}'. Set data.input_format.")
    return mapping[suffix]


def load_raw_data(config: dict[str, Any]) -> LoadedData:
    fmt = infer_format(config["input_path"], config.get("input_format"))
    adapter: DataAdapter = TabularAdapter(fmt) if fmt in {"csv", "jsonl"} else ArrayAdapter()
    return adapter.load(config)


def load_dataset(config: dict[str, Any], features: np.ndarray | None = None) -> DatasetBundle:
    loaded = load_raw_data(config)
    x = loaded.features if features is None else features
    if x is None:
        raise ValueError("The adapter produced text but no embeddings were supplied.")
    return DatasetBundle(x, loaded.sample_ids, loaded.labels, loaded.condition_keys, loaded.metadata)
