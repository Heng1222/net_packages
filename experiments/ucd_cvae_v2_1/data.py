from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch

from .embedders import build_text_embedder, normalize_rows


@dataclass(slots=True)
class PreparedDataset:
    prepared_dir: Path
    x_path: Path
    metadata_path: Path
    manifest_path: Path
    reused: bool


@dataclass(slots=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def payload_to_text(value: Any, parser: str = "auto") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)): return ""
    if isinstance(value, bytes): return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return "\n[PACKET]\n".join(payload_to_text(item, "none") for item in value if item is not None)
    text = str(value)
    if parser in {"auto", "python_literal_list"} and text.lstrip().startswith(("[", "(")):
        try: parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            if parser == "python_literal_list": raise ValueError("Payload is not a valid Python list literal.")
            return text
        if isinstance(parsed, (list, tuple)): return payload_to_text(parsed, "none")
    return text


def expected_manifest(config: dict[str, Any], input_path: Path) -> dict[str, Any]:
    stat = input_path.stat()
    return {
        "input_path": str(input_path.resolve()), "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns, "sample_id_col": config["sample_id_col"],
        "payload_text_col": config["payload_text_col"], "time_col": config["time_col"],
        "metadata_cols": list(config.get("metadata_cols", [])),
        "payload_parser": config.get("payload_parser", "auto"), "max_rows": config.get("max_rows"),
        "embedder": config["embedder"], "filter": "nonempty_payload_only",
    }


def manifest_matches(path: Path, expected: dict[str, Any]) -> bool:
    try: return json.loads(path.read_text(encoding="utf-8")).get("fingerprint") == expected
    except (OSError, json.JSONDecodeError): return False


def _columns(config: dict[str, Any], path: Path) -> list[str]:
    available = set(pd.read_csv(path, nrows=0).columns)
    required = [config["sample_id_col"], config["payload_text_col"], config["time_col"]]
    missing = [column for column in required if column not in available]
    if missing: raise ValueError(f"Input CSV is missing required columns: {missing}")
    return list(dict.fromkeys(required + [c for c in config.get("metadata_cols", []) if c in available]))


def _iter_chunks(config: dict[str, Any], path: Path) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(path, usecols=_columns(config, path), dtype=str,
                           chunksize=int(config.get("read_chunksize", 5000)))


def _accepted_rows(chunk: pd.DataFrame, config: dict[str, Any], already_written: int) -> list[dict[str, str]]:
    limit = int(config["max_rows"]) if config.get("max_rows") is not None else None
    result: list[dict[str, str]] = []
    for source_index, row in chunk.iterrows():
        if limit is not None and already_written + len(result) >= limit: break
        text = payload_to_text(row[config["payload_text_col"]], str(config.get("payload_parser", "auto"))).strip()
        if not text: continue
        item = {
            "source_row_index": str(source_index), "sample_id": "" if pd.isna(row[config["sample_id_col"]]) else str(row[config["sample_id_col"]]),
            "datetime": "" if pd.isna(row[config["time_col"]]) else str(row[config["time_col"]]),
            "payload_hash": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
            "payload_text": text,
        }
        for column in config.get("metadata_cols", []):
            if column in row: item[column] = "" if pd.isna(row[column]) else str(row[column])
        result.append(item)
    return result


def prepare_dataset(config: dict[str, Any], project_root: Path, device: torch.device,
                    force: bool = False) -> PreparedDataset:
    data = config["data"]; input_path = Path(data["input_path"])
    if not input_path.is_absolute(): input_path = project_root / input_path
    if not input_path.is_file(): raise FileNotFoundError(f"Input CSV not found: {input_path}")
    prepared = Path(data["prepared_dir"])
    if not prepared.is_absolute(): prepared = project_root / prepared
    prepared.mkdir(parents=True, exist_ok=True)
    x_path, metadata_path, manifest_path = prepared / "x.npy", prepared / "metadata.csv", prepared / "manifest.json"
    fingerprint = expected_manifest(data, input_path)
    if not force and x_path.is_file() and metadata_path.is_file() and manifest_matches(manifest_path, fingerprint):
        return PreparedDataset(prepared, x_path, metadata_path, manifest_path, True)

    count = 0
    for chunk in _iter_chunks(data, input_path):
        count += len(_accepted_rows(chunk, data, count))
        if data.get("max_rows") is not None and count >= int(data["max_rows"]): break
    if count == 0: raise ValueError("No non-empty payloads found.")

    embedder = build_text_embedder(data["embedder"], device)
    temp_x, temp_meta, temp_manifest = prepared / "x.tmp.npy", prepared / "metadata.tmp.csv", prepared / "manifest.tmp.json"
    matrix = np.lib.format.open_memmap(temp_x, mode="w+", dtype=np.float32,
                                       shape=(count, int(embedder.output_dim)))
    cursor = 0; first = True
    try:
        for chunk in _iter_chunks(data, input_path):
            rows = _accepted_rows(chunk, data, cursor)
            if not rows: continue
            texts = [row.pop("payload_text") for row in rows]
            vectors = normalize_rows(embedder.encode(texts)); end = cursor + len(rows)
            if vectors.shape != (len(rows), int(embedder.output_dim)):
                raise ValueError(f"Embedder returned invalid shape {vectors.shape}.")
            matrix[cursor:end] = vectors
            pd.DataFrame(rows).to_csv(temp_meta, mode="w" if first else "a", header=first,
                                      index=False, encoding="utf-8")
            cursor = end; first = False
            if data.get("max_rows") is not None and cursor >= int(data["max_rows"]): break
        matrix.flush()
        if cursor != count: raise RuntimeError(f"Expected {count} rows but wrote {cursor}.")
        del matrix
        temp_manifest.write_text(json.dumps({"fingerprint": fingerprint, "rows": count,
            "embedding_dim": int(embedder.output_dim), "normalization": "l2"}, indent=2), encoding="utf-8")
        temp_x.replace(x_path); temp_meta.replace(metadata_path); temp_manifest.replace(manifest_path)
    except Exception:
        if "matrix" in locals(): del matrix
        for path in (temp_x, temp_meta, temp_manifest):
            if path.exists(): path.unlink()
        raise
    return PreparedDataset(prepared, x_path, metadata_path, manifest_path, False)


def load_prepared(path: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    root = Path(path); x_path, meta_path = root / "x.npy", root / "metadata.csv"
    if not x_path.is_file() or not meta_path.is_file(): raise FileNotFoundError(f"Incomplete prepared cache: {root}")
    x = np.load(x_path, mmap_mode="r"); metadata = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
    if len(x) != len(metadata): raise ValueError("Embedding and metadata row counts differ.")
    sample = np.asarray(x[:min(1000, len(x))], dtype=np.float32)
    if len(sample) and not np.allclose(np.linalg.norm(sample, axis=1), 1.0, atol=1e-4):
        raise ValueError("Payload embeddings must be L2-normalized.")
    return x, metadata


def make_time_split(metadata: pd.DataFrame, config: dict[str, Any]) -> SplitIndices:
    times = pd.to_datetime(metadata["datetime"], errors="raise"); order = np.argsort(times.to_numpy(), kind="stable")
    n_train = int(len(order) * float(config["train_ratio"])); n_val = int(len(order) * float(config["val_ratio"]))
    return SplitIndices(order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:])


def split_assignments(metadata: pd.DataFrame, split: SplitIndices) -> pd.DataFrame:
    labels = np.full(len(metadata), "", dtype=object)
    labels[split.train] = "train"; labels[split.val] = "val"; labels[split.test] = "test"
    return pd.DataFrame({"row_index": np.arange(len(metadata)), "sample_id": metadata["sample_id"], "split": labels})


def leakage_report(metadata: pd.DataFrame, split: SplitIndices) -> dict[str, Any]:
    hashes = {name: set(metadata.iloc[idx]["payload_hash"].astype(str)) for name, idx in
              (("train", split.train), ("val", split.val), ("test", split.test))}
    return {"rows": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
            "payload_hash_overlap": {"train_val": len(hashes["train"] & hashes["val"]),
            "train_test": len(hashes["train"] & hashes["test"]), "val_test": len(hashes["val"] & hashes["test"])}}
