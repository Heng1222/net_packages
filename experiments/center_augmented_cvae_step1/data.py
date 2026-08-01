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
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return "\n[PACKET]\n".join(payload_to_text(item, "none") for item in value)
    text = str(value)
    if parser in {"auto", "python_literal_list"} and text.lstrip().startswith(("[", "(")):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            if parser == "python_literal_list":
                raise ValueError("payload value is not a valid Python list literal.")
            return text
        if isinstance(parsed, (list, tuple)):
            return payload_to_text(parsed, "none")
    return text


def _fingerprint(config: dict[str, Any], input_path: Path) -> dict[str, Any]:
    stat = input_path.stat()
    return {
        "input_path": str(input_path.resolve()), "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns, "sample_id_col": config["sample_id_col"],
        "payload_text_col": config["payload_text_col"], "label_col": config.get("label_col"),
        "time_col": config["time_col"], "payload_parser": config.get("payload_parser", "auto"),
        "max_rows": config.get("max_rows"), "embedder": config["embedder"],
    }


def _iter_chunks(config: dict[str, Any], path: Path) -> Iterator[pd.DataFrame]:
    cols = [config["sample_id_col"], config["payload_text_col"], config["time_col"]]
    if config.get("label_col"):
        cols.append(config["label_col"])
    yield from pd.read_csv(path, usecols=list(dict.fromkeys(cols)), dtype=str,
                           chunksize=int(config.get("read_chunksize", 5000)))


def _rows(chunk: pd.DataFrame, config: dict[str, Any], written: int) -> list[dict[str, str]]:
    limit = config.get("max_rows")
    limit = int(limit) if limit is not None else None
    result: list[dict[str, str]] = []
    for source_index, row in chunk.iterrows():
        if limit is not None and written + len(result) >= limit:
            break
        text = payload_to_text(row[config["payload_text_col"]], str(config.get("payload_parser", "auto")))
        item = {
            "source_row_index": str(source_index),
            "sample_id": str(row[config["sample_id_col"]]),
            "datetime": str(row[config["time_col"]]),
            "payload_hash": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
            "payload_text": text,
        }
        if config.get("label_col"):
            item["source_label"] = "" if pd.isna(row[config["label_col"]]) else str(row[config["label_col"]])
        result.append(item)
    return result


def prepare_dataset(config: dict[str, Any], project_root: Path, device: torch.device,
                    force: bool = False) -> PreparedDataset:
    data = config["data"]
    input_path = Path(data["input_path"])
    if not input_path.is_absolute(): input_path = project_root / input_path
    if not input_path.is_file(): raise FileNotFoundError(f"Input CSV not found: {input_path}")
    prepared_dir = Path(data["prepared_dir"])
    if not prepared_dir.is_absolute(): prepared_dir = project_root / prepared_dir
    prepared_dir.mkdir(parents=True, exist_ok=True)
    x_path, metadata_path, manifest_path = (
        prepared_dir / "x.npy", prepared_dir / "metadata.csv", prepared_dir / "manifest.json"
    )
    fingerprint = _fingerprint(data, input_path)
    if not force and all(path.is_file() for path in (x_path, metadata_path, manifest_path)):
        try:
            if json.loads(manifest_path.read_text(encoding="utf-8")).get("fingerprint") == fingerprint:
                return PreparedDataset(prepared_dir, x_path, metadata_path, manifest_path, True)
        except json.JSONDecodeError:
            pass

    count = 0
    for chunk in _iter_chunks(data, input_path):
        count += len(_rows(chunk, data, count))
        if data.get("max_rows") is not None and count >= int(data["max_rows"]): break
    if count == 0: raise ValueError("No rows found in Step1 input.")

    embedder = build_text_embedder(data["embedder"], device)
    temp_x, temp_meta = prepared_dir / "x.tmp.npy", prepared_dir / "metadata.tmp.csv"
    matrix = np.lib.format.open_memmap(temp_x, mode="w+", dtype=np.float32,
                                       shape=(count, int(embedder.output_dim)))
    cursor, first = 0, True
    for chunk in _iter_chunks(data, input_path):
        rows = _rows(chunk, data, cursor)
        if not rows: continue
        texts = [row.pop("payload_text") for row in rows]
        embedded = normalize_rows(embedder.encode(texts))
        matrix[cursor:cursor + len(rows)] = embedded
        pd.DataFrame(rows).to_csv(temp_meta, mode="w" if first else "a", header=first,
                                  index=False, encoding="utf-8")
        first = False; cursor += len(rows)
        if data.get("max_rows") is not None and cursor >= int(data["max_rows"]): break
    matrix.flush(); del matrix
    if cursor != count: raise RuntimeError(f"Expected {count} rows but wrote {cursor}.")
    for target in (x_path, metadata_path):
        if target.exists(): target.unlink()
    temp_x.replace(x_path); temp_meta.replace(metadata_path)
    manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "rows": count,
                                         "embedding_dim": int(embedder.output_dim),
                                         "normalization": "l2"}, indent=2), encoding="utf-8")
    return PreparedDataset(prepared_dir, x_path, metadata_path, manifest_path, False)


def load_prepared(path: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    root = Path(path); x_path, meta_path = root / "x.npy", root / "metadata.csv"
    if not x_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Prepared dataset is incomplete: {root}")
    x = np.load(x_path, mmap_mode="r"); metadata = pd.read_csv(meta_path, dtype=str)
    if len(x) != len(metadata): raise ValueError("x.npy and metadata.csv row counts differ.")
    norms = np.linalg.norm(np.asarray(x[:min(len(x), 1000)]), axis=1)
    if len(norms) and not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Prepared payload embeddings are not L2-normalized.")
    return x, metadata


def make_time_split(metadata: pd.DataFrame, config: dict[str, Any]) -> SplitIndices:
    times = pd.to_datetime(metadata["datetime"], errors="raise")
    order = np.argsort(times.to_numpy(), kind="stable")
    n_train = int(len(order) * float(config["train_ratio"]))
    n_val = int(len(order) * float(config["val_ratio"]))
    return SplitIndices(order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:])


def split_assignments(metadata: pd.DataFrame, split: SplitIndices) -> pd.DataFrame:
    values = np.full(len(metadata), "", dtype=object)
    values[split.train] = "train"; values[split.val] = "val"; values[split.test] = "test"
    return pd.DataFrame({"row_index": np.arange(len(metadata)), "sample_id": metadata["sample_id"], "split": values})


def split_label_counts(metadata: pd.DataFrame, split: SplitIndices) -> pd.DataFrame:
    if "source_label" not in metadata: return pd.DataFrame(columns=["split", "source_label", "rows"])
    parts = []
    for name, indices in (("train", split.train), ("val", split.val), ("test", split.test)):
        counts = metadata.iloc[indices]["source_label"].value_counts(dropna=False)
        parts.extend({"split": name, "source_label": str(label), "rows": int(count)} for label, count in counts.items())
    return pd.DataFrame(parts)


def leakage_report(metadata: pd.DataFrame, split: SplitIndices) -> dict[str, Any]:
    hashes = {name: set(metadata.iloc[idx]["payload_hash"].astype(str)) for name, idx in
              (("train", split.train), ("val", split.val), ("test", split.test))}
    return {
        "rows": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "payload_hash_overlap": {
            "train_val": len(hashes["train"] & hashes["val"]),
            "train_test": len(hashes["train"] & hashes["test"]),
            "val_test": len(hashes["val"] & hashes["test"]),
        },
    }
