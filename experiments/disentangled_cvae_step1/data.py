from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .embedders import build_text_embedder


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
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return "\n[PACKET]\n".join(payload_to_text(item, "none") for item in value if item is not None)
    text = str(value)
    if parser in {"auto", "python_literal_list"} and text.lstrip().startswith(("[", "(")):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            if parser == "python_literal_list":
                raise ValueError("Configured payload_parser='python_literal_list' but value is not parseable.")
            return text
        if isinstance(parsed, (list, tuple)):
            return payload_to_text(parsed, "none")
    return text


def parse_step1_label(value: Any, multi_policy: str = "error") -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text in {"set()", "{}", "[]", "None", "nan"}:
            return None
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            if text.startswith(("{", "[", "(")):
                raise ValueError(f"Invalid Step1 tactic label literal: {value!r}")
            return text
    else:
        parsed = value

    if isinstance(parsed, str):
        label = parsed.strip()
        return label or None
    if isinstance(parsed, (set, list, tuple)):
        labels = sorted(str(item).strip() for item in parsed if str(item).strip())
        if not labels:
            return None
        if len(labels) == 1:
            return labels[0]
        if multi_policy == "first":
            return labels[0]
        if multi_policy == "join":
            return "|".join(labels)
        raise ValueError(f"Expected one Step1 tactic label, got {labels}")
    raise ValueError(f"Unsupported Step1 tactic label type: {type(parsed).__name__}")


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def expected_manifest(data_config: dict[str, Any], input_path: Path) -> dict[str, Any]:
    stat = input_path.stat()
    keys = {
        "input_path": str(input_path.resolve()),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "sample_id_col": data_config.get("sample_id_col"),
        "payload_text_col": data_config.get("payload_text_col"),
        "payload_parser": data_config.get("payload_parser", "auto"),
        "label_col": data_config.get("label_col"),
        "time_col": data_config.get("time_col"),
        "metadata_cols": data_config.get("metadata_cols", []),
        "max_rows": data_config.get("max_rows"),
        "label_multi_policy": data_config.get("label_multi_policy", "error"),
        "skip_empty_labels": data_config.get("skip_empty_labels", True),
        "embedder": data_config.get("embedder", {}),
    }
    return _jsonable(keys)


def manifest_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return actual.get("fingerprint") == expected


def _iter_relevant_chunks(data_config: dict[str, Any], input_path: Path):
    required_cols = [
        data_config["sample_id_col"],
        data_config["payload_text_col"],
        data_config["label_col"],
        data_config["time_col"],
    ]
    metadata_cols = data_config.get("metadata_cols", []) or []
    usecols = list(dict.fromkeys(required_cols + metadata_cols))
    chunksize = int(data_config.get("read_chunksize", 5000))
    yield from pd.read_csv(input_path, usecols=usecols, dtype=str, chunksize=chunksize)


def _accepted_rows_from_chunk(
    chunk: pd.DataFrame,
    data_config: dict[str, Any],
    accepted_so_far: int,
) -> list[dict[str, str]]:
    sample_id_col = data_config["sample_id_col"]
    payload_col = data_config["payload_text_col"]
    label_col = data_config["label_col"]
    time_col = data_config["time_col"]
    metadata_cols = data_config.get("metadata_cols", []) or []
    parser = str(data_config.get("payload_parser", "auto"))
    multi_policy = str(data_config.get("label_multi_policy", "error"))
    skip_empty = bool(data_config.get("skip_empty_labels", True))
    max_rows = data_config.get("max_rows")
    max_rows = int(max_rows) if max_rows is not None else None

    rows: list[dict[str, str]] = []
    for source_row_index, row in chunk.iterrows():
        if max_rows is not None and accepted_so_far + len(rows) >= max_rows:
            break
        label = parse_step1_label(row[label_col], multi_policy)
        if label is None and skip_empty:
            continue
        payload_text = payload_to_text(row[payload_col], parser)
        item = {
            "source_row_index": str(source_row_index),
            "sample_id": str(row[sample_id_col]),
            "label": label or "",
            "datetime": str(row[time_col]),
            "payload_text": payload_text,
            "payload_hash": stable_text_hash(payload_text),
        }
        for col in metadata_cols:
            item[col] = "" if pd.isna(row.get(col, "")) else str(row.get(col, ""))
        rows.append(item)
    return rows


def prepare_dataset(
    config: dict[str, Any],
    project_root: Path,
    device: torch.device,
    force: bool = False,
) -> PreparedDataset:
    data_config = config["data"]
    input_path = Path(data_config["input_path"])
    if not input_path.is_absolute():
        input_path = project_root / input_path
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    prepared_dir = Path(data_config["prepared_dir"])
    if not prepared_dir.is_absolute():
        prepared_dir = project_root / prepared_dir
    prepared_dir.mkdir(parents=True, exist_ok=True)
    x_path = prepared_dir / "x.npy"
    metadata_path = prepared_dir / "metadata.csv"
    manifest_path = prepared_dir / "manifest.json"
    fingerprint = expected_manifest(data_config, input_path)
    if (
        not force
        and x_path.is_file()
        and metadata_path.is_file()
        and manifest_matches(manifest_path, fingerprint)
    ):
        return PreparedDataset(prepared_dir, x_path, metadata_path, manifest_path, True)

    accepted_count = 0
    for chunk in _iter_relevant_chunks(data_config, input_path):
        rows = _accepted_rows_from_chunk(chunk, data_config, accepted_count)
        accepted_count += len(rows)
        max_rows = data_config.get("max_rows")
        if max_rows is not None and accepted_count >= int(max_rows):
            break
    if accepted_count == 0:
        raise ValueError("No rows accepted from Step1 CSV.")

    embedder = build_text_embedder(data_config["embedder"], device)
    temp_x_path = prepared_dir / "x.tmp.npy"
    temp_metadata_path = prepared_dir / "metadata.tmp.csv"
    x_memmap = np.lib.format.open_memmap(
        temp_x_path, mode="w+", dtype=np.float32, shape=(accepted_count, int(embedder.output_dim))
    )

    cursor = 0
    first_metadata = True
    for chunk in _iter_relevant_chunks(data_config, input_path):
        rows = _accepted_rows_from_chunk(chunk, data_config, cursor)
        if not rows:
            continue
        texts = [row.pop("payload_text") for row in rows]
        embeddings = embedder.encode(texts)
        end = cursor + len(rows)
        x_memmap[cursor:end] = embeddings
        metadata_frame = pd.DataFrame(rows)
        metadata_frame.to_csv(
            temp_metadata_path,
            mode="w" if first_metadata else "a",
            header=first_metadata,
            index=False,
            encoding="utf-8",
        )
        first_metadata = False
        cursor = end
        max_rows = data_config.get("max_rows")
        if max_rows is not None and cursor >= int(max_rows):
            break
    x_memmap.flush()
    del x_memmap
    if cursor != accepted_count:
        raise RuntimeError(f"Prepared row count mismatch: counted {accepted_count}, wrote {cursor}.")

    if x_path.exists():
        x_path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()
    temp_x_path.replace(x_path)
    temp_metadata_path.replace(metadata_path)
    manifest = {
        "fingerprint": fingerprint,
        "rows": accepted_count,
        "embedding_dim": int(embedder.output_dim),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return PreparedDataset(prepared_dir, x_path, metadata_path, manifest_path, False)


def load_prepared(prepared_dir: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    root = Path(prepared_dir)
    x_path = root / "x.npy"
    metadata_path = root / "metadata.csv"
    if not x_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Prepared dataset is incomplete under: {root}")
    x = np.load(x_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path, dtype=str)
    if len(x) != len(metadata):
        raise ValueError("Prepared x.npy and metadata.csv row counts differ.")
    return x, metadata


def make_time_split(metadata: pd.DataFrame, split_config: dict[str, Any]) -> SplitIndices:
    ratios = np.asarray(
        [split_config["train_ratio"], split_config["val_ratio"], split_config["test_ratio"]],
        dtype=float,
    )
    if np.any(ratios < 0) or abs(float(ratios.sum()) - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must be non-negative and sum to 1.0: {ratios.tolist()}")
    parsed = pd.to_datetime(metadata["datetime"], errors="raise")
    ordered = np.argsort(parsed.to_numpy(), kind="stable")
    n = len(ordered)
    train_count = int(np.floor(n * ratios[0]))
    val_count = int(np.floor(n * ratios[1]))
    test_count = n - train_count - val_count
    if min(train_count, test_count) <= 0:
        raise ValueError("Time split produced an empty train or test split.")
    train = ordered[:train_count]
    val = ordered[train_count : train_count + val_count]
    test = ordered[train_count + val_count : train_count + val_count + test_count]
    return SplitIndices(train.astype(np.int64), val.astype(np.int64), test.astype(np.int64))


def leakage_report(metadata: pd.DataFrame, split: SplitIndices) -> dict[str, Any]:
    split_by_index = np.full(len(metadata), "", dtype=object)
    split_by_index[split.train] = "train"
    split_by_index[split.val] = "val"
    split_by_index[split.test] = "test"
    frame = metadata.loc[:, ["payload_hash"]].copy()
    frame["split"] = split_by_index
    grouped = frame.groupby("payload_hash")["split"].nunique()
    crossing_hashes = grouped[grouped > 1].index
    crossing = frame[frame["payload_hash"].isin(crossing_hashes)]
    return {
        "duplicate_payload_hashes": int((grouped > 1).sum()),
        "duplicate_payload_rows_crossing_splits": int(len(crossing)),
        "crossing_hash_preview": crossing["payload_hash"].drop_duplicates().head(20).tolist(),
    }


def standardize_to_memmap(
    x: np.ndarray,
    split: SplitIndices,
    output_path: Path,
    scaler_path: Path,
    batch_size: int = 20000,
) -> np.ndarray:
    scaler = StandardScaler()
    train_mask = np.zeros(len(x), dtype=bool)
    train_mask[split.train] = True
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        mask = train_mask[start:end]
        if mask.any():
            scaler.partial_fit(np.asarray(x[start:end][mask], dtype=np.float32))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scaled = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=x.shape)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        scaled[start:end] = scaler.transform(np.asarray(x[start:end], dtype=np.float32)).astype(np.float32)
    scaled.flush()
    joblib.dump(scaler, scaler_path)
    return np.load(output_path, mmap_mode="r")
