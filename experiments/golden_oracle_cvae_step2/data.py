from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from experiments.disentangled_cvae_step1.embedders import build_text_embedder


@dataclass(slots=True)
class PreparedGoldenDataset:
    prepared_dir: Path
    reused: bool
    summary: dict[str, Any]


@dataclass(slots=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _fingerprint(config: dict[str, Any], path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "input_path": str(path.resolve()),
        "input_size": int(stat.st_size),
        "input_mtime_ns": int(stat.st_mtime_ns),
        "sample_id_col": config["sample_id_col"],
        "payload_text_col": config["payload_text_col"],
        "label_col": config["label_col"],
        "normal_label": config["normal_label"],
        "min_class_count": int(config.get("min_class_count", 1)),
        "conflicting_payload_policy": str(config.get("conflicting_payload_policy", "exclude")),
        "deduplicate_payloads": bool(config.get("deduplicate_payloads", True)),
        "embedder": config["embedder"],
    }


def prepare_golden_dataset(
    data_config: dict[str, Any],
    project_root: Path,
    device: torch.device,
    force: bool = False,
) -> PreparedGoldenDataset:
    source = Path(data_config["input_path"])
    if not source.is_absolute():
        source = project_root / source
    source = source.resolve()
    prepared = Path(data_config["prepared_dir"])
    if not prepared.is_absolute():
        prepared = project_root / prepared
    prepared = prepared.resolve()
    x_path = prepared / "x.npy"
    metadata_path = prepared / "metadata.csv"
    manifest_path = prepared / "manifest.json"
    fingerprint = _fingerprint(data_config, source)
    if not force and x_path.is_file() and metadata_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint:
            return PreparedGoldenDataset(prepared, True, manifest)

    columns = [
        data_config["sample_id_col"],
        data_config["payload_text_col"],
        data_config["label_col"],
    ]
    frame = pd.read_csv(source, usecols=columns, dtype=str).fillna("")
    frame.columns = ["sample_id", "payload_text", "label"]
    for column in frame.columns:
        frame[column] = frame[column].astype(str).str.strip()
    frame = frame[(frame["sample_id"] != "") & (frame["payload_text"] != "") & (frame["label"] != "")].copy()
    raw_counts = frame["label"].value_counts().to_dict()
    frame["payload_hash"] = frame["payload_text"].map(stable_text_hash)
    conflicts = frame.groupby("payload_hash")["label"].nunique()
    conflicting_hashes = set(conflicts[conflicts > 1].index.astype(str))
    conflicting_rows = int(frame["payload_hash"].isin(conflicting_hashes).sum())
    policy = str(data_config.get("conflicting_payload_policy", "exclude"))
    if conflicting_hashes and policy == "error":
        raise ValueError("The same payload hash has conflicting golden Tactic labels.")
    if policy not in {"exclude", "error"}:
        raise ValueError("conflicting_payload_policy must be 'exclude' or 'error'.")
    if conflicting_hashes:
        frame = frame[~frame["payload_hash"].isin(conflicting_hashes)].copy()
    rows_before_deduplication = len(frame)
    if bool(data_config.get("deduplicate_payloads", True)):
        frame = frame.drop_duplicates(subset=["payload_hash"], keep="first").copy()
    duplicate_rows_removed = int(rows_before_deduplication - len(frame))
    counts = frame["label"].value_counts()
    supported = counts[counts >= int(data_config.get("min_class_count", 1))].index.tolist()
    excluded = counts[counts < int(data_config.get("min_class_count", 1))].to_dict()
    frame = frame[frame["label"].isin(supported)].reset_index(drop=True)

    embedder = build_text_embedder(data_config["embedder"], device)
    x = embedder.encode(frame["payload_text"].tolist()).astype(np.float32)
    prepared.mkdir(parents=True, exist_ok=True)
    np.save(x_path, x)
    frame.drop(columns=["payload_text"]).to_csv(metadata_path, index=False)
    manifest = {
        "fingerprint": fingerprint,
        "rows": int(len(frame)),
        "embedding_dim": int(x.shape[1]),
        "raw_counts": {str(k): int(v) for k, v in raw_counts.items()},
        "supported_counts": {str(k): int(v) for k, v in frame["label"].value_counts().to_dict().items()},
        "excluded_low_support_counts": {str(k): int(v) for k, v in excluded.items()},
        "excluded_conflicting_payload_hashes": int(len(conflicting_hashes)),
        "excluded_conflicting_rows": conflicting_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return PreparedGoldenDataset(prepared, False, manifest)


def load_prepared(prepared_dir: str | Path) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    root = Path(prepared_dir)
    x = np.load(root / "x.npy", mmap_mode="r")
    metadata = pd.read_csv(root / "metadata.csv", dtype=str)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if len(x) != len(metadata):
        raise ValueError("Prepared x.npy and metadata.csv row counts differ.")
    return x, metadata, manifest


def make_stratified_group_split(
    metadata: pd.DataFrame,
    split_config: dict[str, Any],
    seed: int,
) -> SplitIndices:
    ratios = np.asarray(
        [split_config["train_ratio"], split_config["val_ratio"], split_config["test_ratio"]],
        dtype=float,
    )
    if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Golden split ratios must be positive and sum to 1.0.")
    groups = metadata.groupby("payload_hash", sort=False).agg(label=("label", "first"), row=("label", "size")).reset_index()
    group_ids = np.arange(len(groups))
    train_groups, remainder = train_test_split(
        group_ids,
        test_size=float(ratios[1] + ratios[2]),
        random_state=seed,
        stratify=groups["label"],
    )
    relative_test = float(ratios[2] / (ratios[1] + ratios[2]))
    val_groups, test_groups = train_test_split(
        remainder,
        test_size=relative_test,
        random_state=seed + 1,
        stratify=groups.iloc[remainder]["label"],
    )

    def rows(selected: np.ndarray) -> np.ndarray:
        hashes = set(groups.iloc[selected]["payload_hash"].astype(str))
        return np.flatnonzero(metadata["payload_hash"].astype(str).isin(hashes)).astype(np.int64)

    return SplitIndices(rows(train_groups), rows(val_groups), rows(test_groups))


def standardize(x: np.ndarray, split: SplitIndices) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler().fit(np.asarray(x[split.train], dtype=np.float32))
    return scaler.transform(np.asarray(x, dtype=np.float32)).astype(np.float32), scaler


def make_gate_targets(
    labels: np.ndarray,
    condition_labels: list[str],
    normal_label: str,
) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(condition_labels)}
    targets = np.zeros((len(labels), len(condition_labels)), dtype=np.float32)
    for row, label in enumerate(map(str, labels)):
        if label == normal_label:
            continue
        if label not in lookup:
            raise KeyError(f"No condition embedding for supported label: {label}")
        targets[row, lookup[label]] = 1.0
    return targets
