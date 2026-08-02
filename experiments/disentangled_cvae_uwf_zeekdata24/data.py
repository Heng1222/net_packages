from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from experiments.disentangled_cvae_step1.embedders import build_text_embedder

from .download import REQUIRED_COLUMNS


TACTIC_LABELS = (
    "Collection (TA0009)",
    "Command and Control (TA0011)",
    "Credential Access (TA0006)",
    "Defense Evasion (TA0005)",
    "Discovery (TA0007)",
    "Execution (TA0002)",
    "Exfiltration (TA0010)",
    "Initial Access (TA0001)",
    "Lateral Movement (TA0008)",
    "Persistence (TA0003)",
    "Privilege Escalation (TA0004)",
    "Reconnaissance (TA0043)",
    "Resource Development (TA0042)",
)

TECHNIQUE_LABELS = ("T1048", "T1078", "T1110", "T1190", "T1595")

T1078_TACTIC_LABELS = frozenset(
    {
        "Defense Evasion (TA0005)",
        "Initial Access (TA0001)",
        "Persistence (TA0003)",
        "Privilege Escalation (TA0004)",
    }
)

UWF_TACTIC_MAP = {
    "Credential Access": "Credential Access (TA0006)",
    "Defense Evasion": "Defense Evasion (TA0005)",
    "Exfiltration": "Exfiltration (TA0010)",
    "Initial Access": "Initial Access (TA0001)",
    "Persistence": "Persistence (TA0003)",
    "Privilege Escalation": "Privilege Escalation (TA0004)",
    "Reconnaissance": "Reconnaissance (TA0043)",
}

SHORTCUT_COLUMNS = {
    "community_id", "src_ip_zeek", "dest_ip_zeek", "ts", "uid", "datetime",
    "label_tactic", "label_technique", "label_binary", "label_cve",
}


@dataclass(slots=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass(slots=True)
class PreparedDataset:
    directory: Path
    reused: bool
    rows: int


def validate_frame_schema(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"UWF-ZeekData24 CSV is missing required columns: {missing}")


def _clean(value: Any, default: str = "none") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    text = str(value).strip()
    return text if text and text.lower() != "nan" else default


def _numeric_bucket(value: Any) -> str:
    try:
        number = max(0.0, float(value))
    except (TypeError, ValueError):
        return "unknown"
    if number == 0:
        return "zero"
    exponent = int(math.floor(math.log2(number)))
    return f"2^{exponent}..2^{exponent + 1}"


def _port_text(value: Any) -> str:
    try:
        port = int(float(value))
    except (TypeError, ValueError):
        return "unknown"
    kind = "well-known" if port < 1024 else "registered" if port < 49152 else "ephemeral"
    return f"{port} ({kind})"


def flow_to_text(row: pd.Series | dict[str, Any]) -> str:
    get = row.get
    services = ", ".join(sorted(part.strip() for part in _clean(get("service")).split(",") if part.strip()))
    fields = (
        ("protocol", _clean(get("proto"))),
        ("service", services or "none"),
        ("source port", _port_text(get("src_port_zeek"))),
        ("destination port", _port_text(get("dest_port_zeek"))),
        ("connection state", _clean(get("conn_state"))),
        ("connection history", _clean(get("history"))),
        ("local origin", _clean(get("local_orig"))),
        ("local response", _clean(get("local_resp"))),
        ("duration bucket", _numeric_bucket(get("duration"))),
        ("origin bytes bucket", _numeric_bucket(get("orig_bytes"))),
        ("response bytes bucket", _numeric_bucket(get("resp_bytes"))),
        ("origin IP bytes bucket", _numeric_bucket(get("orig_ip_bytes"))),
        ("response IP bytes bucket", _numeric_bucket(get("resp_ip_bytes"))),
        ("origin packets bucket", _numeric_bucket(get("orig_pkts"))),
        ("response packets bucket", _numeric_bucket(get("resp_pkts"))),
        ("missed bytes bucket", _numeric_bucket(get("missed_bytes"))),
    )
    return "; ".join(f"{name}: {value}" for name, value in fields)


def _normalize_tactic(value: Any) -> str | None:
    text = _clean(value, "")
    if not text or text.lower() == "none":
        return None
    text = text.replace("_", " ").title()
    return UWF_TACTIC_MAP.get(text)


def _normalize_technique(value: Any) -> str:
    text = _clean(value, "")
    if not text or text.lower() == "none":
        return "Benign"
    if text.lower() == "duplicate":
        raise ValueError("Duplicate sentinel rows must be removed before technique normalization.")
    match = re.search(r"\bT\d{4}(?:\.\d{3})?\b", text.upper())
    if match is None:
        raise ValueError(f"Cannot extract an ATT&CK technique ID from label_technique={text!r}")
    technique = match.group(0)
    if technique not in TECHNIQUE_LABELS:
        raise ValueError(f"Unexpected UWF-ZeekData24 technique: {technique}")
    return technique


def aggregate_flows(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("At least one UWF frame is required.")
    for frame in frames:
        validate_frame_schema(frame)
    source = pd.concat(frames, ignore_index=True)
    technique_text = source["label_technique"].astype(str).str.strip().str.casefold()
    binary_text = source["label_binary"].astype(str).str.strip().str.casefold()
    source["_duplicate_sentinel"] = technique_text.eq("duplicate") | binary_text.eq("duplicate")
    rows: list[dict[str, Any]] = []
    orphan_duplicate_uid_count = 0
    orphan_duplicate_row_count = 0
    for uid, group in source.groupby("uid", sort=True, dropna=False):
        if not str(uid).strip() or str(uid).lower() == "nan":
            raise ValueError("Every UWF row must have a non-empty uid.")
        canonical = group.loc[~group["_duplicate_sentinel"]]
        if canonical.empty:
            orphan_duplicate_uid_count += 1
            orphan_duplicate_row_count += int(len(group))
            continue
        techniques = sorted({_normalize_technique(value) for value in canonical["label_technique"]})
        non_benign = [value for value in techniques if value != "Benign"]
        if non_benign and "Benign" in techniques:
            raise ValueError(f"Conflicting technique labels for uid={uid}: {techniques}")
        tactics = set(filter(None, (_normalize_tactic(value) for value in canonical["label_tactic"])))
        unknown = sorted(
            {
                _clean(value)
                for value in canonical["label_tactic"]
                if _clean(value).lower() != "none" and _normalize_tactic(value) is None
            }
        )
        if unknown:
            raise ValueError(f"Unknown UWF tactic labels for uid={uid}: {unknown}")
        technique = "|".join(non_benign) if non_benign else "Benign"
        if "T1078" in non_benign:
            tactics.update(T1078_TACTIC_LABELS)
        first = canonical.iloc[0]
        rows.append(
            {
                "sample_id": str(uid),
                "datetime": _clean(first["datetime"], ""),
                "technique": technique,
                "probe_eligible": len(non_benign) == 1,
                "tactic_labels": "|".join(sorted(tactics)),
                "is_malicious": bool(non_benign),
                "flow_text": flow_to_text(first),
                "duplicate_source_rows": int(len(group)),
                "duplicate_sentinel_rows": int(group["_duplicate_sentinel"].sum()),
            }
        )
    if not rows:
        raise ValueError("No canonical labeled UWF flows remain after removing Duplicate sentinels.")
    result = pd.DataFrame(rows).sort_values("sample_id", kind="stable").reset_index(drop=True)
    result.attrs["aggregation_summary"] = {
        "source_rows": int(len(source)),
        "duplicate_sentinel_rows": int(source["_duplicate_sentinel"].sum()),
        "orphan_duplicate_uid_count": int(orphan_duplicate_uid_count),
        "orphan_duplicate_row_count": int(orphan_duplicate_row_count),
        "canonical_uid_count": int(len(result)),
    }
    return result


def sample_by_technique(frame: pd.DataFrame, caps: dict[str, Any] | None, seed: int) -> pd.DataFrame:
    caps = {str(key): int(value) for key, value in (caps or {}).items()}
    parts: list[pd.DataFrame] = []
    for technique, group in frame.groupby("technique", sort=True):
        cap = caps.get(str(technique))
        if cap is not None and len(group) > cap:
            group = group.sample(n=cap, replace=False, random_state=seed)
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sort_values("sample_id", kind="stable").reset_index(drop=True)


def stratified_technique_split(
    techniques: np.ndarray | pd.Series,
    split_config: dict[str, Any],
    seed: int,
) -> SplitIndices:
    values = np.asarray(techniques, dtype=str)
    ratios = np.asarray([split_config[key] for key in ("train_ratio", "val_ratio", "test_ratio")], dtype=float)
    if np.any(ratios < 0) or abs(float(ratios.sum()) - 1.0) > 1e-8:
        raise ValueError("Split ratios must be non-negative and sum to 1.0.")
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[], [], []]
    for label in sorted(set(values)):
        indices = np.flatnonzero(values == label)
        rng.shuffle(indices)
        count = len(indices)
        train_count = int(math.floor(count * ratios[0]))
        val_count = int(math.floor(count * ratios[1]))
        if count >= 3:
            train_count = max(1, min(train_count, count - 2))
            val_count = max(1, min(val_count, count - train_count - 1))
        test_count = count - train_count - val_count
        for bucket, selected in zip(buckets, (indices[:train_count], indices[train_count:train_count + val_count], indices[-test_count:] if test_count else []), strict=True):
            bucket.extend(map(int, selected))
    arrays = []
    for bucket in buckets:
        array = np.asarray(bucket, dtype=np.int64)
        rng.shuffle(array)
        arrays.append(array)
    if any(len(array) == 0 for array in arrays):
        raise ValueError("Stratified split produced an empty split.")
    return SplitIndices(*arrays)


def tactic_target_matrix(frame: pd.DataFrame, labels: tuple[str, ...] = TACTIC_LABELS) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(labels)}
    result = np.zeros((len(frame), len(labels)), dtype=np.float32)
    for row_index, value in enumerate(frame["tactic_labels"].fillna("")):
        for label in filter(None, str(value).split("|")):
            if label not in lookup:
                raise ValueError(f"Unknown normalized tactic label: {label}")
            result[row_index, lookup[label]] = 1.0
    return result


def _raw_fingerprint(raw_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    files = []
    for path in sorted(raw_dir.glob("*.csv")):
        stat = path.stat()
        files.append({"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"files": files, "class_caps": config.get("class_caps", {}), "embedder": config.get("embedder", {})}


def prepare_dataset(data_config: dict[str, Any], device: torch.device, seed: int, force: bool = False) -> PreparedDataset:
    raw_dir = Path(data_config["raw_dir"])
    paths = sorted(raw_dir.glob("*.csv"))
    if len(paths) != 8:
        raise FileNotFoundError(f"Expected eight downloaded UWF CSV files under {raw_dir}; found {len(paths)}")
    prepared = Path(data_config["prepared_dir"])
    prepared.mkdir(parents=True, exist_ok=True)
    manifest_path = prepared / "manifest.json"
    fingerprint = _raw_fingerprint(raw_dir, data_config)
    expected = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    required = [prepared / name for name in ("x.npy", "metadata.csv", "tactic_targets.npy", "split.npz")]
    if not force and manifest_path.is_file() and all(path.is_file() for path in required):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == expected:
            return PreparedDataset(prepared, True, int(manifest["rows"]))
    frames = [pd.read_csv(path, dtype=str) for path in paths]
    aggregated = aggregate_flows(frames)
    aggregation_summary = dict(aggregated.attrs["aggregation_summary"])
    metadata = sample_by_technique(aggregated, data_config.get("class_caps"), seed)
    observed_techniques = set(metadata["technique"])
    expected_techniques = {"Benign", *TECHNIQUE_LABELS}
    observed_single_techniques = {
        technique for technique in observed_techniques if "|" not in technique
    }
    if observed_single_techniques != expected_techniques:
        raise ValueError(
            "Prepared UWF data must contain Benign plus all five expected techniques; "
            f"expected={sorted(expected_techniques)}, observed={sorted(observed_single_techniques)}"
        )
    split = stratified_technique_split(metadata["technique"], data_config["split"], seed)
    targets = tactic_target_matrix(metadata)
    embedder = build_text_embedder(data_config["embedder"], device)
    x = embedder.encode(metadata["flow_text"].tolist()).astype(np.float32)
    np.save(prepared / "x.npy", x)
    np.save(prepared / "tactic_targets.npy", targets)
    np.savez_compressed(prepared / "split.npz", train=split.train, val=split.val, test=split.test)
    metadata.drop(columns=["flow_text"]).to_csv(prepared / "metadata.csv", index=False)
    split_summary = {
        name: {
            "rows": int(len(indices)),
            "techniques": metadata.iloc[indices]["technique"].value_counts().sort_index().to_dict(),
            "tactics": {label: int(targets[indices, position].sum()) for position, label in enumerate(TACTIC_LABELS)},
        }
        for name, indices in (("train", split.train), ("val", split.val), ("test", split.test))
    }
    manifest_path.write_text(
        json.dumps(
            {
                "fingerprint": expected,
                "rows": len(metadata),
                "duplicate_sentinel_rows_ignored": aggregation_summary["duplicate_sentinel_rows"],
                "ambiguous_technique_uid_count": int(
                    metadata["technique"].str.contains("|", regex=False).sum()
                ),
                "aggregation_summary": aggregation_summary,
                "split_summary": split_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PreparedDataset(prepared, False, len(metadata))


def load_prepared(directory: str | Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, SplitIndices]:
    root = Path(directory)
    x = np.load(root / "x.npy", mmap_mode="r")
    metadata = pd.read_csv(root / "metadata.csv", dtype={"sample_id": str, "technique": str, "tactic_labels": str})
    targets = np.load(root / "tactic_targets.npy")
    with np.load(root / "split.npz") as archive:
        split = SplitIndices(archive["train"], archive["val"], archive["test"])
    if not (len(x) == len(metadata) == len(targets)):
        raise ValueError("Prepared UWF artifacts have mismatched row counts.")
    return x, metadata, targets.astype(np.float32), split
