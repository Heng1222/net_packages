from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .dataset import DatasetBundle, SplitIndices


def _allocate_count(n: int, ratios: np.ndarray) -> np.ndarray:
    if n == 1:
        return np.array([1, 0, 0], dtype=int)
    if n == 2:
        return np.array([1, 0, 1], dtype=int)
    raw = n * ratios
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts))[: n - int(counts.sum())]:
        counts[index] += 1
    for index in range(3):
        if ratios[index] > 0 and counts[index] == 0:
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                continue
            counts[donor] -= 1
            counts[index] += 1
    return counts


def make_split(bundle: DatasetBundle, config: dict[str, Any], seed: int) -> SplitIndices:
    strategy = config.get("strategy", "stratified")
    ratios = np.array([config["train_ratio"], config["val_ratio"], config["test_ratio"]], dtype=float)
    rng = np.random.default_rng(seed)
    warnings: list[str] = []

    if strategy == "time":
        time_col = config.get("time_col")
        if not time_col or time_col not in bundle.metadata.columns:
            raise ValueError("Time split requires data.split.time_col to be present in data.metadata_cols.")
        parsed = pd.to_datetime(bundle.metadata[time_col], errors="raise")
        ordered = np.argsort(parsed.to_numpy(), kind="stable")
        counts = _allocate_count(len(ordered), ratios)
        train = ordered[: counts[0]]
        val = ordered[counts[0] : counts[0] + counts[1]]
        test = ordered[counts[0] + counts[1] :]
        return SplitIndices(train, val, test, warnings)

    if strategy == "stratified" and bundle.labels is not None:
        partitions: list[list[np.ndarray]] = [[], [], []]
        for label in sorted(np.unique(bundle.labels).tolist()):
            indices = np.flatnonzero(bundle.labels == label)
            rng.shuffle(indices)
            counts = _allocate_count(len(indices), ratios)
            if len(indices) < 3:
                warnings.append(
                    f"Rare class '{label}' has {len(indices)} sample(s); split counts are "
                    f"train={counts[0]}, val={counts[1]}, test={counts[2]}."
                )
            cursor = 0
            for split_index, count in enumerate(counts):
                partitions[split_index].append(indices[cursor : cursor + count])
                cursor += count
        arrays = [np.concatenate(parts) if parts else np.array([], dtype=int) for parts in partitions]
        for array in arrays:
            rng.shuffle(array)
        return SplitIndices(arrays[0], arrays[1], arrays[2], warnings)

    indices = np.arange(len(bundle.x))
    rng.shuffle(indices)
    counts = _allocate_count(len(indices), ratios)
    return SplitIndices(
        indices[: counts[0]],
        indices[counts[0] : counts[0] + counts[1]],
        indices[counts[0] + counts[1] :],
        warnings,
    )
