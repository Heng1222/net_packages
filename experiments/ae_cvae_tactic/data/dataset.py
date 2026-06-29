from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(slots=True)
class LoadedData:
    sample_ids: np.ndarray
    features: np.ndarray | None = None
    texts: list[str] | None = None
    labels: np.ndarray | None = None
    condition_keys: np.ndarray | None = None
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(slots=True)
class DatasetBundle:
    x: np.ndarray
    sample_ids: np.ndarray
    labels: np.ndarray | None = None
    condition_keys: np.ndarray | None = None
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float32)
        if self.x.ndim != 2:
            raise ValueError(f"Input features must be a 2D matrix; got shape {self.x.shape}")
        if len(self.x) != len(self.sample_ids):
            raise ValueError("Feature and sample_id row counts differ.")
        for name, values in (("labels", self.labels), ("condition_keys", self.condition_keys)):
            if values is not None and len(values) != len(self.x):
                raise ValueError(f"Feature and {name} row counts differ.")
        if not np.isfinite(self.x).all():
            raise ValueError("Input features contain NaN or infinite values.")


@dataclass(slots=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    warnings: list[str] = field(default_factory=list)

    def as_assignment(self, sample_ids: np.ndarray, labels: np.ndarray | None) -> pd.DataFrame:
        split = np.full(len(sample_ids), "", dtype="U5")
        split[self.train] = "train"
        split[self.val] = "val"
        split[self.test] = "test"
        frame = pd.DataFrame({"row_index": np.arange(len(sample_ids)), "sample_id": sample_ids, "split": split})
        if labels is not None:
            frame["label"] = labels
        return frame
