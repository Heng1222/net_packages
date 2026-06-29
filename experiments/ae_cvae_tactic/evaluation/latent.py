from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def export_latent(
    path: str | Path,
    latent: np.ndarray,
    sample_ids: np.ndarray,
    labels: np.ndarray | None,
    split: str,
    fmt: str = "npz",
) -> Path:
    target = Path(path)
    safe_labels = labels.astype(str) if labels is not None else np.full(len(latent), "", dtype=str)
    if fmt == "npz":
        target = target.with_suffix(".npz")
        np.savez_compressed(
            target,
            latent=np.asarray(latent, dtype=np.float32),
            sample_id=np.asarray(sample_ids, dtype=str),
            label=np.asarray(safe_labels, dtype=str),
            split=np.full(len(latent), split, dtype=str),
        )
        return target
    if fmt == "parquet":
        target = target.with_suffix(".parquet")
        frame = pd.DataFrame(latent, columns=[f"latent_{index:04d}" for index in range(latent.shape[1])])
        frame.insert(0, "split", split)
        frame.insert(0, "label", safe_labels)
        frame.insert(0, "sample_id", sample_ids)
        try:
            frame.to_parquet(target, index=False)
        except ImportError as exc:
            raise RuntimeError("Parquet export requires pyarrow or fastparquet; use output.latent_format=npz.") from exc
        return target
    raise ValueError("output.latent_format must be npz or parquet.")
