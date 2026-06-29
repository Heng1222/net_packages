from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, Normalizer, StandardScaler

from .dataset import SplitIndices


def build_transformer(name: str) -> TransformerMixin:
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "l2":
        return Normalizer(norm="l2")
    if name == "none":
        return FunctionTransformer(validate=False)
    raise ValueError(f"Unknown normalization: {name}")


def fit_transform_splits(
    x: np.ndarray, split: SplitIndices, normalization: str, scaler_path: str
) -> tuple[np.ndarray, TransformerMixin]:
    transformer = build_transformer(normalization)
    transformer.fit(x[split.train])
    transformed = transformer.transform(x).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("Preprocessing produced NaN or infinite values.")
    joblib.dump(transformer, scaler_path)
    return transformed, transformer
