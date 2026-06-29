from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_classifier(config: dict[str, Any]) -> BaseEstimator:
    kind = config.get("type", "logistic_regression")
    random_state = int(config.get("random_state", 42))
    max_iter = int(config.get("max_iter", 1000))
    class_weight = config.get("class_weight")
    if class_weight == "none":
        class_weight = None
    if kind == "logistic_regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=max_iter, class_weight=class_weight, random_state=random_state)),
            ]
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight=class_weight, random_state=random_state, n_jobs=-1
        )
    if kind == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=tuple(config.get("mlp_hidden_dims", [128, 64])),
                        max_iter=max_iter,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError("classifier.type must be logistic_regression, random_forest, or mlp.")
