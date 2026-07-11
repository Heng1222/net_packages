from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F


if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from experiments.disentangled_cvae_step1.conditions import load_condition_embeddings  # noqa: E402
from experiments.disentangled_cvae_step1.evaluate import (  # noqa: E402
    plot_condition_similarity_heatmap,
    write_similarity_matrix,
)
from experiments.disentangled_cvae_step1.utils import (  # noqa: E402
    capture_environment,
    make_run_dir,
    resolve_device,
    write_json,
)
from experiments.golden_oracle_cvae_step2.data import (  # noqa: E402
    load_prepared,
    make_gate_targets,
    make_stratified_group_split,
    prepare_golden_dataset,
    standardize,
)
from experiments.golden_oracle_cvae_step2.evaluate import (  # noqa: E402
    classification_metrics,
    plot_confusions,
    plot_histories,
)
from experiments.golden_oracle_cvae_step2.model import GoldenConditionalVAE, PayloadClassifier  # noqa: E402
from experiments.golden_oracle_cvae_step2.training import train_classifier, train_cvae  # noqa: E402


PATHS = (
    ("data", "input_path"),
    ("data", "prepared_dir"),
    ("conditions", "path"),
    ("output", "base_dir"),
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping.")
    config = deepcopy(config)
    for section, key in PATHS:
        value = Path(str(config[section][key]))
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        config[section][key] = str(value.resolve())
    ratios = config["data"]["split"]
    if not np.isclose(sum(float(ratios[key]) for key in ("train_ratio", "val_ratio", "test_ratio")), 1.0):
        raise ValueError("Split ratios must sum to 1.0.")
    config["_meta"] = {"config_path": str(config_path), "project_root": str(PROJECT_ROOT)}
    return config


def save_config(config: dict[str, Any], path: Path) -> None:
    payload = deepcopy(config)
    payload.pop("_meta", None)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("golden_oracle_cvae_step2")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Golden-only Oracle/Predicted Gate CVAE feasibility experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "all"), default="all")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--run-dir")
    return parser.parse_args()


def _model(config: dict[str, Any], condition_count: int, input_dim: int, condition_dim: int) -> GoldenConditionalVAE:
    values = dict(config["model"])
    values.update(input_dim=input_dim, condition_dim=condition_dim, condition_count=condition_count)
    return GoldenConditionalVAE(**values)


def _split_summary(metadata: pd.DataFrame, split) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, indices in (("train", split.train), ("val", split.val), ("test", split.test)):
        result[name] = {
            "rows": int(len(indices)),
            "counts_by_label": {
                str(k): int(v) for k, v in metadata.iloc[indices]["label"].value_counts().to_dict().items()
            },
            "unique_payload_hashes": int(metadata.iloc[indices]["payload_hash"].nunique()),
        }
    hash_sets = [set(metadata.iloc[idx]["payload_hash"]) for idx in (split.train, split.val, split.test)]
    result["payload_hash_overlap"] = {
        "train_val": len(hash_sets[0] & hash_sets[1]),
        "train_test": len(hash_sets[0] & hash_sets[2]),
        "val_test": len(hash_sets[1] & hash_sets[2]),
    }
    return result


@torch.inference_mode()
def _evaluate(
    oracle: GoldenConditionalVAE,
    predicted: GoldenConditionalVAE,
    classifier: PayloadClassifier,
    x: np.ndarray,
    metadata: pd.DataFrame,
    test_indices: np.ndarray,
    gate_targets: np.ndarray,
    class_targets: np.ndarray,
    condition_matrix: np.ndarray,
    condition_labels: list[str],
    class_labels: list[str],
    normal_label: str,
    threshold: float,
    device: torch.device,
    seed: int,
    majority_class: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    oracle.eval()
    predicted.eval()
    classifier.eval()
    xb = torch.from_numpy(np.asarray(x[test_indices], dtype=np.float32)).to(device)
    gold_gates = torch.from_numpy(gate_targets[test_indices]).to(device)
    conditions = torch.from_numpy(np.asarray(condition_matrix, dtype=np.float32)).to(device)
    zero_gates = torch.zeros_like(gold_gates)
    permutation = np.random.default_rng(seed + 100).permutation(len(test_indices))
    shuffled_gates = gold_gates[torch.from_numpy(permutation).to(device)]

    oracle_output = oracle(xb, conditions, gates_override=gold_gates, sample=False)
    oracle_zero = oracle.decode(oracle_output["h"], zero_gates, conditions)
    oracle_shuffled = oracle.decode(oracle_output["h"], shuffled_gates, conditions)
    predicted_output = predicted(xb, conditions, sample=False)
    predicted_zero = predicted.decode(predicted_output["h"], zero_gates, conditions)
    classifier_logits = classifier(xb)

    def row_mse(value: torch.Tensor) -> np.ndarray:
        return F.mse_loss(value, xb, reduction="none").mean(dim=1).cpu().numpy()

    oracle_mse = row_mse(oracle_output["x_recon"])
    oracle_zero_mse = row_mse(oracle_zero)
    oracle_shuffled_mse = row_mse(oracle_shuffled)
    predicted_mse = row_mse(predicted_output["x_recon"])
    predicted_zero_mse = row_mse(predicted_zero)
    gates = predicted_output["predicted_gates"].cpu().numpy()
    max_indices = gates.argmax(axis=1)
    max_values = gates[np.arange(len(gates)), max_indices]
    normal_index = class_labels.index(normal_label)
    condition_to_class = np.asarray([class_labels.index(label) for label in condition_labels], dtype=np.int64)
    predicted_classes = np.where(max_values >= threshold, condition_to_class[max_indices], normal_index)
    classifier_classes = classifier_logits.argmax(dim=1).cpu().numpy()
    truth = class_targets[test_indices]

    majority_classes = np.full(len(truth), int(majority_class), dtype=np.int64)
    metrics = {
        "classifier": classification_metrics(truth, classifier_classes, class_labels),
        "predicted_gate": classification_metrics(truth, predicted_classes, class_labels),
        "majority": classification_metrics(truth, majority_classes, class_labels),
        "reconstruction": {
            "oracle_gold_mse": float(oracle_mse.mean()),
            "oracle_zero_mse": float(oracle_zero_mse.mean()),
            "oracle_shuffled_mse": float(oracle_shuffled_mse.mean()),
            "oracle_zero_gain": float((oracle_zero_mse - oracle_mse).mean()),
            "oracle_shuffled_gain": float((oracle_shuffled_mse - oracle_mse).mean()),
            "predicted_gate_mse": float(predicted_mse.mean()),
            "predicted_zero_mse": float(predicted_zero_mse.mean()),
            "predicted_condition_gain": float((predicted_zero_mse - predicted_mse).mean()),
        },
    }
    metrics["decision"] = {
        "oracle_beats_zero": metrics["reconstruction"]["oracle_zero_gain"] > 0,
        "oracle_beats_shuffled": metrics["reconstruction"]["oracle_shuffled_gain"] > 0,
        "predicted_beats_majority_macro_f1": (
            metrics["predicted_gate"]["macro_f1"] > metrics["majority"]["macro_f1"]
        ),
        "predicted_condition_is_used": metrics["reconstruction"]["predicted_condition_gain"] > 0,
    }
    frame = metadata.iloc[test_indices].reset_index(drop=True).copy()
    frame["gold_label"] = [class_labels[index] for index in truth]
    frame["classifier_prediction"] = [class_labels[index] for index in classifier_classes]
    frame["predicted_gate_prediction"] = [class_labels[index] for index in predicted_classes]
    frame["max_gate"] = max_values
    frame["oracle_gold_mse"] = oracle_mse
    frame["oracle_zero_mse"] = oracle_zero_mse
    frame["oracle_shuffled_mse"] = oracle_shuffled_mse
    frame["predicted_gate_mse"] = predicted_mse
    frame["predicted_zero_mse"] = predicted_zero_mse
    for column, label in enumerate(condition_labels):
        frame[f"gate__{label}"] = gates[:, column]
    return frame, metrics, {
        "truth": truth,
        "classifier": classifier_classes,
        "predicted_gate": predicted_classes,
    }


def _write_report(
    run_dir: Path,
    manifest: dict[str, Any],
    split_summary: dict[str, Any],
    condition_labels: list[str],
    metrics: dict[str, Any],
) -> None:
    rec = metrics["reconstruction"]
    decision = metrics["decision"]
    lines = [
        "# Golden-only Oracle/Predicted Gate CVAE Report",
        "",
        "## Scope",
        "",
        "This is a small labeled feasibility experiment, not a chronological Step1 generalization result.",
        "Gold Tactic is fed only to the Oracle model; predicted-gate and classifier test inference never receive it.",
        "",
        "## Data",
        "",
        f"- Prepared rows: {manifest['rows']}",
        f"- Supported counts: {manifest['supported_counts']}",
        f"- Excluded low-support counts: {manifest['excluded_low_support_counts']}",
        f"- Split summary: {split_summary}",
        f"- Condition labels: {condition_labels}",
        "",
        "## Classification",
        "",
        f"- Plain classifier macro F1: {metrics['classifier']['macro_f1']:.6f}",
        f"- Predicted-gate macro F1: {metrics['predicted_gate']['macro_f1']:.6f}",
        f"- Majority macro F1: {metrics['majority']['macro_f1']:.6f}",
        "",
        "## Reconstruction controls",
        "",
        f"- Oracle gold MSE: {rec['oracle_gold_mse']:.6f}",
        f"- Oracle zero MSE: {rec['oracle_zero_mse']:.6f}",
        f"- Oracle shuffled MSE: {rec['oracle_shuffled_mse']:.6f}",
        f"- Oracle zero gain: {rec['oracle_zero_gain']:.6f}",
        f"- Oracle shuffled gain: {rec['oracle_shuffled_gain']:.6f}",
        f"- Predicted-gate MSE: {rec['predicted_gate_mse']:.6f}",
        f"- Predicted-zero MSE: {rec['predicted_zero_mse']:.6f}",
        f"- Predicted condition gain: {rec['predicted_condition_gain']:.6f}",
        "",
        "## Decision",
        "",
        *[f"- {key}: {value}" for key, value in decision.items()],
        "",
        "Oracle success establishes only that a known label can help the conditional pathway. Deployable evidence requires the predicted-gate model to classify held-out payloads without gold input.",
    ]
    (run_dir / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_train(config: dict[str, Any], run_dir: Path, device: torch.device, logger: logging.Logger) -> None:
    x_raw, metadata, manifest = load_prepared(config["data"]["prepared_dir"])
    split = make_stratified_group_split(metadata, config["data"]["split"], int(config["seed"]))
    split_summary = _split_summary(metadata, split)
    write_json(split_summary, run_dir / "metrics" / "split_summary.json")
    assignments = np.full(len(metadata), "", dtype=object)
    assignments[split.train], assignments[split.val], assignments[split.test] = "train", "val", "test"
    metadata.assign(split=assignments).to_csv(run_dir / "metrics" / "split_assignments.csv", index=False)
    x, scaler = standardize(x_raw, split)
    np.save(run_dir / "scalers" / "x_standard.npy", x)
    joblib.dump(scaler, run_dir / "scalers" / "scaler.pkl")

    normal_label = str(config["data"]["normal_label"])
    observed = sorted(set(metadata["label"].astype(str)))
    if normal_label not in observed:
        raise ValueError("Golden-only experiment requires supported Normal rows for all-zero gate controls.")
    malicious = [label for label in observed if label != normal_label]
    condition_bundle = load_condition_embeddings(
        config["conditions"], malicious, device, run_dir / "embeddings"
    )
    condition_labels = condition_bundle.labels
    condition_matrix = condition_bundle.matrix
    np.savez_compressed(
        run_dir / "embeddings" / "condition_embeddings.npz",
        labels=np.asarray(condition_labels),
        matrix=condition_matrix,
        raw_matrix=condition_bundle.raw_matrix,
    )
    write_json(condition_bundle.metadata, run_dir / "embeddings" / "condition_embeddings_metadata.json")
    raw_conditions = condition_bundle.raw_matrix if condition_bundle.raw_matrix is not None else condition_matrix
    write_similarity_matrix(
        condition_labels, raw_conditions, run_dir / "metrics" / "condition_raw_cosine_similarity.csv"
    )
    write_similarity_matrix(
        condition_labels, condition_matrix, run_dir / "metrics" / "condition_cosine_similarity.csv"
    )
    plot_condition_similarity_heatmap(
        condition_labels, raw_conditions, run_dir / "plots" / "condition_raw_cosine_similarity.png",
        "Raw condition cosine similarity",
    )
    plot_condition_similarity_heatmap(
        condition_labels, condition_matrix, run_dir / "plots" / "condition_cosine_similarity.png",
        "Model-used condition cosine similarity",
    )
    class_labels = [*condition_labels, normal_label]
    class_lookup = {label: index for index, label in enumerate(class_labels)}
    class_targets = metadata["label"].map(class_lookup).to_numpy(dtype=np.int64)
    gate_targets = make_gate_targets(metadata["label"].to_numpy(), condition_labels, normal_label)

    oracle = _model(config, len(condition_labels), x.shape[1], condition_matrix.shape[1])
    predicted = _model(config, len(condition_labels), x.shape[1], condition_matrix.shape[1])
    classifier = PayloadClassifier(
        x.shape[1], len(class_labels), config["model"]["classifier_hidden_dims"],
        float(config["model"]["dropout"]), str(config["model"]["activation"]),
    )
    oracle_result = train_cvae(
        "oracle", oracle, x, gate_targets, condition_matrix, split.train, split.val,
        config["training"], config["loss"], device, run_dir / "checkpoints" / "oracle_cvae.pt",
        int(config["seed"]), logger,
    )
    predicted_result = train_cvae(
        "predicted", predicted, x, gate_targets, condition_matrix, split.train, split.val,
        config["training"], config["loss"], device, run_dir / "checkpoints" / "predicted_gate_cvae.pt",
        int(config["seed"]), logger,
    )
    classifier_result = train_classifier(
        classifier, x, class_targets, split.train, split.val, config["training"], device,
        run_dir / "checkpoints" / "payload_classifier.pt", int(config["seed"]), logger,
    )
    histories = {
        "oracle": oracle_result.history,
        "predicted_gate": predicted_result.history,
        "classifier": classifier_result.history,
    }
    for name, rows in histories.items():
        pd.DataFrame(rows).to_csv(run_dir / "metrics" / f"training_history_{name}.csv", index=False)
    combined_history = pd.concat(
        [pd.DataFrame(rows).assign(model=name) for name, rows in histories.items()], ignore_index=True
    )
    combined_history.to_csv(run_dir / "metrics" / "training_history.csv", index=False)
    write_json(
        {
            "oracle": {"best_epoch": oracle_result.best_epoch, "best_val_loss": oracle_result.best_val_loss},
            "predicted_gate": {"best_epoch": predicted_result.best_epoch, "best_val_loss": predicted_result.best_val_loss},
            "classifier": {"best_epoch": classifier_result.best_epoch, "best_val_loss": classifier_result.best_val_loss},
        },
        run_dir / "metrics" / "training_summary.json",
    )
    predictions, metrics, confusion_values = _evaluate(
        oracle, predicted, classifier, x, metadata, split.test, gate_targets, class_targets,
        condition_matrix, condition_labels, class_labels, normal_label,
        float(config["evaluation"]["condition_threshold"]), device, int(config["seed"]),
        int(np.bincount(class_targets[split.train], minlength=len(class_labels)).argmax()),
    )
    predictions.to_csv(run_dir / "metrics" / "test_predictions.csv", index=False)
    predictions.to_csv(run_dir / "metrics" / "testset_condition_predictions.csv", index=False)
    predictions.groupby("predicted_gate_prediction", sort=False, group_keys=False).head(100).to_csv(
        run_dir / "metrics" / "testset_subset_100.csv", index=False
    )
    write_json(metrics, run_dir / "metrics" / "model_comparison.json")
    write_json(metrics["reconstruction"], run_dir / "metrics" / "loss_summary.json")
    write_json(
        {key: metrics[key] for key in ("classifier", "predicted_gate", "majority")},
        run_dir / "metrics" / "behavior_alignment_metrics.json",
    )
    gate_columns = [column for column in predictions if column.startswith("gate__")]
    predictions[gate_columns].describe(percentiles=[0.5, 0.9, 0.99]).T.to_csv(
        run_dir / "metrics" / "predicted_gate_summary.csv"
    )
    threshold = float(config["evaluation"]["condition_threshold"])
    gate_summary = []
    for column, label in zip(gate_columns, condition_labels, strict=True):
        values = predictions[column]
        gate_summary.append(
            {
                "condition": label,
                "mean_gate": float(values.mean()),
                "std_gate": float(values.std(ddof=0)),
                "p50_gate": float(values.quantile(0.50)),
                "p90_gate": float(values.quantile(0.90)),
                "p99_gate": float(values.quantile(0.99)),
                "active_rate": float((values >= threshold).mean()),
            }
        )
    pd.DataFrame(gate_summary).to_csv(run_dir / "metrics" / "condition_gate_summary.csv", index=False)
    ablation = predictions.assign(
        delta_mse=predictions["oracle_zero_mse"] - predictions["oracle_gold_mse"]
    )
    ablation = ablation[ablation["gold_label"] != normal_label]
    ablation.groupby("gold_label")["delta_mse"].agg(
        count="count", mean_delta_mse="mean", std_delta_mse="std",
        p50_delta_mse="median", p90_delta_mse=lambda values: values.quantile(0.90),
    ).reset_index(names="condition").to_csv(
        run_dir / "metrics" / "condition_ablation_delta_mse_summary.csv", index=False
    )
    plot_histories(histories, run_dir / "plots" / "training_reconstruction_losses.png")
    plot_confusions(
        confusion_values["truth"],
        {"plain classifier": confusion_values["classifier"], "predicted gate": confusion_values["predicted_gate"]},
        class_labels,
        run_dir / "plots" / "confusion_matrices.png",
    )
    _write_report(run_dir, manifest, split_summary, condition_labels, metrics)
    logger.info("Finished golden-only experiment: %s", run_dir)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["training"]["device"])
    run_dir = Path(args.run_dir).resolve() if args.run_dir else make_run_dir(config["output"]["base_dir"])
    for folder in ("logs", "checkpoints", "scalers", "metrics", "plots", "reports", "embeddings"):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_dir / "logs" / "experiment.log")
    save_config(config, run_dir / "config_resolved.yaml")
    capture_environment(run_dir / "environment.json")
    logger.info("stage=%s device=%s run_dir=%s", args.stage, device, run_dir)
    if args.stage in {"prepare", "all"}:
        prepared = prepare_golden_dataset(config["data"], PROJECT_ROOT, device, args.force_prepare)
        logger.info("prepared=%s reused=%s rows=%s", prepared.prepared_dir, prepared.reused, prepared.summary.get("rows"))
    if args.stage in {"train", "all"}:
        run_train(config, run_dir, device, logger)
    print(run_dir)


if __name__ == "__main__":
    main()
