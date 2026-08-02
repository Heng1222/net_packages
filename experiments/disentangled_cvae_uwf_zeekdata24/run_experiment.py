from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from experiments.disentangled_cvae_step1.conditions import load_condition_embeddings  # noqa: E402
from experiments.disentangled_cvae_step1.data import standardize_to_memmap  # noqa: E402
from experiments.disentangled_cvae_step1.evaluate import (  # noqa: E402
    plot_condition_similarity_heatmap,
    plot_training_reconstruction_losses,
    plot_umap_projection,
    write_condition_ablation_summary,
    write_condition_gate_summary,
    write_similarity_matrix,
)
from experiments.disentangled_cvae_uwf_zeekdata24.data import (  # noqa: E402
    TACTIC_LABELS,
    load_prepared,
    prepare_dataset,
)
from experiments.disentangled_cvae_uwf_zeekdata24.download import download_dataset  # noqa: E402
from experiments.disentangled_cvae_uwf_zeekdata24.evaluate import (  # noqa: E402
    bootstrap_macro_f1_difference,
    bootstrap_mean_interval,
    calibrate_thresholds,
    majority_baseline,
    multilabel_metrics,
    observed_tactics,
    run_technique_probes,
    semantic_acceptance,
    shuffled_label_baseline,
)
from experiments.disentangled_cvae_uwf_zeekdata24.model import (  # noqa: E402
    MultiLabelDisentangledConditionalVAE,
)
from experiments.disentangled_cvae_uwf_zeekdata24.training import (  # noqa: E402
    extract_batches,
    train_model,
)
from experiments.disentangled_cvae_uwf_zeekdata24.utils import (  # noqa: E402
    capture_environment,
    load_config,
    make_run_dir,
    resolve_device,
    save_config,
    write_json,
)


def configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("disentangled_cvae_uwf_zeekdata24")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UWF-ZeekData24 disentangled CVAE validation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("download", "prepare", "train", "all"), default="all")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args()


def _standardize(config: dict, x_raw: np.ndarray, split, run_dir: Path) -> np.ndarray:
    method = config.get("preprocessing", {}).get("normalization", "standard")
    if method == "none":
        return x_raw
    if method != "standard":
        raise ValueError("preprocessing.normalization must be 'standard' or 'none'.")
    return standardize_to_memmap(
        x_raw,
        split,
        run_dir / "scalers" / "x_standard.npy",
        run_dir / "scalers" / "scaler.pkl",
        int(config.get("preprocessing", {}).get("batch_size", 20000)),
    )


def _malicious_representations(
    outputs: dict[str, dict[str, np.ndarray]],
    x: np.ndarray,
    metadata: pd.DataFrame,
    split,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    split_indices = (split.train, split.val, split.test)
    names = ("train", "val", "test")
    masks = tuple(metadata.iloc[indices]["technique"].to_numpy() != "Benign" for indices in split_indices)
    labels = tuple(metadata.iloc[indices]["technique"].to_numpy(dtype=str)[mask] for indices, mask in zip(split_indices, masks, strict=True))
    representations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for representation in ("h", "c", "hc"):
        representations[representation] = tuple(outputs[name][representation][mask] for name, mask in zip(names, masks, strict=True))
    representations["gates"] = tuple(outputs[name]["gates"][mask, : len(TACTIC_LABELS)] for name, mask in zip(names, masks, strict=True))
    representations["x"] = tuple(np.asarray(x[indices], dtype=np.float32)[mask] for indices, mask in zip(split_indices, masks, strict=True))
    return representations, labels


def _write_test_predictions(
    path: Path,
    metadata: pd.DataFrame,
    test_indices: np.ndarray,
    targets: np.ndarray,
    gates: np.ndarray,
    labels: list[str],
    thresholds: np.ndarray,
) -> None:
    frame = metadata.iloc[test_indices].reset_index(drop=True).copy()
    for index, label in enumerate(labels):
        frame[f"gold_tactic__{label}"] = targets[:, index].astype(np.int8)
        frame[f"condition_prob__{label}"] = gates[:, index]
        frame[f"predicted_tactic__{label}"] = (gates[:, index] >= thresholds[index]).astype(np.int8)
    frame["condition_prob__Common Tactic Component"] = gates[:, len(labels)]
    frame.to_csv(path, index=False)


def run_train(config: dict, run_dir: Path, device: torch.device, logger: logging.Logger) -> None:
    x_raw, metadata, targets, split = load_prepared(config["data"]["prepared_dir"])
    logger.info("Loaded prepared UWF dataset rows=%d dim=%d", len(metadata), x_raw.shape[1])
    x = _standardize(config, x_raw, split, run_dir)
    conditions = load_condition_embeddings(config["conditions"], None, device, run_dir / "embeddings")
    if tuple(conditions.tactic_labels) != TACTIC_LABELS:
        raise ValueError("Condition label order does not match the UWF multi-hot target order.")
    if len(conditions.labels) != 14 or conditions.labels[-1] != "Common Tactic Component":
        raise ValueError("This experiment requires 13 tactic conditions plus the common 14th conditional.")
    np.savez_compressed(
        run_dir / "embeddings" / "condition_embeddings.npz",
        labels=np.asarray(conditions.labels, dtype=str),
        tactic_labels=np.asarray(conditions.tactic_labels, dtype=str),
        matrix=conditions.matrix,
        raw_matrix=conditions.raw_matrix,
    )
    write_json(conditions.metadata, run_dir / "embeddings" / "condition_embeddings_metadata.json")
    model_config = dict(config["model"])
    model_config.update(
        {
            "condition_count": len(conditions.labels),
            "supervised_condition_count": len(conditions.tactic_labels),
            "condition_dim": conditions.dimension,
        }
    )
    model = MultiLabelDisentangledConditionalVAE.from_config(model_config)
    checkpoint = run_dir / "checkpoints" / "disentangled_cvae.pt"
    training_result = train_model(
        model, x, targets, conditions.matrix, split, config["training"], model_config,
        device, checkpoint, int(config["seed"]), logger,
    )
    write_json(
        {
            "best_epoch": training_result.best_epoch,
            "best_val_loss": training_result.best_val_loss,
            "pos_weight": training_result.pos_weight,
        },
        run_dir / "metrics" / "training_summary.json",
    )
    plot_training_reconstruction_losses(training_result.history, run_dir / "plots" / "training_reconstruction_losses.png")
    outputs = {
        name: extract_batches(
            model, x, targets, conditions.matrix, indices, device, int(config["training"]["batch_size"])
        )
        for name, indices in (("train", split.train), ("val", split.val), ("test", split.test))
    }
    evaluation = config["evaluation"]
    thresholds = calibrate_thresholds(
        targets[split.val], outputs["val"]["gates"][:, : len(TACTIC_LABELS)], evaluation["threshold_grid"]
    )
    observed = observed_tactics(targets[split.train])
    tactic_metrics = multilabel_metrics(
        targets[split.test], outputs["test"]["gates"][:, : len(TACTIC_LABELS)], thresholds,
        TACTIC_LABELS, observed,
    )
    shuffle = shuffled_label_baseline(
        targets[split.test], outputs["test"]["gates"][:, : len(TACTIC_LABELS)], thresholds,
        TACTIC_LABELS, int(evaluation["shuffle_repeats"]), int(evaluation["random_state"]),
    )
    majority = majority_baseline(targets[split.train], targets[split.test], TACTIC_LABELS)
    representations, technique_labels = _malicious_representations(outputs, x, metadata, split)
    probe = run_technique_probes(
        representations, *technique_labels, evaluation["probe_c_grid"], int(evaluation["random_state"])
    )
    probe_dir = run_dir / "probes"
    probe_dir.mkdir(exist_ok=True)
    for name, fitted_model in probe.models.items():
        joblib.dump(fitted_model, probe_dir / f"technique_probe_{name}.pkl")
    best_condition = max(("gates", "c"), key=lambda name: probe.validation_macro_f1[name])
    bootstrap_repeats = int(evaluation["bootstrap_repeats"])
    seed = int(evaluation["random_state"])
    technique_delta = bootstrap_macro_f1_difference(
        technique_labels[2], probe.predictions[best_condition], probe.predictions["h"],
        bootstrap_repeats, seed,
    )
    reconstruction_gain = bootstrap_mean_interval(
        outputs["test"]["h_only_mse"] - outputs["test"]["recon_mse"], bootstrap_repeats, seed + 1
    )
    reconstruction_metrics = {
        "full_mse": float(outputs["test"]["recon_mse"].mean()),
        "h_only_mse": float(outputs["test"]["h_only_mse"].mean()),
        "c_only_mse": float(outputs["test"]["c_only_mse"].mean()),
        "h_only_minus_full": reconstruction_gain,
        "c_only_minus_full": bootstrap_mean_interval(
            outputs["test"]["c_only_mse"] - outputs["test"]["recon_mse"],
            bootstrap_repeats,
            seed + 2,
        ),
    }
    acceptance = semantic_acceptance(
        tactic_metrics, shuffle, probe.metrics, best_condition, technique_delta, reconstruction_gain
    )
    common_gate = outputs["test"]["gates"][:, -1]
    binary_truth = metadata.iloc[split.test]["technique"].to_numpy() != "Benign"
    common_metrics = {
        "attack_vs_benign_roc_auc": float(roc_auc_score(binary_truth, common_gate))
        if len(np.unique(binary_truth)) == 2 else None,
        "mean_attack": float(common_gate[binary_truth].mean()) if binary_truth.any() else None,
        "mean_benign": float(common_gate[~binary_truth].mean()) if (~binary_truth).any() else None,
    }
    write_json({"thresholds": dict(zip(TACTIC_LABELS, thresholds, strict=True)), **tactic_metrics}, run_dir / "metrics" / "tactic_metrics.json")
    write_json(shuffle, run_dir / "metrics" / "label_shuffle_baseline.json")
    write_json(majority, run_dir / "metrics" / "majority_baseline.json")
    write_json(probe.metrics, run_dir / "metrics" / "probe_metrics.json")
    write_json(technique_delta, run_dir / "metrics" / "technique_probe_delta_bootstrap.json")
    write_json(reconstruction_gain, run_dir / "metrics" / "reconstruction_gain_bootstrap.json")
    write_json(reconstruction_metrics, run_dir / "metrics" / "reconstruction_metrics.json")
    write_json(common_metrics, run_dir / "metrics" / "common_condition_metrics.json")
    write_json(acceptance, run_dir / "metrics" / "semantic_acceptance.json")
    _write_test_predictions(
        run_dir / "metrics" / "testset_predictions.csv", metadata, split.test, targets[split.test],
        outputs["test"]["gates"], list(TACTIC_LABELS), thresholds,
    )
    technique_predictions = pd.DataFrame({"gold_technique": technique_labels[2]})
    for name, prediction in probe.predictions.items():
        technique_predictions[f"predicted__{name}"] = prediction
    technique_predictions.to_csv(run_dir / "metrics" / "technique_probe_predictions.csv", index=False)
    write_condition_gate_summary(outputs["test"]["gates"], conditions.labels, run_dir / "metrics" / "condition_gate_summary.csv")
    write_condition_ablation_summary(outputs["test"]["ablation_delta_mse"], conditions.labels, run_dir / "metrics" / "condition_ablation_summary.csv")
    raw_matrix = conditions.raw_matrix if conditions.raw_matrix is not None else conditions.matrix
    write_similarity_matrix(conditions.labels, raw_matrix, run_dir / "metrics" / "condition_raw_cosine_similarity.csv")
    write_similarity_matrix(conditions.labels, conditions.matrix, run_dir / "metrics" / "condition_cosine_similarity.csv")
    plot_condition_similarity_heatmap(conditions.labels, conditions.matrix, run_dir / "plots" / "condition_cosine_similarity.png", "UWF model-used condition cosine similarity")
    if evaluation.get("run_visualization", True):
        categories = metadata.iloc[split.test]["technique"].to_numpy(dtype=str)
        for name, values in (("original", np.asarray(x[split.test])), ("h", outputs["test"]["h"]), ("c", outputs["test"]["c"])):
            plot_umap_projection(values, run_dir / "plots" / f"umap_{name}_space.png", f"UWF {name} space", evaluation, categories)
    source_manifest = Path(config["data"]["raw_dir"]) / "source_manifest.json"
    if source_manifest.is_file():
        write_json(json.loads(source_manifest.read_text(encoding="utf-8")), run_dir / "metrics" / "source_manifest.json")
    write_report(run_dir, metadata, split, tactic_metrics, probe.metrics, best_condition, technique_delta, reconstruction_gain, acceptance)
    logger.info("Semantic acceptance supported=%s checks=%s", acceptance["supported"], acceptance["checks"])


def write_report(
    run_dir: Path,
    metadata: pd.DataFrame,
    split,
    tactic_metrics: dict,
    probe_metrics: dict,
    best_condition: str,
    technique_delta: dict,
    reconstruction_gain: dict,
    acceptance: dict,
) -> None:
    lines = [
        "# UWF-ZeekData24 Disentangled CVAE Report", "",
        "## Dataset", "",
        "- Source: UWF-ZeekData24 (CC BY 4.0).",
        "- Inputs are label-free serialized Zeek connection records, not packet payload bytes.",
        f"- Rows: {len(metadata)}; train/val/test: {len(split.train)}/{len(split.val)}/{len(split.test)}.",
        "- Official Duplicate technique sentinel rows restored as T1078: "
        f"{int(metadata['duplicate_sentinel_rows'].sum())}.",
        "- T1078 maps to four inseparable co-occurring tactic labels in this dataset.", "",
        "## Tactic gates", "",
        f"- Test micro/macro F1: {tactic_metrics['micro_f1']:.6f} / {tactic_metrics['macro_f1']:.6f}",
        f"- Test macro AUPRC: {tactic_metrics['macro_auprc']}", "",
        "## Frozen technique probes", "",
    ]
    lines.extend(f"- {name}: test macro F1={metrics.get('test_macro_f1')}" for name, metrics in probe_metrics.items())
    lines.extend(
        [
            "", "## Separation decision", "",
            f"- Best condition-derived representation: {best_condition}",
            f"- Condition minus H accuracy bootstrap: {technique_delta}",
            f"- H-only minus full reconstruction MSE bootstrap: {reconstruction_gain}",
            f"- Supported: **{acceptance['supported']}**", "",
        ]
    )
    lines.extend(f"- {key}: {value}" for key, value in acceptance["checks"].items())
    (run_dir / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    device = resolve_device(config["training"].get("device", "auto"))
    if args.stage in {"download", "all"}:
        manifest = download_dataset(config["data"], force=args.force_download)
        print(manifest)
    if args.stage in {"prepare", "all"}:
        prepared = prepare_dataset(config["data"], device, int(config["seed"]), force=args.force_prepare)
        print(prepared.directory)
    if args.stage in {"train", "all"}:
        run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else make_run_dir(config["output"]["base_dir"])
        for name in ("logs", "checkpoints", "scalers", "metrics", "plots", "reports", "embeddings"):
            (run_dir / name).mkdir(parents=True, exist_ok=True)
        logger = configure_logging(run_dir / "logs" / "experiment.log")
        try:
            save_config(config, run_dir / "config_resolved.yaml")
            capture_environment(run_dir / "environment.json")
            run_train(config, run_dir, device, logger)
        finally:
            for handler in tuple(logger.handlers):
                handler.flush()
                handler.close()
                logger.removeHandler(handler)
        print(run_dir)


if __name__ == "__main__":
    main()
