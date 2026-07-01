from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from experiments.disentangled_cvae_step1.conditions import load_condition_embeddings  # noqa: E402
from experiments.disentangled_cvae_step1.data import (  # noqa: E402
    leakage_report,
    load_prepared,
    make_time_split,
    prepare_dataset,
    standardize_to_memmap,
)
from experiments.disentangled_cvae_step1.evaluate import (  # noqa: E402
    AMBIGUOUS_LABEL,
    build_test_condition_predictions,
    plot_condition_similarity_heatmap,
    plot_training_reconstruction_losses,
    plot_umap_projection,
    write_condition_ablation_summary,
    write_condition_gate_summary,
    write_similarity_matrix,
    write_testset_subset,
)
from experiments.disentangled_cvae_step1.model import DisentangledConditionalVAE  # noqa: E402
from experiments.disentangled_cvae_step1.training import extract_batches, train_model  # noqa: E402
from experiments.disentangled_cvae_step1.utils import (  # noqa: E402
    capture_environment,
    load_config,
    make_run_dir,
    resolve_device,
    save_config,
    write_json,
)


def configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("disentangled_cvae_step1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step1 disentangled CVAE experiment")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--stage", choices=("prepare", "train", "all"), default="all")
    parser.add_argument("--run-dir", default=None, help="Existing/new run directory for train outputs")
    parser.add_argument("--force-prepare", action="store_true", help="Regenerate prepared embedding cache")
    return parser.parse_args()


def _make_split_assignments(metadata: pd.DataFrame, split) -> pd.DataFrame:
    values = np.full(len(metadata), "", dtype=object)
    values[split.train] = "train"
    values[split.val] = "val"
    values[split.test] = "test"
    return pd.DataFrame(
        {
            "row_index": np.arange(len(metadata)),
            "sample_id": metadata["sample_id"].astype(str),
            "split": values,
        }
    )


def run_prepare(config: dict, device: torch.device, force: bool, logger: logging.Logger):
    prepared = prepare_dataset(config, PROJECT_ROOT, device, force=force)
    logger.info("Prepared dataset: %s (reused=%s)", prepared.prepared_dir, prepared.reused)
    return prepared


def run_train(config: dict, run_dir: Path, device: torch.device, logger: logging.Logger) -> None:
    prepared_dir = Path(config["data"]["prepared_dir"])
    x_raw, metadata = load_prepared(prepared_dir)
    logger.info("Loaded prepared dataset rows=%d dim=%d", len(x_raw), x_raw.shape[1])

    split = make_time_split(metadata, config["data"]["split"])
    _make_split_assignments(metadata, split).to_csv(run_dir / "metrics" / "split_assignments.csv", index=False)
    write_json(leakage_report(metadata, split), run_dir / "metrics" / "leakage_report.json")

    if config.get("preprocessing", {}).get("normalization", "standard") == "standard":
        x = standardize_to_memmap(
            x_raw,
            split,
            run_dir / "scalers" / "x_standard.npy",
            run_dir / "scalers" / "scaler.pkl",
            int(config.get("preprocessing", {}).get("batch_size", 20000)),
        )
    elif config.get("preprocessing", {}).get("normalization", "standard") == "none":
        x = x_raw
    else:
        raise ValueError("This experiment supports preprocessing.normalization='standard' or 'none'.")

    conditions = load_condition_embeddings(
        config["conditions"], None, device, run_dir / "embeddings"
    )
    if conditions.dimension != int(config["model"]["condition_dim"]):
        raise ValueError(
            f"Condition dim {conditions.dimension} does not match condition_dim={config['model']['condition_dim']}."
        )
    model_config = dict(config["model"])
    model_config["condition_count"] = len(conditions.labels)
    model_config["condition_dim"] = conditions.dimension
    np.savez_compressed(
        run_dir / "embeddings" / "condition_embeddings.npz",
        labels=np.asarray(conditions.labels, dtype=str),
        matrix=conditions.matrix,
    )

    model = DisentangledConditionalVAE.from_config(model_config)
    checkpoint = run_dir / "checkpoints" / "disentangled_cvae.pt"
    result = train_model(
        model,
        x,
        conditions.matrix,
        split,
        config["training"],
        model_config,
        device,
        checkpoint,
        int(config.get("seed", 42)),
    )
    write_json(
        {"best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss},
        run_dir / "metrics" / "training_summary.json",
    )
    plot_training_reconstruction_losses(result.history, run_dir / "plots" / "training_reconstruction_losses.png")

    test_outputs = extract_batches(
        model,
        x,
        conditions.matrix,
        split.test,
        device,
        int(config["training"]["batch_size"]),
    )
    test_predictions = build_test_condition_predictions(
        metadata,
        split.test,
        test_outputs["gates"],
        conditions.labels,
    )
    test_predictions.to_csv(run_dir / "metrics" / "testset_condition_predictions.csv", index=False)
    write_testset_subset(test_predictions, run_dir / "metrics" / "testset_subset_100.csv")
    write_json(test_outputs["loss_summary"], run_dir / "metrics" / "loss_summary.json")
    write_condition_gate_summary(
        test_outputs["gates"], conditions.labels, run_dir / "metrics" / "condition_gate_summary.csv"
    )
    write_condition_ablation_summary(
        test_outputs["ablation_delta_mse"],
        conditions.labels,
        run_dir / "metrics" / "condition_ablation_delta_mse_summary.csv",
    )

    write_similarity_matrix(
        conditions.labels,
        conditions.matrix,
        run_dir / "metrics" / "condition_cosine_similarity.csv",
    )
    plot_condition_similarity_heatmap(
        conditions.labels,
        conditions.matrix,
        run_dir / "plots" / "condition_cosine_similarity.png",
        "Condition cosine similarity",
    )

    if config.get("evaluation", {}).get("run_visualization", True):
        predicted_conditions = test_predictions["predicted_condition"].to_numpy()
        category_order = [*conditions.labels, AMBIGUOUS_LABEL]
        plot_umap_projection(
            np.asarray(x[split.test], dtype=np.float32),
            run_dir / "plots" / "umap_original_space.png",
            "Original payload embedding space",
            config["evaluation"],
            predicted_conditions,
            category_order,
        )
        plot_umap_projection(
            test_outputs["h"],
            run_dir / "plots" / "umap_h_space.png",
            "Residual H space",
            config["evaluation"],
            predicted_conditions,
            category_order,
        )
        plot_umap_projection(
            test_outputs["c"],
            run_dir / "plots" / "umap_gated_c_space.png",
            "Gated filtered C semantic space",
            config["evaluation"],
            predicted_conditions,
            category_order,
        )

    write_report(run_dir, metadata, split, conditions.labels, test_outputs["loss_summary"])


def write_report(
    run_dir: Path,
    metadata: pd.DataFrame,
    split,
    condition_labels: list[str],
    loss_summary: dict,
) -> None:
    lines = [
        "# Step1 Disentangled CVAE Report",
        "",
        "## Data",
        "",
        f"- Rows: {len(metadata)}",
        f"- Train/val/test: {len(split.train)} / {len(split.val)} / {len(split.test)}",
        "- Normal (TA9000) is not a condition.",
        "- `Sess_Tactic_predict` is retained in prepared metadata only; it is not used for training.",
        "- Test-set `predicted_condition` is derived from softmax-normalized CVAE condition gates.",
        "",
        "## Condition Labels",
        "",
        *[f"- {label}" for label in condition_labels],
        "",
        "## Reconstruction",
        "",
        f"- Test full reconstruction MSE: {loss_summary.get('recon_mse')}",
        f"- Test H-only reconstruction MSE: {loss_summary.get('h_only_mse')}",
        f"- Test C-only reconstruction MSE: {loss_summary.get('c_only_mse')}",
        "",
    ]
    lines.extend(
        [
            "## Figures",
            "",
            "- `plots/condition_cosine_similarity.png`",
            "- `plots/training_reconstruction_losses.png`",
            "- `plots/umap_original_space.png`",
            "- `plots/umap_h_space.png`",
            "- `plots/umap_gated_c_space.png`",
            "",
            "## Test Classification Outputs",
            "",
            "- `metrics/testset_condition_predictions.csv`",
            "- `metrics/testset_subset_100.csv`",
            "",
            "## Interpretation Boundary",
            "",
            "`Sess_Tactic_predict` is a model prediction from an earlier step and is intentionally not used for training "
            "or condition summaries. Plot colors and test CSV labels are model-derived CVAE condition-gate predictions.",
            "",
        ]
    )
    (run_dir / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else make_run_dir(config["output"]["base_dir"])
    for name in ("logs", "checkpoints", "scalers", "metrics", "plots", "reports", "embeddings"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_dir / "logs" / "experiment.log")
    save_config(config, run_dir / "config_resolved.yaml")
    capture_environment(run_dir / "environment.json")
    device = resolve_device(config["training"].get("device", "auto"))
    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)
    try:
        if args.stage in {"prepare", "all"}:
            run_prepare(config, device, args.force_prepare, logger)
        if args.stage in {"train", "all"}:
            run_train(config, run_dir, device, logger)
    except Exception:
        logger.exception("Experiment failed")
        raise
    logger.info("Done")
    print(run_dir)


if __name__ == "__main__":
    main()
