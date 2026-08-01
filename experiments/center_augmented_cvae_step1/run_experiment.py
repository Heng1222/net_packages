from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from experiments.center_augmented_cvae_step1.conditions import (  # noqa: E402
    ConditionSet, cosine_matrix, load_condition_set, save_condition_set,
)
from experiments.center_augmented_cvae_step1.data import (  # noqa: E402
    leakage_report, load_prepared, make_time_split, prepare_dataset,
    split_assignments, split_label_counts,
)
from experiments.center_augmented_cvae_step1.evaluate import (  # noqa: E402
    evaluate_main_model, gate_summary, golden_alignment, prediction_frame,
    random_condition_matrices, reconstruction_metrics,
)
from experiments.center_augmented_cvae_step1.model import CenterAugmentedCVAE, PlainVAE  # noqa: E402
from experiments.center_augmented_cvae_step1.training import (  # noqa: E402
    evaluate_plain_vae, train_model,
)
from experiments.center_augmented_cvae_step1.utils import (  # noqa: E402
    capture_environment, configure_logging, load_config, make_run_dir, resolve_device,
    save_config, seed_everything, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsupervised centroid-augmented Step1 CVAE")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate", "all"), default="all")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args()


def _prepare_run_dir(config: dict, requested: str | None) -> Path:
    run_dir = Path(requested).expanduser().resolve() if requested else make_run_dir(config["output"]["base_dir"])
    for name in ("logs", "checkpoints", "metrics", "plots", "reports", "embeddings"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_condition_diagnostics(conditions: ConditionSet, run_dir: Path) -> None:
    save_condition_set(conditions, run_dir / "embeddings")
    for name, matrix in (("raw_tactics", conditions.raw_tactics),
                         ("model_gate", conditions.gate_matrix)):
        labels = conditions.tactic_labels if name == "raw_tactics" else conditions.labels
        similarities = cosine_matrix(matrix)
        pd.DataFrame(similarities, index=labels, columns=labels).to_csv(
            run_dir / "metrics" / f"condition_{name}_cosine_similarity.csv"
        )
        fig, ax = plt.subplots(figsize=(10, 8)); image = ax.imshow(similarities, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(labels)), [label.split(" (")[0] for label in labels], rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), [label.split(" (")[0] for label in labels], fontsize=7)
        fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(
            run_dir / "plots" / f"condition_{name}_cosine_similarity.png", dpi=160
        ); plt.close(fig)


def _plot_history(history: list[dict[str, float]], path: Path, title: str) -> None:
    frame = pd.DataFrame(history); fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame["epoch"], frame["train_loss"], label="train")
    ax.plot(frame["epoch"], frame["val_loss"], label="validation")
    ax.set(title=title, xlabel="epoch", ylabel="loss"); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=160); plt.close(fig)


def _model_config(config: dict, conditions: ConditionSet) -> dict:
    result = dict(config["model"]); result["condition_count"] = len(conditions.labels)
    result["condition_dim"] = conditions.dimension
    if int(result["input_dim"]) != conditions.dimension:
        raise ValueError("Payload and condition embedding dimensions differ.")
    return result


def _split_names(row_count: int, split) -> np.ndarray:
    names = np.full(row_count, "", dtype=object)
    names[split.train] = "train"; names[split.val] = "val"; names[split.test] = "test"
    return names


def _evaluate_main(model: CenterAugmentedCVAE, name: str, x: np.ndarray, metadata: pd.DataFrame,
                   split, conditions: ConditionSet, decode: np.ndarray, gate: np.ndarray,
                   config: dict, run_dir: Path, device: torch.device) -> dict:
    metrics, ablations, gates = evaluate_main_model(
        model, x, split.test, decode, gate, device, int(config["training"]["batch_size"]),
        int(config.get("seed", 42)),
    )
    ablations["condition"] = conditions.labels[1:]
    write_json(metrics, run_dir / "metrics" / f"{name}_ablation_summary.json")
    ablations.to_csv(run_dir / "metrics" / f"{name}_per_tactic_ablation.csv", index=False)
    threshold = float(config["evaluation"].get("condition_threshold", 0.5))
    gate_summary(gates, conditions.labels, threshold).to_csv(
        run_dir / "metrics" / f"{name}_condition_gate_summary.csv", index=False
    )
    if name == "main_cvae":
        prediction_frame(metadata, split.test, gates, conditions.labels, threshold).to_csv(
            run_dir / "metrics" / "testset_condition_predictions.csv", index=False
        )
    return metrics


def train_and_evaluate(config: dict, run_dir: Path, device: torch.device,
                       logger: logging.Logger, train: bool = True) -> None:
    x, metadata = load_prepared(config["data"]["prepared_dir"])
    split = make_time_split(metadata, config["data"]["split"])
    split_assignments(metadata, split).to_csv(run_dir / "metrics" / "split_assignments.csv", index=False)
    split_label_counts(metadata, split).to_csv(run_dir / "metrics" / "split_label_counts.csv", index=False)
    write_json(leakage_report(metadata, split), run_dir / "metrics" / "leakage_report.json")
    conditions = load_condition_set(config["conditions"], device); _save_condition_diagnostics(conditions, run_dir)
    model_config = _model_config(config, conditions); seed = int(config.get("seed", 42))
    summaries: dict[str, dict] = {}

    main = CenterAugmentedCVAE.from_config(model_config)
    main_checkpoint = run_dir / "checkpoints" / "main_cvae.pt"
    if train:
        logger.info("Training main CVAE without labels")
        result = train_model(main, x, split, config["training"], config["loss"], device,
                             main_checkpoint, seed, logger, conditions.decode_matrix, conditions.gate_matrix)
        summaries["main_training"] = {"best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss}
        _plot_history(result.history, run_dir / "plots" / "main_cvae_training.png", "Main CVAE")
    else:
        main.load_state_dict(torch.load(main_checkpoint, map_location=device, weights_only=False)["model_state"])
    summaries["main_cvae"] = _evaluate_main(main, "main_cvae", x, metadata, split, conditions,
                                             conditions.decode_matrix, conditions.gate_matrix,
                                             config, run_dir, device)

    if bool(config.get("baselines", {}).get("train_plain_vae", True)):
        plain = PlainVAE.from_config(model_config); checkpoint = run_dir / "checkpoints" / "plain_vae.pt"
        if train:
            logger.info("Training plain VAE baseline")
            result = train_model(plain, x, split, config["training"], config["loss"], device,
                                 checkpoint, seed, logger)
            summaries["plain_training"] = {"best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss}
            _plot_history(result.history, run_dir / "plots" / "plain_vae_training.png", "Plain VAE")
        else:
            plain.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state"])
        summaries["plain_vae"] = evaluate_plain_vae(
            plain, x, split.test, device, int(config["training"]["batch_size"])
        )

    if bool(config.get("baselines", {}).get("train_random_condition_cvae", True)):
        random_decode, random_gate = random_condition_matrices(conditions.decode_matrix, seed)
        random_model = CenterAugmentedCVAE.from_config(model_config)
        checkpoint = run_dir / "checkpoints" / "random_condition_cvae.pt"
        if train:
            logger.info("Training random-condition CVAE baseline")
            result = train_model(random_model, x, split, config["training"], config["loss"], device,
                                 checkpoint, seed, logger, random_decode, random_gate)
            summaries["random_training"] = {"best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss}
            _plot_history(result.history, run_dir / "plots" / "random_condition_training.png", "Random-condition CVAE")
        else:
            random_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state"])
        summaries["random_condition_cvae"] = _evaluate_main(
            random_model, "random_condition_cvae", x, metadata, split, conditions,
            random_decode, random_gate, config, run_dir, device,
        )

    alignment = golden_alignment(metadata, x, _split_names(len(metadata), split), conditions.gate_matrix,
                                 conditions.labels, config["evaluation"], run_dir / "metrics",
                                 float(config["model"]["gate_temperature"]))
    summaries["golden_alignment"] = alignment
    write_json(summaries, run_dir / "metrics" / "experiment_summary.json")
    write_report(run_dir, summaries, conditions)


def write_report(run_dir: Path, summaries: dict, conditions: ConditionSet) -> None:
    main = summaries.get("main_cvae", {}); gold = summaries.get("golden_alignment", {})
    lines = ["# Center-Augmented CVAE Report", "", "Training is fully unsupervised; no label enters the model or loss.", "",
             "## Condition geometry", "", f"- Conditions: {len(conditions.labels)} (one centroid plus 13 centered tactics)",
             f"- Recomposition error: {conditions.metadata['recomposition_error']:.8g}", "", "## Main reconstruction", ""]
    if main:
        lines.extend([f"- Full MSE: {main['full']['recon_mse']:.8g}",
                      f"- Condition gain: {main['condition_gain']:.8g}",
                      f"- Common-condition gain: {main['common_condition_gain']:.8g}",
                      f"- Tactic-condition gain: {main['tactic_condition_gain']:.8g}"])
    lines.extend(["", "## Step2 diagnostic", "",
                  "This is a transductive alignment diagnostic, not supervised training or a held-out generalization claim.",
                  f"- Matched rows: {gold.get('matched_rows', 0)}",
                  f"- Macro F1: {gold.get('macro_f1', 'n/a')}"])
    (run_dir / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args(); config = load_config(args.config, PROJECT_ROOT)
    seed_everything(int(config.get("seed", 42))); device = resolve_device(config["training"].get("device", "auto"))
    run_dir = _prepare_run_dir(config, args.run_dir); logger = configure_logging(run_dir / "logs" / "experiment.log")
    save_config(config, run_dir / "config_resolved.yaml"); capture_environment(run_dir / "environment.json")
    logger.info("Run directory: %s", run_dir); logger.info("Device: %s", device)
    if args.stage in {"prepare", "all"}:
        prepared = prepare_dataset(config, PROJECT_ROOT, device, force=args.force_prepare)
        logger.info("Prepared dataset: %s reused=%s", prepared.prepared_dir, prepared.reused)
        conditions = load_condition_set(config["conditions"], device); _save_condition_diagnostics(conditions, run_dir)
    if args.stage in {"train", "all"}: train_and_evaluate(config, run_dir, device, logger, train=True)
    if args.stage == "evaluate": train_and_evaluate(config, run_dir, device, logger, train=False)
    print(run_dir)


if __name__ == "__main__": main()
