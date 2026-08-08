from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ucd_cvae_v2_1.data import (  # noqa: E402
    leakage_report, load_prepared, make_time_split, prepare_dataset, split_assignments,
)
from experiments.ucd_cvae_v2_1.evaluate import (  # noqa: E402
    evaluate_model, golden_diagnostics, prediction_frame,
)
from experiments.ucd_cvae_v2_1.geometry import (  # noqa: E402
    cosine_matrix, load_condition_geometry, save_condition_geometry,
)
from experiments.ucd_cvae_v2_1.inference import export_gate_checkpoint  # noqa: E402
from experiments.ucd_cvae_v2_1.model import UCDCVAE  # noqa: E402
from experiments.ucd_cvae_v2_1.training import train_model  # noqa: E402
from experiments.ucd_cvae_v2_1.utils import (  # noqa: E402
    capture_environment, configure_logging, load_config, make_run_dir, resolve_device,
    save_config, seed_everything, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UCD-CVAE v2.1 independent experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("prepare", "geometry", "train", "evaluate", "all"), default="all")
    parser.add_argument("--run-dir", default=None); parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--geometry-variant", choices=("all", "full_orthogonal", "common_removal_only"), default="all")
    return parser.parse_args()


def _run_dir(config: dict[str, Any], requested: str | None, require_existing: bool) -> Path:
    if requested:
        path = Path(requested).expanduser().resolve()
        if require_existing and not path.is_dir(): raise FileNotFoundError(f"Run directory not found: {path}")
    elif require_existing:
        raise ValueError("--run-dir is required for evaluate stage.")
    else: path = make_run_dir(config["output"]["base_dir"])
    for name in ("logs", "checkpoints", "metrics", "plots", "reports", "embeddings"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def _variants(config: dict[str, Any], requested: str) -> list[str]:
    return list(config.get("variants", ["full_orthogonal", "common_removal_only"])) if requested == "all" else [requested]


def _save_geometry_diagnostics(geometry, run_dir: Path) -> None:
    save_condition_geometry(geometry, run_dir / "embeddings")
    matrices = {"raw": geometry.raw_tactics,
                "common_removed": np.vstack((geometry.common, geometry.common_removed)),
                "initial": np.vstack((geometry.common, geometry.initial_tactics))}
    for name, matrix in matrices.items():
        similarities = cosine_matrix(matrix); labels = geometry.tactic_labels if name == "raw" else geometry.labels
        pd.DataFrame(similarities, index=labels, columns=labels).to_csv(
            run_dir / "metrics" / f"geometry_{geometry.variant}_{name}_cosine.csv")
        fig, ax = plt.subplots(figsize=(10, 8)); image = ax.imshow(similarities, vmin=-1, vmax=1, cmap="coolwarm")
        short = [label.split(" (")[0] for label in labels]
        ax.set_xticks(range(len(labels)), short, rotation=90, fontsize=7); ax.set_yticks(range(len(labels)), short, fontsize=7)
        fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(
            run_dir / "plots" / f"geometry_{geometry.variant}_{name}_cosine.png", dpi=150); plt.close(fig)


def _build_model(config: dict[str, Any], geometry, variant: str) -> UCDCVAE:
    model_config = deepcopy(config["model"]); model_config["geometry_variant"] = variant
    model_config["input_dim"] = geometry.dimension
    return UCDCVAE.from_config(model_config, torch.from_numpy(geometry.common),
                               torch.from_numpy(geometry.initial_tactics))


def _plot_history(path: Path, output: Path, variant: str) -> None:
    frame = pd.read_csv(path); fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frame["epoch"], frame["train_loss"], label="train"); ax.plot(frame["epoch"], frame["val_loss"], label="validation")
    ax.axvline(5, color="gray", linestyle="--"); ax.axvline(15, color="gray", linestyle="--")
    ax.set(title=f"UCD-CVAE {variant}", xlabel="epoch", ylabel="loss"); ax.legend(); fig.tight_layout()
    fig.savefig(output, dpi=150); plt.close(fig)


def train_variants(config: dict[str, Any], run_dir: Path, device: torch.device,
                   variants: list[str], logger) -> dict[str, Any]:
    x, metadata = load_prepared(config["data"]["prepared_dir"]); split = make_time_split(metadata, config["data"]["split"])
    split_assignments(metadata, split).to_csv(run_dir / "metrics" / "split_assignments.csv", index=False)
    write_json(leakage_report(metadata, split), run_dir / "metrics" / "leakage_report.json")
    summaries = {}
    for variant in variants:
        geometry = load_condition_geometry(config["conditions"], device, variant,
                                           float(config["model"].get("geometry_epsilon", 1e-6)))
        _save_geometry_diagnostics(geometry, run_dir); model = _build_model(config, geometry, variant)
        checkpoint = run_dir / "checkpoints" / f"ucd_cvae_{variant}.pt"
        result = train_model(model, x, split, config["training"], config["loss"], device,
                             checkpoint, int(config.get("seed", 42)), logger)
        export_gate_checkpoint(model, run_dir / "checkpoints" / f"gate_only_{variant}.pt",
                               geometry.labels, config["data"]["embedder"],
                               str(config["data"].get("payload_parser", "auto")), config["evaluation"])
        projected = model.projected_basis().detach().cpu().numpy()
        np.savez_compressed(run_dir / "embeddings" / f"projected_basis_{variant}.npz",
                            labels=np.asarray(geometry.labels, dtype=str), matrix=projected)
        _plot_history(run_dir / "metrics" / f"training_history_{variant}.csv",
                      run_dir / "plots" / f"training_history_{variant}.png", variant)
        summaries[variant] = {"best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss}
    write_json(summaries, run_dir / "metrics" / "training_summary.json")
    return summaries


def evaluate_variants(config: dict[str, Any], run_dir: Path, device: torch.device,
                      variants: list[str]) -> dict[str, Any]:
    x, metadata = load_prepared(config["data"]["prepared_dir"]); split = make_time_split(metadata, config["data"]["split"])
    summaries = {}
    for variant in variants:
        geometry = load_condition_geometry(config["conditions"], device, variant,
                                           float(config["model"].get("geometry_epsilon", 1e-6)))
        model = _build_model(config, geometry, variant); checkpoint_path = run_dir / "checkpoints" / f"ucd_cvae_{variant}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        metrics, gates = evaluate_model(model, x, split.test, device, int(config["training"]["batch_size"]),
                                        int(config.get("seed", 42)))
        predictions = prediction_frame(metadata, split.test, gates, geometry.labels)
        predictions.to_csv(run_dir / "metrics" / f"test_predictions_{variant}.csv", index=False)
        gold = golden_diagnostics(metadata, split.test, gates, geometry.labels, config["evaluation"],
                                  run_dir / "metrics" / f"golden_predictions_{variant}.csv")
        metrics["golden_diagnostics"] = gold; summaries[variant] = metrics
        write_json(metrics, run_dir / "metrics" / f"evaluation_{variant}.json")
    write_json(summaries, run_dir / "metrics" / "experiment_summary.json")
    return summaries


def write_report(run_dir: Path, summaries: dict[str, Any]) -> None:
    lines = ["# UCD-CVAE Version 2.1 Report", "", "All training is unsupervised. Golden labels are post-training diagnostics only.",
             "Gate values are uncalibrated evidence scores, not probabilities or malicious percentages.", ""]
    for variant, metrics in summaries.items():
        lines.extend([f"## {variant}", "", f"- Full reconstruction cosine: {metrics.get('full_cosine', 'n/a')}",
                      f"- Condition gain cosine: {metrics.get('condition_gain_cosine', 'n/a')}",
                      f"- Basis orthogonality error: {metrics.get('basis_orthogonality_error', 'n/a')}",
                      f"- Maximum residual projection: {metrics.get('max_residual_projection', 'n/a')}", ""])
    lines.extend(["## Latency semantics", "", "Encoder_Gate has fixed dimensional cost after embedding. ModernBERT and end-to-end cost still depend on payload token length."])
    (run_dir / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args(); config = load_config(args.config, PROJECT_ROOT); seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config["training"].get("device", "auto")); variants = _variants(config, args.geometry_variant)
    run_dir = _run_dir(config, args.run_dir, args.stage == "evaluate"); logger = configure_logging(run_dir / "logs" / "experiment.log")
    save_config(config, run_dir / "config_resolved.yaml"); capture_environment(run_dir / "environment.json")
    logger.info("run_dir=%s device=%s variants=%s", run_dir, device, variants)
    if args.stage in {"prepare", "all"}:
        prepared = prepare_dataset(config, PROJECT_ROOT, device, args.force_prepare)
        logger.info("prepared=%s reused=%s", prepared.prepared_dir, prepared.reused)
    if args.stage in {"geometry", "all"}:
        for variant in variants:
            geometry = load_condition_geometry(config["conditions"], device, variant,
                                               float(config["model"].get("geometry_epsilon", 1e-6)))
            _save_geometry_diagnostics(geometry, run_dir)
    if args.stage in {"train", "all"}: train_variants(config, run_dir, device, variants, logger)
    summaries = evaluate_variants(config, run_dir, device, variants) if args.stage in {"evaluate", "all"} else {}
    if summaries: write_report(run_dir, summaries)
    print(run_dir)


if __name__ == "__main__": main()
