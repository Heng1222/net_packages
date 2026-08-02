from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
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
from experiments.disentangled_cvae_uwf_zeekdata24.controls import (  # noqa: E402
    CONTROL_NAMES,
    aggregate_control_decision,
    build_condition_control_matrices,
    load_control_result,
    paired_control_comparison,
    run_control_variant,
)
from experiments.disentangled_cvae_uwf_zeekdata24.data import TACTIC_LABELS, load_prepared  # noqa: E402
from experiments.disentangled_cvae_uwf_zeekdata24.evaluate import observed_tactics  # noqa: E402
from experiments.disentangled_cvae_uwf_zeekdata24.run_experiment import _standardize  # noqa: E402
from experiments.disentangled_cvae_uwf_zeekdata24.utils import (  # noqa: E402
    capture_environment,
    load_config,
    make_run_dir,
    resolve_device,
    save_config,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UWF ATT&CK semantic/random condition controls")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--variants", nargs="+", choices=CONTROL_NAMES, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("disentangled_cvae_uwf_controls")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _directories(root: Path) -> None:
    for name in ("logs", "scalers", "metrics", "reports", "embeddings", "controls"):
        (root / name).mkdir(parents=True, exist_ok=True)


def _variant_result(
    variant: str,
    seed: int,
    directory: Path,
    resume: bool,
    **run_kwargs,
):
    summary_path = directory / "metrics" / "summary.json"
    arrays_path = directory / "metrics" / "control_result.npz"
    if resume and summary_path.is_file() and arrays_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return load_control_result(directory, summary)
    return run_control_variant(variant, seed, directory=directory, **run_kwargs)


def _write_report(path: Path, decision: dict, table: pd.DataFrame) -> None:
    lines = [
        "# UWF ATT&CK Semantic Geometry Controls",
        "",
        f"- Semantic geometry supported: **{decision['semantic_geometry_supported']}**",
        f"- Seeds: {decision['seeds']}",
        f"- Required supported-seed fraction: {decision['minimum_seed_fraction']}",
        "",
        "The semantic model must beat matched-norm Gaussian, matched-norm orthogonal, and train/validation "
        "label-shuffle controls. Test labels are never shuffled.",
        "",
        "## Comparator decisions",
        "",
    ]
    for name, item in decision["comparators"].items():
        lines.append(
            f"- {name}: passes={item['passes']}, supported seeds="
            f"{item['supported_seeds']}/{item['total_seeds']}, mean deltas={item['mean_deltas']}"
        )
    lines.extend(["", "## Per-run metrics", "", "```", table.to_string(index=False), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_suite(
    config: dict,
    run_dir: Path,
    device: torch.device,
    seeds: list[int],
    variants: list[str],
    resume: bool,
    logger: logging.Logger,
) -> dict:
    if "semantic" not in variants:
        raise ValueError("The control suite must include the semantic variant.")
    comparators = [name for name in variants if name != "semantic"]
    if not comparators:
        raise ValueError("At least one non-semantic control is required.")
    x_raw, metadata, targets, split = load_prepared(config["data"]["prepared_dir"])
    x = _standardize(config, x_raw, split, run_dir)
    conditions = load_condition_embeddings(config["conditions"], None, device, run_dir / "embeddings")
    if tuple(conditions.tactic_labels) != TACTIC_LABELS or len(conditions.labels) != 14:
        raise ValueError("Controls require the canonical 13 tactics plus common 14th condition.")
    np.savez_compressed(
        run_dir / "embeddings" / "semantic_conditions.npz",
        labels=np.asarray(conditions.labels, dtype=str),
        matrix=conditions.matrix,
        raw_matrix=conditions.raw_matrix,
    )
    rows = []
    comparisons = []
    control_config = config["controls"]
    repeats = int(control_config["bootstrap_repeats"])
    tactic_truth = targets[split.test]
    tactic_strata = metadata.iloc[split.test]["technique"].to_numpy(dtype=str)
    probe_mask = metadata.iloc[split.test]["probe_eligible"].astype(str).str.casefold().eq("true").to_numpy()
    probe_strata = tactic_strata[probe_mask]
    observed = observed_tactics(targets[split.train])
    for seed in seeds:
        matrices = build_condition_control_matrices(conditions.matrix, seed)
        results = {}
        for variant in variants:
            directory = run_dir / "controls" / f"seed_{seed}" / variant
            logger.info("Starting control seed=%d variant=%s", seed, variant)
            result = _variant_result(
                variant,
                seed,
                directory,
                resume,
                x=x,
                metadata=metadata,
                true_targets=targets,
                split=split,
                condition_matrix=matrices[variant],
                config=config,
                device=device,
                logger=logger,
            )
            results[variant] = result
            rows.append(
                {
                    "seed": seed,
                    "control": variant,
                    "tactic_macro_f1": result.summary["tactic"]["macro_f1"],
                    "tactic_macro_auprc": result.summary["tactic"]["macro_auprc"],
                    "probe_c_macro_f1": result.summary["probes"]["c"]["test_macro_f1"],
                    "probe_gates_macro_f1": result.summary["probes"]["gates"]["test_macro_f1"],
                    "probe_h_macro_f1": result.summary["probes"]["h"]["test_macro_f1"],
                    "full_mse": result.summary["reconstruction"]["full_mse"],
                    "c_only_mse": result.summary["reconstruction"]["c_only_mse"],
                    "h_only_mse": result.summary["reconstruction"]["h_only_mse"],
                }
            )
        for index, comparator in enumerate(comparators):
            comparison = paired_control_comparison(
                results["semantic"],
                results[comparator],
                tactic_truth,
                observed,
                tactic_strata,
                probe_strata,
                repeats,
                seed + 50_021 + index,
            )
            comparisons.append(comparison)
            write_json(
                comparison,
                run_dir / "metrics" / f"comparison_seed_{seed}__semantic_vs_{comparator}.json",
            )
            logger.info("Comparison seed=%d control=%s supported=%s", seed, comparator, comparison["supported"])
    table = pd.DataFrame(rows).sort_values(["seed", "control"], kind="stable")
    table.to_csv(run_dir / "metrics" / "control_metrics.csv", index=False)
    decision = aggregate_control_decision(
        comparisons,
        comparators,
        seeds,
        float(control_config.get("minimum_seed_fraction", 0.8)),
    )
    write_json(decision, run_dir / "metrics" / "semantic_geometry_decision.json")
    _write_report(run_dir / "reports" / "semantic_geometry_controls.md", decision, table)
    return decision


def main() -> None:
    args = parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    config = deepcopy(config)
    controls = config.setdefault("controls", {})
    seeds = list(map(int, args.seeds or controls.get("seeds", [config["seed"]])))
    variants = list(map(str, args.variants or controls.get("variants", CONTROL_NAMES)))
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = int(args.max_epochs)
        config["training"]["early_stopping_patience"] = min(
            int(config["training"]["early_stopping_patience"]), int(args.max_epochs)
        )
    if args.bootstrap_repeats is not None:
        controls["bootstrap_repeats"] = int(args.bootstrap_repeats)
    base_dir = Path(config["output"]["base_dir"]) / "semantic_controls"
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else make_run_dir(base_dir)
    _directories(run_dir)
    logger = configure_logging(run_dir / "logs" / "controls.log")
    try:
        save_config(config, run_dir / "config_resolved.yaml")
        capture_environment(run_dir / "environment.json")
        device = resolve_device(config["training"].get("device", "auto"))
        decision = run_suite(config, run_dir, device, seeds, variants, args.resume, logger)
        logger.info("Semantic geometry supported=%s", decision["semantic_geometry_supported"])
    finally:
        for handler in tuple(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
    print(run_dir)


if __name__ == "__main__":
    main()
