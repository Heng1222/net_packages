from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
else:  # pragma: no cover
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from experiments.ae_cvae_tactic.pipeline import ExperimentRunner
from experiments.ae_cvae_tactic.utils.config import load_config
from experiments.ae_cvae_tactic.utils.io import capture_environment, make_run_dir
from experiments.ae_cvae_tactic.utils.logging import configure_logging
from experiments.ae_cvae_tactic.utils.seed import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AE/CVAE tactic latent-space experiment runner")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--run", default="all", choices=("ae", "cvae", "compatibility", "ablation", "all"),
        help="Experiment stage to run",
    )
    parser.add_argument(
        "--run-dir", default=None,
        help="Reuse an existing output directory and compatible checkpoints instead of creating a new run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
        for name in ("logs", "checkpoints", "scalers", "latent", "metrics", "plots", "reports", "embeddings"):
            (run_dir / name).mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(config["output"]["base_dir"])
    logger = configure_logging(run_dir / "logs" / "experiment.log")
    capture_environment(run_dir / "environment.json")
    device = resolve_device(config["training"].get("device", "auto"))
    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)
    try:
        runner = ExperimentRunner(config, run_dir, device, reuse_checkpoints=bool(args.run_dir))
        if args.run in {"ae", "all"}:
            runner.run_ae()
        if args.run in {"cvae", "all"}:
            runner.run_cvae()
        if args.run == "compatibility" or (
            args.run == "all" and config.get("evaluation", {}).get("run_compatibility_test", True)
        ):
            runner.run_compatibility()
        if args.run in {"ablation", "all"}:
            runner.run_ablation()
        runner.finalize_report()
    except Exception:
        logger.exception("Experiment failed")
        raise
    logger.info("Experiment complete: %s", run_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
