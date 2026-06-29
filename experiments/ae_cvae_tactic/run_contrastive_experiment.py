from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
else:  # pragma: no cover
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from experiments.ae_cvae_tactic.contrastive_pipeline import ContrastiveExperimentRunner
from experiments.ae_cvae_tactic.utils.config import load_config
from experiments.ae_cvae_tactic.utils.io import capture_environment, make_run_dir
from experiments.ae_cvae_tactic.utils.logging import configure_logging
from experiments.ae_cvae_tactic.utils.seed import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Opt-in payload/description contrastive CVAE experiment"
    )
    parser.add_argument("--config", required=True, help="Contrastive experiment YAML config")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Reuse a contrastive output directory and compatible checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
        for name in (
            "logs",
            "checkpoints",
            "scalers",
            "latent",
            "metrics",
            "plots",
            "reports",
            "embeddings",
        ):
            (run_dir / name).mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(config["output"]["base_dir"])

    logger = configure_logging(run_dir / "logs" / "contrastive_experiment.log")
    capture_environment(run_dir / "environment.json")
    device = resolve_device(config["training"].get("device", "auto"))
    logger.info("Contrastive run directory: %s", run_dir)
    logger.info("Device: %s", device)
    try:
        runner = ContrastiveExperimentRunner(
            config, run_dir, device, reuse_checkpoints=bool(args.run_dir)
        )
        runner.run_contrastive()
        runner.finalize_contrastive_report()
    except Exception:
        logger.exception("Contrastive experiment failed")
        raise
    logger.info("Contrastive experiment complete: %s", run_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
