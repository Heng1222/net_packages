# Repository Guidelines

## Project Structure & Module Organization

This repository contains Python 3.12 data preparation scripts and AE/CVAE experiment code for network payload representation work. Root scripts handle dataset preparation: `Step1_create_dataset.py`, `Step1_clean_rawdata.py`, and `Step2_create_golden_review.py`. Experiment packages live under `experiments/`: `ae_cvae_tactic/` for AE, CVAE, and contrastive tactic experiments, and `disentangled_cvae_step1/` for the Step1 disentangled CVAE pipeline. Each experiment keeps its own `configs/`, `conditions/`, modules, and `tests/`. Local data belongs under `Year=2022/`; generated run artifacts belong under `outputs/`.

## Build, Test, and Development Commands

Use `uv` from the repository root.

- `uv sync`: create or update the Python environment from `pyproject.toml` and `uv.lock`.
- `uv sync --extra parquet` or `uv sync --extra notebook`: install optional parquet or notebook dependencies.
- `uv run python Step1_create_dataset.py --folder Year=2022`: build and clean the Step1 raw dataset from local pickle inputs.
- `uv run python Step2_create_golden_review.py`: create the golden review sample.
- `uv run python experiments\ae_cvae_tactic\run_experiment.py --config experiments\ae_cvae_tactic\configs\default.yaml --run all`: run the AE/CVAE tactic pipeline.
- `uv run python experiments\disentangled_cvae_step1\run_experiment.py --config experiments\disentangled_cvae_step1\configs\default.yaml --stage all`: run the Step1 disentangled pipeline.

## Coding Style & Naming Conventions

Follow existing Python style: 4-space indentation, snake_case functions and variables, PascalCase classes, and modules grouped by responsibility (`data`, `models`, `training`, `evaluation`, `utils`). Prefer typed, explicit interfaces where surrounding code already uses them. Keep configuration in YAML files under each experiment's `configs/`; avoid hardcoding dataset paths or hyperparameters in model code.

## Testing Guidelines

Tests use Python `unittest` and live in each experiment's `tests/` directory. Name files `test_*.py` and keep lightweight fixtures inside the test package. Run:

```powershell
uv run python -m unittest discover -s experiments\ae_cvae_tactic\tests -v
uv run python -m unittest discover -s experiments\disentangled_cvae_step1\tests -v
```

Add or update tests when changing data adapters, model shapes, training losses, metrics, or CLI behavior.

## Commit & Pull Request Guidelines

Recent history uses short conventional-style subjects such as `feat: predict category`, `perf: change parameter setting`, and `chore: uv env`. Keep commits focused and use the same prefix style. Pull requests should describe the experiment affected, list commands run, note required local data files, and include screenshots or report paths when visualization outputs change.

## Security & Configuration Tips

Do not commit raw packet data, curated CSVs, model checkpoints, embedding caches, or generated outputs. Keep local inputs under `Year=2022/` and run outputs under `outputs/`. The first embedding run may download Hugging Face model weights into the normal local cache.
