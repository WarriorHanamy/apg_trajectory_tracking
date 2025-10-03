# Repository Guidelines

## Project Structure & Module Organization
Source code for controllers and dynamics lives in `neural_control/`, organized by submodules such as `controllers/`, `dynamics/`, and `environments/`. Experiment entry points reside in `scripts/` (e.g., `train_drone.py`, `evaluate_cartpole.py`). Configuration JSON files are under `configs/`, reusable assets in `assets/`, pretrained weights within `trained_models/`, and Python-based regression tests in `tests/`.

## Build, Test & Development Commands
Create an isolated environment before installing: `python -m venv .venv && source .venv/bin/activate`. Install dependencies editable for local iteration with `pip install -e .`. Generate quadrotor trajectories via `python scripts/generate_trajectories.py`, then launch training runs such as `python scripts/train_drone.py` or `python scripts/train_fixed_wing.py`. Evaluate trained policies with `python scripts/evaluate_drone.py -a 50` (use `--animate` for visualizations). Run the automated checks using `pytest tests` and add `-k` to target individual suites.

## Coding Style & Naming Conventions
Python files use PEP 8 defaults: 4-space indentation, lower_snake_case modules, and CamelCase classes. Keep controller names descriptive (`QuadLstmController`, `FixedWingMpc`). When touching configs, match existing key casing and numeric precision. Lint with `ruff` (configured via `pyproject.toml`) and run `ruff check .` before opening a pull request. Prefer type hints where feasible and keep plotting scripts notebook-friendly.

## Testing Guidelines
Unit tests follow the `test_*.py` pattern described in `pyproject.toml`. Place new coverage alongside related modules inside `tests/`. Favor deterministic seeds for stochastic simulations so results are reproducible in CI. Use `pytest --maxfail=1 --disable-warnings -q` for quick validation, and document any long-running simulations in the PR description.

## Commit & Pull Request Guidelines
Commit messages are imperative and concise, often referencing issues (`fix #7 by adapting …`). Compose standalone commits for logical changes and include brief context in the body when touching dynamics or controllers. Pull requests should summarize the intent, call out affected scripts/configs, list validation commands, and add before/after plots when behavior shifts. Request review once linting and tests pass locally and link to tracked issues or experiments.

## Configuration & Artifacts Tips
Tune training horizons, controller modes, and curriculum settings through the JSON files in `configs/`; keep backups of any new config variants. Store large experiment outputs under `trained_models/` with clear timestamps or scenario names. Check `assets/` before duplicating figures, and update `training_details.pdf` references if your workflow diverges from the documented baseline.
