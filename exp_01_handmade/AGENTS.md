# Repository Guidelines

## Project Structure & Module Organization
Keep each experiment self-contained inside `exp_<index>_<tag>/`. Within `exp_01_handmade/`, use the following layout and mirror it when adding new experiments:
```
exp_01_handmade/
├── src/            # Core masking logic packaged as `masking_experiments`
├── tests/          # Pytest suites mirroring the src tree
├── assets/         # SplitMNIST mirror plus sample prompts/checkpoints (<10 MB each; auto-downloads)
├── configs/        # YAML configs per experiment (e.g., exp_01.yaml)
└── scripts/        # Entry points such as run_masking.py, profile.py
```
Prefer feature-focused modules under `src/experiments/<name>/` and keep shared utilities in `src/common/` to avoid circular imports.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — isolate dependencies per experiment.
- `python -m pip install -r requirements.txt` — install runtime + dev deps; ensure `torch`/`torchvision` install succeeds with the right wheel index.
- `python scripts/run_experiment.py --config configs/stage1_parallel.yaml` — launches SplitMNIST masking with the provided YAML (downloads MNIST into `assets/data` automatically; set `download: false` once cached or toggle `use_fake_data` offline).
- `python scripts/run_experiment.py --config configs/stage2_shared_columns.yaml` — sequentially trains Task 1 → Task 2 → Task 1′ → Task 2′ using the same two columns to quantify forgetting.
- `python scripts/run_experiment.py --config configs/stage2_disjoint_columns.yaml` — sequentially trains the same tasks but allocates dedicated columns for the primed variants.
- `python scripts/run_experiment.py --config configs/stage2_residual_columns.yaml` — lets Task 1′/Task 2′ act as residuals over new columns while reusing their parent columns for inference (parent weights frozen, residual columns continue learning).
- `pytest -q` — exercises the PyTorch SNN + SplitMNIST loader quickly using the synthetic fallback.

## Coding Style & Naming Conventions
Target Python 3.11+, 4-space indentation, and 100-character lines. Run `ruff check src tests` for lint + import order and `black src tests scripts` for formatting; commit only clean trees. Use descriptive module names (`masking_strategy.py`, `prompt_loader.py`) and snake_case for functions/variables. Classes use PascalCase, and config files follow `exp_<id>.yaml`.

## Testing Guidelines
Add paired tests for every feature module under `tests/<module_name>/test_<case>.py`. Use `pytest` fixtures for sample prompts and mark stochastic tests with `@pytest.mark.flaky`. Maintain >=90% coverage on `src/experiments/` (verify with `pytest --cov=src/experiments`). Record slow, data-heavy tests with the `slow` mark and keep golden files inside `tests/data/`. Test configs should enable `use_fake_data: true` to avoid leaking real MNIST into CI artifacts.

## Commit & Pull Request Guidelines
Follow `type(scope): summary` (e.g., `feat(masking): add diffusion noise schedule`). Keep summaries ≤72 chars and explain reasoning in the body. Each PR should link the tracking issue, describe config or asset changes, paste the command used for validation, and attach key plots or console excerpts under **Testing**. Request review once CI and `pytest` pass locally; drafts are preferred for work in progress.

## Security & Configuration Tips
Never commit API keys or raw datasets—store secrets in `.env` files ignored by Git. Reference them via `python-dotenv` inside `scripts/run_masking.py`. When sharing configs, redact absolute paths and ensure sample assets stay under the repository’s size budget.
