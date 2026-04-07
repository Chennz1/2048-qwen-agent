# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/`. Use `src/data_gen/` for raw trajectory generation, schema checks, and prompt assembly; `src/envs/` for the 2048 environment; `src/models/` for SFT, ReST, and GRPO trainers; `src/eval/` for random and fixed-set evaluation; and `src/utils/` for shared monitoring and stats helpers. Automation lives in `scripts/`. Runtime outputs should stay under `data/`, `cache/`, and `checkpoints/` rather than inside `src/`.

## Build, Test, and Development Commands
Install Python dependencies with:

```bash
pip install -r requirements.txt
```

Run the smallest end-to-end check with:

```bash
bash scripts/smoke_pipeline.sh --monitor_backend none
```

Common workflow commands:

```bash
python -m src.data_gen.processor --input_dir data/raw --output_dir data/processed --use_thinking --validate
bash scripts/train.sh --mode sft --train_data data/processed/train --val_data data/processed/val --output_dir checkpoints/sft
bash scripts/evaluate.sh --model_path checkpoints/sft --base_model Qwen/Qwen3-1.7B --output_dir data/eval/sft
```

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for functions and modules, PascalCase for classes, and explicit type hints where they add value. Keep CLI entry points under `src/...` importable with `python -m ...`. Bash scripts use uppercase config variables and long `--flag_name` arguments. There is no repo-local formatter config yet, so match surrounding code and keep comments brief and functional.

## Testing Guidelines
`pytest` is listed in `requirements.txt` and referenced by the docs:

```bash
pytest -q tests
```

If you add tests, place them in a top-level `tests/` directory with names like `test_processor.py`. Prefer focused unit tests for data contracts, move legality, and evaluator metrics. For pipeline changes, run the smoke script before opening a PR.

## Commit & Pull Request Guidelines
Recent history uses short, version-oriented commit subjects such as `v1.4.0 ...` and `v2.0.0 ...`. Keep commits concise, scoped, and descriptive; include the affected stage when useful, for example `v2.1.0 eval: tighten baseline aggregation`. PRs should state the experiment or pipeline impact, list commands run, link related issues, and attach result artifacts or screenshots when evaluation outputs change.

## Security & Configuration Tips
Do not hardcode secrets. The code already respects `HF_ENDPOINT` and `HF_HOME`; prefer environment variables for model and cache configuration. Large generated datasets, model weights, and caches should remain out of git.
