# Repository Guidelines

## Operating Environment and Tooling

This project targets Linux. On Windows, work inside WSL Bash and run every Git command—including status checks, commits, branch operations, and pushes—from WSL, never Windows Git. Conda owns the Python interpreter and native runtime; Poetry owns project packages and the lock file. Do not use `pip`, `venv`, `uv`, or create a repository `.venv`.

```bash
conda env create -f environment.yml
conda activate manga
poetry install --with dev
```

For an existing environment, use `conda env update -f environment.yml --prune`, reactivate it, then run `poetry install --with dev`. Treat the Conda + Poetry policy as authoritative if an older branch contains conflicting setup text.

## Project Structure & Module Organization

Application code lives in `src/manga_translator/`. The main pipeline is in `pipeline.py`; detection, OCR, translation, inpainting, and typesetting have dedicated modules. `src/manga_translator/ctd/` contains vendored detector code. Tests in `tests/test_*.py` mirror source modules. `models/` and `fonts/` hold runtime assets; `samples/` and `validation_samples/` contain regression evidence. Treat `input/`, `output/`, `build/`, and `dist/` as generated or local working data.

## Build, Test, and Development Commands

Run commands after `conda activate manga`:

```bash
poetry run manga-translate doctor --config config.yaml --strict-api-key
poetry run manga-translate run --config config.yaml
poetry run pytest -q
poetry run ruff check .
poetry build
```

Use `manga-translate test --image input/page.jpg --dump-json` for a focused page regression and `detect-only` when translation is unnecessary.

## Coding Style & Naming Conventions

Target Python 3.11, use four-space indentation, type hints, and a 100-character line limit. Follow Ruff defaults. Name functions and modules `snake_case`, classes `PascalCase`, and constants `UPPER_SNAKE_CASE`. Preserve fail-safe behavior: uncertain OCR, translation, or layout must leave source pixels intact.

## Testing Guidelines

Use pytest. Name files `test_<module>.py` and tests `test_<behavior>`. Run targeted tests first, then the full suite. Changes affecting rendered output require representative evidence from `validation_samples/`. Never run paid API integration tests without explicit approval.

## Commit & Pull Request Guidelines

Use focused Conventional Commit messages such as `fix(typesetting): preserve vertical spacing` or `test(pipeline): cover API failure`. Every completed change must be committed; do not leave requested work only in the working tree. PRs should explain the problem, approach, tests run, configuration impact, and include before/after images for visual changes. Never commit API keys, private manga pages, model caches, or generated output.
