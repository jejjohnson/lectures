# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A general **lectures** repo built with Quarto: a deep reference site under `topics/` and curated reveal.js slide decks under `courses/`, meant to hold many courses over time. Keep the top-level framing (home page, README, site title) **subject-agnostic** — individual courses are subject-specific. The first/current course is *Probabilistic ML for Finance* (dependence, Gaussianization, uncertainty, fair learning); new courses get their own `courses/<id>/` folder. Retains the research-template bones (pixi/uv, DVC, Hydra) for heavy demos. The template supports two environment managers: [uv](https://github.com/astral-sh/uv) (via `Makefile`) and [pixi](https://pixi.sh) (via `pixi.toml`).

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Run tests: uv run pytest -v -o addopts=
make test-cov             # Tests with coverage
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check src/lectures
make precommit            # Run pre-commit on all files
make docs-serve           # Local Quarto preview (site + slides)
make docs                 # Render site + slides -> _site/
```

### Running a single test

```bash
uv run pytest tests/test_example.py::test_case -v
```

### Alternative pixi tasks

```bash
pixi run test
pixi run lint
pixi run typecheck
pixi run -e docs preview
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -v
uv run --group lint ruff check .
uv run --group lint ruff format --check .
uv run --group typecheck ty check src/lectures
```

**Critical**: Always lint the entire repo with `.` from the root. The template includes tests, configs, scripts, and docs glue outside the package directory.

## Architecture

### Package structure

The installable package lives in [src/lectures](src/lectures/).

### Key directories

| Path | Purpose |
|------|---------|
| `src/lectures/` | Installable library and public exports |
| `src/lectures/data/` | Data loading utilities |
| `src/lectures/models/` | Model implementations |
| `src/lectures/trainers/` | Training loops |
| `src/lectures/utils/` | Utility functions |
| `configs/` | Hydra configuration hierarchy |
| `data/` | DVC-managed data directories |
| `results/` | DVC-managed experiment results |
| `scripts/` | Data-generation scripts (download_market_data, make_credit_data) |
| `topics/` | Deep evergreen lecture content (Quarto) |
| `courses/` | Curated courses: slides + demos |
| `styles/` | Site + slide SCSS themes |
| `notebooks/` | Jupytext percent-format `.py` notebooks |
| `marimo_notebooks/` | Marimo reactive notebooks |
| `tests/` | Test suite |

## Content authoring

Two layers (see README): deep evergreen material in `topics/` (Quarto `.qmd` with
executable Python cells that generate figures at render time), and curated reveal.js
decks in `courses/<id>/slides/` that reuse the same figures/helpers.

- Author deep material + figures **once** in `topics/`; courses reference them.
- Figure styling + toy datasets live in `src/lectures/` (`plotting.py`, `datasets.py`)
  so notes and slides look identical. Import via `from lectures import plotting, datasets`.
- Quarto caches executed cells under `_freeze/` (committed) so CI re-renders only what changed.
- Slides override format per-file with `format: revealjs` in the front matter.
- Demos in `courses/*/demos/` are Jupytext percent-format `.py`; they `import` the
  library repos (rbig, gauss_flows, pyrox, keras-fairkl) from the opt-in `demos` env.

## Coding Conventions

- `from __future__ import annotations` at the top of Python modules
- Type hints on public functions and methods
- Use `pathlib.Path` for filesystem work
- Keep scientific computations pure; isolate IO and CLI side effects
- Match existing numerical style and avoid refactoring unrelated code

## Plans

Plans and scratch implementation docs go in `.plans/` and should not be committed.

## PR Review Comments

When addressing PR review comments, resolve each review thread after fixing it via the GitHub GraphQL API. Use the workflow documented in [AGENTS.md](AGENTS.md).

## Code Review

Follow the guidance in [CODE_REVIEW.md](CODE_REVIEW.md) for all code review tasks.
