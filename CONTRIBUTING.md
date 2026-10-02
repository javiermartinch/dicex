# Contributing to dicex

Thank you for considering a contribution! This guide will help you get started.

## Prerequisites

- **Python ≥ 3.12**
- **[uv](https://docs.astral.sh/uv/)**, the Python package manager used by the project
- **A [Rust toolchain](https://rustup.rs)**: the version pinned in `rust-toolchain.toml` is installed automatically by rustup

## Local Setup

```bash
git clone https://github.com/javiermartinch/dicex.git
cd dicex
uv sync --dev                 # creates .venv and builds the Rust extension
uv run pre-commit install     # runs the checks below on every commit
```

After changing the Rust code in `rust/`, run `uv sync --dev --reinstall-package dicex` to rebuild the extension.

## Development Workflow

### Run the quality pipeline

```bash
uv run ruff check .            # Lint
uv run ruff format --check .   # Format check
uv run pyright                 # Type check
uv run pytest --cov            # Test + coverage
cargo fmt --check              # Rust format
cargo clippy -- -D warnings    # Rust lint
```

### Auto-fix formatting and lint

```bash
uv run ruff format .
uv run ruff check --fix .
```

### Build documentation locally

```bash
uv sync --group docs
uv run mkdocs serve
```

## Submitting Changes

1. Fork the repository and create a branch from `main`.
2. Make your changes. Add or update tests as needed.
3. Ensure the full quality pipeline passes locally.
4. Open a Pull Request with a clear description of what and why.

## Commit Messages

Use clear, imperative-mood messages: `Add feature X`, `Fix bug in Y`, `Remove deprecated Z`.

## Code Style

- All code must pass `ruff check`, `ruff format --check`, and `pyright` in strict mode.
- Tests go in `tests/unit/`, `tests/integration/`, or `tests/smoke/`. The Python examples of `README.md` are run by `tests/smoke/test_readme_example.py`, so keep them runnable.
- Follow the [Google docstring convention](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

## Reporting Issues

Use the [issue templates](https://github.com/javiermartinch/dicex/issues/new/choose) to report bugs or request features.
