## What and why

<!-- What does this change do, and why is it needed? Link the issue if there is one: "Fixes #123". -->

## Checklist

- [ ] Tests added or updated for the change (`uv run pytest --cov`; coverage stays above the threshold).
- [ ] `uv run pre-commit run --all-files` passes (ruff, pyright strict, cargo fmt and clippy).
- [ ] Public API changes are documented (docstrings, README, `docs/`) and backward compatible, or flagged as breaking.
- [ ] Changes to the mathematics reference the equation or result of the paper they implement.
