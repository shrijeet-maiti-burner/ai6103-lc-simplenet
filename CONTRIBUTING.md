# Contributing

Use a short branch and describe the problem, change and validation in the PR. Substantive experiment changes need teammate review. Keep the upstream pin fixed and apply compatibility changes equally to compared methods.

```bash
uv sync --frozen --extra cpu
uv run --frozen --extra cpu ruff check src scripts tests
uv run --frozen --extra cpu ruff format --check src scripts tests
uv run --frozen --extra cpu pytest -q
uv run --frozen --extra cpu python scripts/check_repository.py
git diff --check
```

Keep datasets, weights, credentials, generated runs and personal research out of Git. Retain the original results and provenance when changing source; a new code revision is not a new experiment. Follow the [evaluation protocol](docs/protocol.md).
