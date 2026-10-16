# Contributing

Clone with `--recurse-submodules`, run `uv sync --frozen`, and work on a short branch such as `research/lighting-review` or `feat/ad2-loader`. Open a PR with the problem, change, evidence and limitations. Obtain a teammate review before merging substantive experiment changes.

Research can begin in any member's notes; roles are unassigned. Record work when it happens. Do not backfill another person's contribution or describe a source-reported result as our own.

Before a PR:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/check_repository.py
git diff --check
```

Use `--extra cpu` or `--extra cuda` consistently after selecting a runtime; plain `uv run` can remove optional packages. See [setup](docs/setup.md). Tests of PyTorch contracts skip in the lightweight environment and run in CPU CI.

Keep the upstream submodule fixed. Record the reason for shared compatibility/protocol fixes and apply them to every compared variant. Commit reviewed configs and small summaries; retain raw outputs and hashes in local run directories or team artifact storage.
