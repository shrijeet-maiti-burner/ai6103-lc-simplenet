# coding freeze: 16 november

freeze the reviewed training, evaluation and demo code. report text, author details and presentation edits can continue afterward. changes to frozen code need a concrete defect, a regression check and a new source identity.

## result boundary

the original study is complete. relighting improves public localization, but its recall loss exceeds the agreed tolerance. no consistency setting qualifies as an overall detection improvement. the [score-field follow-up](score-field-validation-results.md) is also a negative validation-only result; selected k=4 is not a recommended replacement detector. keep these findings and the exposed-test boundary explicit.

no new coefficient, checkpoint, category, threshold or readout may be chosen from test outcomes. a working demo, successful replay and passing software tests do not establish detector improvement.

## completed follow-up, 18 november

the already-fixed R0 matrix has finished: all 168 original AD 2 heads, all eight categories, all three seeds and the independent raw-score audit. the [complete result](routed-results.md) retains both baseline anchors and all controls. the original recall-preservation target fails. the separate blade-lighting result meets B0 point targets on three-seed averages, with per-seed and background failures and uncertain gain over the stronger matched control.

the [aggregate evidence release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-results-2026-11-18) is separate from the reviewed code freeze. [PR #37](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/37) and [PR #39](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/39) still need teammate review; publishing results does not merge them or replace the reviewed source snapshot.

the report and video drafts still need these outcomes integrated. no new training recipe is introduced by this results update.

## source checks

from the selected clean commit with its pinned submodule:

```bash
uv sync --frozen --extra cpu
uv run --frozen --extra cpu ruff check .
uv run --frozen --extra cpu ruff format --check .
uv run --frozen --extra cpu pytest
uv run --frozen --extra cpu python -m lc_simplenet doctor
uv run --frozen --extra cpu python scripts/check_repository.py
uv run --frozen --extra cpu python scripts/smoke_upstream.py --device cpu
uv run --frozen --extra cpu python scripts/smoke_paired.py
git diff --check
```

build the [complete source package](source-package.md), then check a fresh extraction. it must include the original simplenet source, licences and lockfile. datasets, weights, generated runs and personal research stay outside git and the source zip.

the historical score-field manifest describes its recorded source only. use `scripts/write_source_provenance.py` for a new execution; do not rewrite an old receipt to match changed code.

## final handoff

- record the selected source commit, upstream pin, archive checksum and passed checks.
- retain original, augmentation and consistency controls and their actual evidence; do not promote a failed control as the proposed method.
- keep the report within six body pages in the supplied unchanged layout. inspect the rendered pdf after content changes.
- each member supplies their own required identifiers and reviews their recorded contribution.
- play the final video and audio in full at normal speed; keep it within five minutes.
- confirm the portal cutoff, sole report submitter, instructor access and upload receipts. the working submission date remains 23 november.

see the [submission checklist](submission.md). coding freeze is a source-control milestone, not evidence that these final submission steps are complete.
