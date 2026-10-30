# Harsh - paired lighting

GitHub: [HarshSaand](https://github.com/HarshSaand). Role C: paired lighting and consistency. Status: local implementation tested; trainer integration pending.

## Question

Does adding consistency between aligned original and relit normal patch scores reduce lighting-related false alarms beyond lighting augmentation alone, without losing defect sensitivity?

## Contribution log

26 October 2026: added paired image construction and reproducible brightness/gamma sampling with parameter and clipping metadata in `src/lc_simplenet/pairs.py`. Added the combined two-view hinge and normal-score consistency objective in `src/lc_simplenet/losses.py`, and tests in `tests/test_pairs.py` and `tests/test_paired_objective.py`.

The shared repository already supplied `photometric.py` and `patch_score_consistency`; those are not new contributions from this work. The new components build on them.

## Evidence and limits

The CPU test suite passed all 21 tests. Tests check alignment, input preservation, sampling, clipping, objective values and gradients. The frozen-extractor gradient check uses a toy network, not SimpleNet. No dataset training, robustness measurement or accuracy claim is available from this implementation.

See the [integration handoff](../paired-lighting.md) for exact input conventions and next steps with A, B and D. Transform ranges and consistency weight remain open pending validation. Nothing has been pushed to the shared repository as of this entry.

## 30 October update

Added `paired_training.py` for CHW normalization, one joint discriminator forward and detached score diagnostics. Added integration tests, an actual upstream architecture smoke script, and a reusable lighting preview script. `configs/paired-lighting.json` records the team's proposed [0.8, 1.2] ranges for both parameters; the B2 weight is not selected.

All 25 unit tests passed. The new B1/B2 test confirms identical forward diagnostics and BatchNorm statistics from identical inputs/weights, with only the intended loss term differing. Real-data results are still pending. Changes remain local for review before pushing.

The actual upstream architecture smoke check also passed: frozen backbone, nonzero trainable gradients, successful adapter update and exact saved-weight score restoration. It used synthetic inputs and random weights; no benchmark improvement is claimed.

Final pre-publication review: two independent reviewers checked scientific correctness and implementation. Fixed BatchNorm mutation on invalid settings and half-precision loss overflow, with regression tests and defensive pair validation. All 34 tests pass; the second review found no remaining blocker. The paired architecture smoke check is now included in CPU CI. This review does not replace teammate review before merging.
