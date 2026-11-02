# AD study results, 27 October

Development results for the provisional AD study in [the plan](ad-study-plan.md). All settings were fixed before any test image was scored. The test scores below are B's quick check (image and pixel AUROC inside the 288 crop), not D's evaluation. MVTec AD test images share one lighting setup, so they cannot show a lighting benefit; that needs AD 2.

## Setup

- Bottle and grid; B0, B1 and B2; seeds 0, 1 and 2; batch 8; resize 329, center crop 288; brightness and gamma 0.8 to 1.2.
- 80 source passes: 1,680 updates for bottle and 2,160 for grid. One A40 per run, 15 to 18 minutes each.
- Paired runs use the #2 runner with the normal diagnostics from this branch (`cce0cb4`). The single-view references are the #3 baseline: grid at `eac7968`, bottle seeds 1 and 2 at `e94f57c`, and bottle seed 0 at `4c91c64`, a commit lost in a rebase. All six were rerun at `22fed6a` and gave the same checkpoints; bottle seed 0's is `3f3452bd46b6a9a4429c10af59f35b4fdc7cfcb4922578ccfd0b70120753d656` (see [baseline results](baseline-results.md)).
- All 18 paired runs completed their fixed horizon with exact checkpoint replay. The three grid single-view runs reload with zero score difference.

## Consistency weight

Tuning seed 3, 256 updates, normal validation only. Ratios are B2 over B1; the spread and hinge columns are the worst category.

| Weight | Pair MSE, bottle | Pair MSE, grid | Mean | Spread | Synthetic hinge | Eligible |
|---|---:|---:|---:|---:|---:|---|
| 0.01 | 1.06 | 1.01 | 1.03 | 0.91 | 1.02 | yes |
| 0.1 | 0.96 | 1.13 | 1.05 | 0.88 | 1.14 | yes |
| 1.0 | 0.56 | 1.16 | 0.86 | 0.91 | 0.71 | yes |

The rule chose 1.0. Bottle drove the choice; grid showed no gain in this short search. `scripts/select_consistency_weight.py` reads the eight search runs and writes `selection.json`, which holds every trial's metrics and is in the study release (SHA-256 `47af2820860d07aaff8564ee4755cc327c6b94ae310dc449c6abbdd640776fa8`).

## Normal validation

Mean over the three seeds. Pair MSE is the patch-score change across the four brightness/gamma corners; lower is more consistent.

| Category | Variant | Pair MSE | Score spread | Synthetic hinge |
|---|---|---:|---:|---:|
| bottle | B0 | 4.19 | 0.367 | 0.045 |
| bottle | B1 | 0.0399 | 0.339 | 0.023 |
| bottle | B2 | 0.0115 | 0.269 | 0.029 |
| grid | B0 | 0.296 | 0.285 | 0.011 |
| grid | B1 | 0.0221 | 0.279 | 0.016 |
| grid | B2 | 0.0114 | 0.248 | 0.008 |

Against B1, B2 lowers pair MSE by 71% on bottle and 48% on grid, and the score spread by 21% and 11%. On bottle, B2's synthetic hinge is above B1's on two of three seeds (B1 / B2: 0.02356 / 0.03775, 0.03127 / 0.01969 and 0.01404 / 0.02969 for seeds 0, 1 and 2), and the ratio of the seed means is 1.27, above the 1.25 limit (grid 0.49). The rule applied that limit only to the seed-3 search, so the choice stands, but the full-run check later written for AD 2 would rule out 1.0 on bottle.

## Test AUROC (%)

Bottle has 83 test images (63 defective), grid 78 (57 defective). Values are seeds 0 / 1 / 2, then mean ± sample SD. The SimpleNet paper reports bottle 100 / 98.0 and grid 99.7 / 98.8 (image / pixel), with the checkpoint chosen on test AUROC.

| Category | Model | Image | Mean | Pixel | Mean |
|---|---|---|---:|---|---:|
| bottle | single-view | 99.92 / 100.00 / 100.00 | 99.97 ± 0.05 | 97.98 / 98.06 / 98.11 | 98.05 ± 0.07 |
| bottle | B0 | 99.84 / 100.00 / 100.00 | 99.95 ± 0.09 | 98.04 / 98.05 / 97.96 | 98.02 ± 0.05 |
| bottle | B1 | 99.92 / 100.00 / 100.00 | 99.97 ± 0.05 | 98.02 / 98.03 / 98.06 | 98.04 ± 0.02 |
| bottle | B2 | 99.84 / 100.00 / 100.00 | 99.95 ± 0.09 | 98.05 / 98.09 / 98.10 | 98.08 ± 0.03 |
| grid | single-view | 69.92 / 48.12 / 98.16 | 72.07 ± 25.09 | 66.15 / 42.52 / 97.20 | 68.62 ± 27.42 |
| grid | B0 | 98.50 / 98.91 / 98.41 | 98.61 ± 0.27 | 98.75 / 98.88 / 98.77 | 98.80 ± 0.07 |
| grid | B1 | 98.50 / 98.91 / 98.50 | 98.64 ± 0.24 | 98.75 / 98.84 / 98.75 | 98.78 ± 0.05 |
| grid | B2 | 98.83 / 99.50 / 99.00 | 99.11 ± 0.35 | 98.75 / 98.85 / 98.78 | 98.79 ± 0.05 |

## Findings

- Bottle reproduces the paper with a fixed final checkpoint and the 20% normal holdout. Every variant is at the ceiling, so bottle cannot separate them.
- On grid, B2 is above B1 in image AUROC on every seed, by 0.33, 0.59 and 0.50 points. Three seeds on one category do not establish an effect. Pixel AUROC does not change in either category.
- The upstream single-view recipe is unstable on grid. Its loss sits near 0.0002 and then the discriminator collapses (loss back near 1.0, p_true about 0.2): in the final meta-epoch for seeds 0 and 1, and about four meta-epochs before the end for seed 2, which partly recovers. The fixed final checkpoints score 69.9, 48.1 and 98.2 image AUROC. Upstream's test-selected checkpoint would hide this. Bottle never collapsed.
- The paired runner did not collapse on grid. On bottle, B0 spiked late on two seeds, and one had not fully settled by the end; B1 and B2 never spiked. Whether relit views stabilize training has not been tested.

## Grid single-view collapse, 28 October

To check whether the grid gap to the paper comes from our code or from upstream's checkpoint rule, the three grid single-view runs were repeated at `eac7968` and scored after every meta-epoch on normal validation and on the test set. Random-number state was saved before each scoring pass and restored after it. Each rerun's final validation scores match the original run exactly, and its final test AUROC equals the table above. This scores the test set 40 times per seed on purpose. It changes no setting and selects nothing. It shows what upstream's rule, which keeps the epoch with the best test AUROC, would report (see upstream [issue #7](https://github.com/DonaldRR/SimpleNet/issues/7)).

Image AUROC (%):

| Seed | Best test epoch (upstream rule) | Meta-epochs 10 to 36, mean (range) | Final (ours) | Collapse |
|---|---|---|---:|---|
| 0 | 99.83 (epoch 4) | 98.75 (98.41 to 99.08) | 69.92 | epoch 40 |
| 1 | 99.08 (epoch 20) | 98.51 (98.25 to 99.08) | 48.12 | epoch 40 |
| 2 | 99.50 (epoch 9) | 98.45 (98.25 to 99.16) | 98.16 | epoch 37, partly recovers |
| mean | 99.47 | 98.57 | 72.07 | |

Pixel AUROC over meta-epochs 10 to 36 averages 98.80, 98.90 and 98.82.

- Upstream's rule gives 99.47 on average, close to the paper's 99.7. For seed 0 it picks epoch 4, where the training loss (0.76) has not settled.
- Between meta-epochs 10 and 36 the single-view baseline averages 98.45 to 98.75 per seed, the level of the paired B0 (98.61) and B1 (98.64). The low single-view number in the main table comes from where training stops.
- Each collapse starts with a jump in the mean training loss per meta-epoch, from about 0.0002 to 0.78 (seed 0) and 0.61 (seed 1) in the last meta-epoch, and for seed 2 to 0.04 in epoch 37 and 0.86 in epoch 38. Normal-validation score spread rises with it (seed 0: 0.22 to 0.39). Both are normal-only signals that a stopping rule could use, but none was declared before testing, so the baseline keeps the fixed final checkpoint.

## Inference cost

Batch 1 on one A40, CUDA-synchronized, 30 warmup and 300 timed calls, input a normalized 288 × 288 tensor already on the GPU, each model in its own process.

| Model | Network to patch scores | With score map | Peak allocated | Parameters, heads / backbone |
|---|---:|---:|---:|---|
| single-view | 12.2 ms | 328 ms | 424 MiB (788 with map) | 3.94M / 68.9M |
| paired (B0, B1, B2) | 12.1 ms | 16.6 ms | 424 MiB | 3.94M / 68.9M |

- The networks are the same size and take the same time, so relit training and the consistency loss add nothing at inference.
- The single-view map time is upstream `_predict`, which also resizes the 1536-channel feature map to 288 × 288 and copies it to the CPU without using it for scoring (193 ms on another node). The paired path builds the same score map without that step.
- The backbone count is the full Wide ResNet-50-2; only layers up to layer3 run.

## Limits

- Test results have now been seen. Changing a setting from here starts a new study and must disclose this exposure.
- The grid trajectory above scored the test set at every meta-epoch. It is a diagnostic of upstream's checkpoint rule; no setting or checkpoint was chosen from it.
- C confirmed the brightness and gamma ranges, 0.8 to 1.2, on 30 October, so B1 and B2 stand.
- D's evaluation, including PRO, is separate. These numbers are a development check.

## Artifacts

The [study release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad-study-bottle-grid-2026-10-27) has the trained heads for all 18 paired runs, the grid single-view checkpoints, normal-validation outputs, run metadata, the weight selection, these test scores and the scoring script. The [diagnostics release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad-grid-diagnostics-2026-10-28) has the per-epoch grid trajectory, the inference timings and the two scripts behind them. Dataset images and pretrained backbone weights are not included.
