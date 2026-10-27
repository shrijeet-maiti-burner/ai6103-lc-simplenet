# AD study results, 27 October

Development results for the provisional AD study in [the plan](ad-study-plan.md). All settings were fixed before any test image was scored. The test scores below are B's quick check (image and pixel AUROC inside the 288 crop), not D's evaluation. MVTec AD test images share one lighting setup, so they cannot show a lighting benefit; that needs AD 2.

## Setup

- Bottle and grid; B0, B1 and B2; seeds 0, 1 and 2; batch 8; resize 329, center crop 288; brightness and gamma 0.8 to 1.2.
- 80 source passes: 1,680 updates for bottle and 2,160 for grid. One A40 per run, 15 to 18 minutes each.
- Paired runs use the #2 runner with the normal diagnostics from this branch (`cce0cb4`). The single-view references are the #3 baseline (`eac7968`; bottle seed 0 at `4c91c64`, see [baseline results](baseline-results.md)).
- All 18 paired runs completed their fixed horizon with exact checkpoint replay. The three grid single-view runs reload with zero score difference.

## Consistency weight

Tuning seed 3, 256 updates, normal validation only. Ratios are B2 over B1; the spread and hinge columns are the worst category.

| Weight | Pair MSE, bottle | Pair MSE, grid | Mean | Spread | Synthetic hinge | Eligible |
|---|---:|---:|---:|---:|---:|---|
| 0.01 | 1.06 | 1.01 | 1.03 | 0.91 | 1.02 | yes |
| 0.1 | 0.96 | 1.13 | 1.05 | 0.88 | 1.14 | yes |
| 1.0 | 0.56 | 1.16 | 0.86 | 0.91 | 0.71 | yes |

The rule chose 1.0. Bottle drove the choice; grid showed no gain in this short search.

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

Against B1, B2 lowers pair MSE by 71% on bottle and 48% on grid, and the score spread by 21% and 11%. On bottle, B2's synthetic hinge is above B1's on two of three seeds.

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

## Limits

- Test results have now been seen. Changing a setting from here starts a new study and must disclose this exposure.
- Brightness and gamma ranges are pending C. A change reruns B1 and B2.
- D's evaluation, including PRO, is separate. These numbers are a development check.

## Artifacts

The [study release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad-study-bottle-grid-2026-10-27) has the trained heads for all 18 paired runs, the grid single-view checkpoints, normal-validation outputs, run metadata, the weight selection, these test scores and the scoring script. Dataset images and pretrained backbone weights are not included.
