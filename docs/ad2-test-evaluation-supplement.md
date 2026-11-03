# AD 2 public-test evaluation: supplement

Companion to [the evaluation](ad2-test-evaluation.md). It gives both declared macro scopes, every comparison agreed on #8, the categories that exceed the 5-point recall drop, and the achieved validation false-positive rate with counts. All numbers come from `aggregate.json`, `metrics.csv` and `thresholds.json` (hashes in the main document). Macros are unweighted over categories per seed, then mean ± sample SD over seeds 0, 1 and 2. `n/a` means the condition is not available in every category of the scope, so the tool computes no macro.

Conditions: `nonregular` is every non-regular image; `exposure_changed` pools underexposed and overexposed; `additional_light` pools shift_1 to shift_4. Within each category, pooled rates use every image in the group; category macros remain unweighted (see [lighting definitions](ad2-lighting.md)).

## Eight categories: Recall

| Method | regular | underexposed | overexposed | shift_1 | nonregular | exposure_changed | additional_light |
|---|---|---|---|---|---|---|---|
| B0 | 0.292 ± 0.029 | 0.525 ± 0.066 | 0.397 ± 0.034 | 0.825 ± 0.038 | 0.682 ± 0.041 | 0.461 ± 0.050 | 0.834 ± 0.035 |
| B1 | 0.278 ± 0.013 | 0.464 ± 0.017 | 0.261 ± 0.010 | 0.747 ± 0.038 | 0.606 ± 0.016 | 0.362 ± 0.004 | 0.764 ± 0.023 |
| B2 at 0.01 | 0.264 ± 0.017 | 0.464 ± 0.021 | 0.269 ± 0.038 | 0.722 ± 0.046 | 0.604 ± 0.024 | 0.367 ± 0.026 | 0.758 ± 0.027 |
| B2 at 0.1 | 0.283 ± 0.008 | 0.464 ± 0.027 | 0.283 ± 0.008 | 0.714 ± 0.064 | 0.616 ± 0.026 | 0.374 ± 0.015 | 0.778 ± 0.041 |
| B2 at 1.0 | 0.264 ± 0.021 | 0.478 ± 0.059 | 0.264 ± 0.013 | 0.683 ± 0.095 | 0.574 ± 0.078 | 0.371 ± 0.034 | 0.709 ± 0.106 |

## Eight categories: FPR (target 0.05)

| Method | regular | underexposed | overexposed | shift_1 | nonregular | exposure_changed | additional_light |
|---|---|---|---|---|---|---|---|
| B0 | 0.060 ± 0.025 | 0.362 ± 0.047 | 0.208 ± 0.030 | 0.634 ± 0.032 | 0.495 ± 0.041 | 0.285 ± 0.038 | 0.640 ± 0.043 |
| B1 | 0.064 ± 0.024 | 0.310 ± 0.045 | 0.064 ± 0.033 | 0.599 ± 0.080 | 0.435 ± 0.026 | 0.187 ± 0.014 | 0.601 ± 0.055 |
| B2 at 0.01 | 0.047 ± 0.021 | 0.291 ± 0.038 | 0.058 ± 0.021 | 0.561 ± 0.067 | 0.419 ± 0.036 | 0.174 ± 0.023 | 0.583 ± 0.059 |
| B2 at 0.1 | 0.075 ± 0.042 | 0.282 ± 0.022 | 0.088 ± 0.047 | 0.534 ± 0.123 | 0.415 ± 0.084 | 0.185 ± 0.034 | 0.574 ± 0.116 |
| B2 at 1.0 | 0.051 ± 0.019 | 0.275 ± 0.012 | 0.054 ± 0.025 | 0.501 ± 0.090 | 0.383 ± 0.055 | 0.165 ± 0.015 | 0.533 ± 0.080 |

## Eight categories: Image AUROC

| Method | regular | underexposed | overexposed | shift_1 | nonregular | exposure_changed | additional_light |
|---|---|---|---|---|---|---|---|
| B0 | 0.703 ± 0.010 | 0.642 ± 0.029 | 0.685 ± 0.027 | 0.673 ± 0.010 | 0.631 ± 0.005 | 0.654 ± 0.011 | 0.639 ± 0.011 |
| B1 | 0.696 ± 0.016 | 0.671 ± 0.030 | 0.695 ± 0.031 | 0.678 ± 0.026 | 0.635 ± 0.007 | 0.665 ± 0.021 | 0.641 ± 0.015 |
| B2 at 0.01 | 0.698 ± 0.069 | 0.663 ± 0.053 | 0.681 ± 0.077 | 0.667 ± 0.028 | 0.634 ± 0.013 | 0.660 ± 0.046 | 0.645 ± 0.016 |
| B2 at 0.1 | 0.668 ± 0.063 | 0.668 ± 0.021 | 0.684 ± 0.081 | 0.662 ± 0.013 | 0.638 ± 0.021 | 0.657 ± 0.047 | 0.645 ± 0.028 |
| B2 at 1.0 | 0.689 ± 0.023 | 0.667 ± 0.022 | 0.694 ± 0.021 | 0.646 ± 0.027 | 0.636 ± 0.003 | 0.666 ± 0.012 | 0.636 ± 0.015 |

## Seven categories (without sheet_metal): Recall

| Method | regular | underexposed | overexposed | shift_1 | nonregular | exposure_changed | additional_light |
|---|---|---|---|---|---|---|---|
| B0 | 0.302 ± 0.031 | 0.568 ± 0.070 | 0.422 ± 0.033 | 0.800 ± 0.044 | 0.681 ± 0.044 | 0.495 ± 0.052 | 0.812 ± 0.039 |
| B1 | 0.279 ± 0.024 | 0.489 ± 0.038 | 0.260 ± 0.015 | 0.717 ± 0.040 | 0.594 ± 0.026 | 0.375 ± 0.023 | 0.734 ± 0.026 |
| B2 at 0.01 | 0.263 ± 0.011 | 0.495 ± 0.025 | 0.270 ± 0.024 | 0.692 ± 0.045 | 0.593 ± 0.018 | 0.383 ± 0.018 | 0.729 ± 0.025 |
| B2 at 0.1 | 0.286 ± 0.010 | 0.498 ± 0.027 | 0.286 ± 0.010 | 0.683 ± 0.061 | 0.609 ± 0.022 | 0.392 ± 0.010 | 0.754 ± 0.037 |
| B2 at 1.0 | 0.267 ± 0.025 | 0.511 ± 0.068 | 0.267 ± 0.019 | 0.670 ± 0.118 | 0.576 ± 0.092 | 0.389 ± 0.040 | 0.701 ± 0.125 |

## Seven categories (without sheet_metal): FPR (target 0.05)

| Method | regular | underexposed | overexposed | shift_1 | nonregular | exposure_changed | additional_light |
|---|---|---|---|---|---|---|---|
| B0 | 0.069 ± 0.028 | 0.414 ± 0.053 | 0.238 ± 0.035 | 0.581 ± 0.037 | 0.487 ± 0.039 | 0.326 ± 0.044 | 0.600 ± 0.040 |
| B1 | 0.073 ± 0.028 | 0.354 ± 0.051 | 0.073 ± 0.038 | 0.565 ± 0.052 | 0.426 ± 0.018 | 0.214 ± 0.016 | 0.568 ± 0.036 |
| B2 at 0.01 | 0.054 ± 0.024 | 0.332 ± 0.043 | 0.066 ± 0.024 | 0.522 ± 0.061 | 0.402 ± 0.033 | 0.199 ± 0.026 | 0.539 ± 0.055 |
| B2 at 0.1 | 0.086 ± 0.048 | 0.323 ± 0.025 | 0.101 ± 0.054 | 0.503 ± 0.092 | 0.412 ± 0.061 | 0.212 ± 0.039 | 0.553 ± 0.074 |
| B2 at 1.0 | 0.058 ± 0.022 | 0.314 ± 0.014 | 0.062 ± 0.028 | 0.466 ± 0.103 | 0.373 ± 0.069 | 0.188 ± 0.017 | 0.502 ± 0.102 |

## Seven categories (without sheet_metal): Image AUROC

| Method | regular | underexposed | overexposed | shift_1 | nonregular | exposure_changed | additional_light |
|---|---|---|---|---|---|---|---|
| B0 | 0.701 ± 0.003 | 0.636 ± 0.033 | 0.665 ± 0.011 | 0.667 ± 0.016 | 0.630 ± 0.006 | 0.640 ± 0.012 | 0.636 ± 0.011 |
| B1 | 0.682 ± 0.008 | 0.654 ± 0.036 | 0.688 ± 0.022 | 0.672 ± 0.023 | 0.631 ± 0.009 | 0.650 ± 0.022 | 0.633 ± 0.005 |
| B2 at 0.01 | 0.691 ± 0.070 | 0.644 ± 0.053 | 0.668 ± 0.079 | 0.664 ± 0.029 | 0.630 ± 0.013 | 0.642 ± 0.046 | 0.637 ± 0.010 |
| B2 at 0.1 | 0.657 ± 0.064 | 0.646 ± 0.019 | 0.677 ± 0.080 | 0.647 ± 0.011 | 0.632 ± 0.019 | 0.640 ± 0.041 | 0.633 ± 0.025 |
| B2 at 1.0 | 0.678 ± 0.024 | 0.654 ± 0.014 | 0.689 ± 0.025 | 0.652 ± 0.033 | 0.636 ± 0.005 | 0.655 ± 0.006 | 0.637 ± 0.015 |

## Eight categories: recall change in percentage points

`!` marks a drop of more than 5 points (recall not preserved).

| Candidate against reference | regular | underexposed | overexposed | shift_1 | nonregular | exposure_changed | additional_light |
|---|---|---|---|---|---|---|---|
| B1 against B0 | -1.4 | -6.1 ! | -13.6 ! | -7.8 ! | -7.5 ! | -9.9 ! | -7.0 ! |
| B2 at 0.01 against B0 | -2.8 | -6.1 ! | -12.8 ! | -10.3 ! | -7.8 ! | -9.4 ! | -7.6 ! |
| B2 at 0.1 against B0 | -0.8 | -6.1 ! | -11.4 ! | -11.1 ! | -6.6 ! | -8.7 ! | -5.6 ! |
| B2 at 1.0 against B0 | -2.8 | -4.7 | -13.3 ! | -14.2 ! | -10.7 ! | -9.0 ! | -12.5 ! |
| B2 at 0.01 against B1 | -1.4 | +0.0 | +0.8 | -2.5 | -0.2 | +0.4 | -0.6 |
| B2 at 0.1 against B1 | +0.6 | +0.0 | +2.2 | -3.3 | +0.9 | +1.1 | +1.4 |
| B2 at 1.0 against B1 | -1.4 | +1.4 | +0.3 | -6.4 ! | -3.2 | +0.8 | -5.5 ! |

## Seven categories (without sheet_metal): recall change in percentage points

`!` marks a drop of more than 5 points (recall not preserved).

| Candidate against reference | regular | underexposed | overexposed | shift_1 | nonregular | exposure_changed | additional_light |
|---|---|---|---|---|---|---|---|
| B1 against B0 | -2.2 | -7.9 ! | -16.2 ! | -8.3 ! | -8.7 ! | -12.1 ! | -7.7 ! |
| B2 at 0.01 against B0 | -3.8 | -7.3 ! | -15.2 ! | -10.8 ! | -8.8 ! | -11.3 ! | -8.3 ! |
| B2 at 0.1 against B0 | -1.6 | -7.0 ! | -13.7 ! | -11.7 ! | -7.3 ! | -10.3 ! | -5.8 ! |
| B2 at 1.0 against B0 | -3.5 | -5.7 ! | -15.6 ! | -13.0 ! | -10.5 ! | -10.6 ! | -11.1 ! |
| B2 at 0.01 against B1 | -1.6 | +0.6 | +1.0 | -2.5 | -0.1 | +0.8 | -0.6 |
| B2 at 0.1 against B1 | +0.6 | +1.0 | +2.5 | -3.5 | +1.5 | +1.7 | +2.0 |
| B2 at 1.0 against B1 | -1.3 | +2.2 | +0.6 | -4.8 | -1.8 | +1.4 | -3.4 |

## Categories more than 5 points below the reference

Points of recall lost per category, from `category_recall_checks`. Conditions not listed have no such category. A category appears only for lighting conditions it has.

### B1 against B0

- regular: fruit_jelly -18, wallplugs -9
- underexposed: fabric -9, fruit_jelly -11, rice -22, wallplugs -7, walnuts -7
- overexposed: can -7, fabric -18, fruit_jelly -11, rice -7, vial -53, wallplugs -9, walnuts -9
- shift_1: fabric -22, fruit_jelly -24, rice -9
- shift_2: fabric -11
- shift_3: fabric -24, wallplugs -18
- nonregular: fabric -17, fruit_jelly -16, rice -7, vial -10, wallplugs -7
- exposure_changed: fabric -13, fruit_jelly -11, rice -14, vial -27, wallplugs -8, walnuts -8
- additional_light: fabric -19, fruit_jelly -24, wallplugs -7
- all: fabric -14, fruit_jelly -16, vial -9, wallplugs -7

### B2 at 0.01 against B0

- regular: fruit_jelly -9, wallplugs -7
- underexposed: rice -33
- overexposed: can -13, fabric -18, rice -9, vial -47, wallplugs -9, walnuts -9
- shift_1: fabric -22, fruit_jelly -16, rice -36, sheet_metal -7
- shift_2: fabric -13, rice -13
- shift_3: fabric -13, wallplugs -22
- nonregular: fabric -14, fruit_jelly -7, rice -19, vial -9, wallplugs -7
- exposure_changed: can -7, fabric -11, rice -21, vial -23, wallplugs -7, walnuts -7
- additional_light: fabric -16, fruit_jelly -16, rice -17, wallplugs -7
- all: fabric -13, fruit_jelly -8, rice -16, vial -8, wallplugs -7

### B2 at 0.1 against B0

- regular: fruit_jelly -11
- underexposed: fabric -7, fruit_jelly -7, rice -31
- overexposed: can -11, fabric -11, fruit_jelly -9, rice -13, vial -44
- shift_1: fabric -18, fruit_jelly -11, rice -42, sheet_metal -7, wallplugs -7
- shift_3: wallplugs -31
- nonregular: fabric -6, fruit_jelly -9, rice -16, vial -8, wallplugs -8
- exposure_changed: can -6, fabric -9, fruit_jelly -8, rice -22, vial -22
- additional_light: fruit_jelly -11, rice -12, wallplugs -13
- all: fruit_jelly -9, rice -14, vial -8, wallplugs -7

### B2 at 1.0 against B0

- regular: fruit_jelly -13, vial -7, wallplugs -7
- underexposed: fruit_jelly -13, rice -29
- overexposed: can -11, fabric -11, fruit_jelly -9, rice -11, vial -51, wallplugs -11
- shift_1: fabric -13, fruit_jelly -18, rice -20, sheet_metal -22, vial -7, wallplugs -33
- shift_2: rice -7, sheet_metal -22, vial -11, wallplugs -33
- shift_3: sheet_metal -22, wallplugs -49
- nonregular: fruit_jelly -13, rice -13, sheet_metal -12, vial -13, wallplugs -26
- exposure_changed: can -6, fruit_jelly -11, rice -20, vial -26, wallplugs -8
- additional_light: fabric -6, fruit_jelly -18, rice -9, sheet_metal -22, vial -7, wallplugs -39
- all: fruit_jelly -13, rice -11, sheet_metal -10, vial -12, wallplugs -23

### B2 at 0.01 against B1

- regular: can -7, rice -11
- underexposed: rice -11
- overexposed: can -7
- shift_1: rice -27
- shift_2: rice -18
- nonregular: rice -12
- exposure_changed: rice -7
- additional_light: rice -16
- all: rice -12

### B2 at 0.1 against B1

- regular: rice -11
- underexposed: rice -9, sheet_metal -7
- overexposed: rice -7
- shift_1: rice -33, wallplugs -7
- shift_3: wallplugs -13
- nonregular: rice -9
- exposure_changed: rice -8
- additional_light: rice -10, wallplugs -6
- all: rice -10

### B2 at 1.0 against B1

- regular: can -7, rice -9, vial -7
- underexposed: rice -7
- shift_1: rice -11, sheet_metal -18, wallplugs -33
- shift_2: rice -11, sheet_metal -20, vial -7, wallplugs -31
- shift_3: sheet_metal -22, wallplugs -31
- nonregular: rice -7, sheet_metal -13, wallplugs -19
- exposure_changed: rice -6
- additional_light: rice -7, sheet_metal -20, wallplugs -32
- all: rice -7, sheet_metal -11, wallplugs -16

## Achieved validation false-positive rate

Each run's threshold is the higher empirical 95% quantile of its normal validation scores, so the achieved validation FPR is at or below 5% by construction (ties and small samples make it lower). Per-run counts are in `thresholds.json` (`validation_images`, `validation_false_positives`, `validation_fpr`).

| Method | runs | validation images (min–max) | false positives (min–max) | achieved FPR (min–max) | mean FPR |
|---|---|---|---|---|---|
| B0 | 24 | 19–48 | 0–2 | 0.000–0.049 | 0.033 |
| B1 | 24 | 19–48 | 0–2 | 0.000–0.049 | 0.033 |
| B2@0.01 | 24 | 19–48 | 0–2 | 0.000–0.049 | 0.033 |
| B2@0.1 | 24 | 19–48 | 0–2 | 0.000–0.049 | 0.033 |
| B2@1 | 24 | 19–48 | 0–2 | 0.000–0.049 | 0.033 |

Test-set FPR and recall counts (false positives over normal images, true positives over defective images) for every category, method, seed and condition are in `metrics.csv`.
