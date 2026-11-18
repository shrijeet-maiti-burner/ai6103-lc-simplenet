# AD 2 study results, 31 October

Development results for the AD 2 study fixed in [the policy](../configs/ad2-study-policy.json). Everything below is normal validation. This study has not scored any AD 2 test image; D scores the test sets. The fabric runs in #9 used another protocol and were scored on fabric's public test set, so they are not part of this study.

## Setup

- All eight categories; B0, B1 and B2; seeds 0, 1 and 2; batch 8; full-frame bilinear resize to 256 × 256 with no crop; brightness and gamma 0.8 to 1.2.
- 80 source passes: 1,440 (sheet_metal) to 4,320 (walnuts) updates per run.
- Paired runs at `55dabd2` with `--cache-images`, `--verify-replay` and `--normal-diagnostics`. The single-view references are upstream SimpleNet at `212c11b` (#6) with the same input, 40 meta-epochs, fixed final epoch.
- A40 GPUs, up to three runs sharing one. A paired run took a median of 21 minutes, a single-view run 53 minutes.
- Every paired run reached its fixed step and replayed its checkpoint exactly. Two single-view runs, fruit_jelly seed 1 and sheet_metal seed 2, had not settled by the final epoch and are kept as they are.

## Consistency weight

The policy's search (seed 3, 256 updates per trial) found no eligible weight: at every candidate the synthetic hinge rose above 1.25 times B1 in three or four categories. B2 was then trained at 1.0, the MVTec AD weight, as a recorded deviation (see the policy).

A seed-3 search this short cannot settle eligibility. Within a category, B1's synthetic hinge alone varies 1.6 to 4.4 times across seeds 0 to 2, against a limit of 1.25. So on 31 October at 13:43 (+08), before any full-length B2 run at 0.01 or 0.1 existed, we wrote a second rule: the same safeguards and objective, applied to seed 0 to 2 means of full-length runs at each candidate. If no weight passed, B2 would be exploratory and every weight would be reported as an ablation. The 1.0 runs had finished by then and already failed it. The rule stayed in our notes until this document.

Full-length runs, mean over seeds 0 to 2, B2 over B1:

| Weight | Categories passing | Failing (hinge ratio) | Lowest spread ratio | Macro pair MSE ratio |
|---|---|---|---:|---:|
| 0.01 | 6 of 8 | fabric 1.44, rice 2.22 | 0.90 (vial) | 0.82 |
| 0.1 | 7 of 8 | vial 1.65 | 0.80 (can) | 0.61 |
| 1.0 | 3 of 8 | can 2.07, fabric 2.04, sheet_metal 1.66, vial 1.60, wallplugs 43.5 | 0.55 (fruit_jelly) | 0.40 |

No weight passes, so B2 is exploratory. The main comparison is B0 against B1. B2 at all three weights goes to test scoring as an ablation, and no weight is chosen from test results.

## Normal validation

Mean over the three seeds. Pair MSE is the patch-score change across the four brightness/gamma corners; lower is more consistent. B2 is labeled with its weight.

| Category | Variant | Pair MSE | Score spread | Synthetic hinge |
|---|---|---:|---:|---:|
| can | B0 | 3.22 | 0.323 | 0.00683 |
| can | B1 | 0.0404 | 0.296 | 0.00584 |
| can | B2 0.01 | 0.0318 | 0.286 | 0.00617 |
| can | B2 0.1 | 0.0201 | 0.237 | 0.00335 |
| can | B2 1.0 | 0.011 | 0.19 | 0.0121 |
| fabric | B0 | 0.262 | 0.271 | 0.00437 |
| fabric | B1 | 0.0177 | 0.263 | 0.00257 |
| fabric | B2 0.01 | 0.0176 | 0.277 | 0.00369 |
| fabric | B2 0.1 | 0.0133 | 0.259 | 0.00254 |
| fabric | B2 1.0 | 0.00614 | 0.206 | 0.00524 |
| fruit_jelly | B0 | 10 | 0.575 | 0.0267 |
| fruit_jelly | B1 | 0.0752 | 0.462 | 0.0282 |
| fruit_jelly | B2 0.01 | 0.0559 | 0.492 | 0.0223 |
| fruit_jelly | B2 0.1 | 0.0427 | 0.376 | 0.0223 |
| fruit_jelly | B2 1.0 | 0.0262 | 0.253 | 0.0232 |
| rice | B0 | 0.411 | 0.286 | 0.0025 |
| rice | B1 | 0.0215 | 0.266 | 0.00185 |
| rice | B2 0.01 | 0.0173 | 0.251 | 0.0041 |
| rice | B2 0.1 | 0.0133 | 0.254 | 0.00191 |
| rice | B2 1.0 | 0.00614 | 0.185 | 0.002 |
| sheet_metal | B0 | 0.13 | 0.362 | 0.0357 |
| sheet_metal | B1 | 0.0227 | 0.349 | 0.0274 |
| sheet_metal | B2 0.01 | 0.0216 | 0.348 | 0.029 |
| sheet_metal | B2 0.1 | 0.0186 | 0.339 | 0.0216 |
| sheet_metal | B2 1.0 | 0.0143 | 0.31 | 0.0456 |
| vial | B0 | 9.53 | 0.518 | 0.0267 |
| vial | B1 | 0.0939 | 0.446 | 0.0164 |
| vial | B2 0.01 | 0.0448 | 0.401 | 0.0156 |
| vial | B2 0.1 | 0.0264 | 0.363 | 0.0271 |
| vial | B2 1.0 | 0.0164 | 0.253 | 0.0263 |
| wallplugs | B0 | 0.153 | 0.353 | 0.0178 |
| wallplugs | B1 | 0.0184 | 0.341 | 0.0152 |
| wallplugs | B2 0.01 | 0.0165 | 0.334 | 0.0136 |
| wallplugs | B2 0.1 | 0.0129 | 0.32 | 0.0148 |
| wallplugs | B2 1.0 | 0.0144 | 0.785 | 0.66 |
| walnuts | B0 | 0.111 | 0.287 | 0.00702 |
| walnuts | B1 | 0.0159 | 0.281 | 0.00661 |
| walnuts | B2 0.01 | 0.0139 | 0.287 | 0.00645 |
| walnuts | B2 0.1 | 0.0101 | 0.275 | 0.00553 |
| walnuts | B2 1.0 | 0.0058 | 0.216 | 0.00382 |

## Findings

- B1 lowers pair MSE against B0 in every category, by 5.7 times (sheet_metal) to 133 times (fruit_jelly). Its score spread is 2 to 20% lower.
- B2 lowers pair MSE further, more at higher weights: the macro ratio to B1 is 0.82 at 0.01, 0.61 at 0.1 and 0.40 at 1.0. It also narrows the score spread, to 0.55 of B1 in fruit_jelly at 1.0, and raises the synthetic hinge in some categories at every weight.
- wallplugs B2 at 1.0, seed 2, spiked late and had not settled by its final step. Its spread and hinge drive that row's means. It is kept under the fixed-final rule.
- Lower pair MSE on normal images does not show that defects are found more reliably under changed lighting. That is the test question for D.

## Second environment (#11)

#11 repeats B0, B1 and B2 at 0.01 for all eight categories and three seeds. Data manifests, initial weights, source hashes and training configs match the runs here. The software does not: torch 2.11.0, CUDA 12.8 and cuDNN 9.19 under Python 3.11, against the locked torch 2.6.0, CUDA 12.4 and cuDNN 9.1 under Python 3.12 used here. Runtime options differ too. 69 of its runs used 2 CPU threads and a GPU memory cap of 0.25; the three can B0 runs used 4 threads and a cap of 0.8, and two of them ran without the image cache. None checked checkpoint replay. Every run here used 3 threads, no memory cap, the cache and the replay check. Training losses first differ at step 1 to 3 and then drift apart.

- Seed means agree within 20% for pair MSE and within 25% for score spread.
- B1 lowers pair MSE against B0 by 5.9 to 132 times there.
- Under the full-run rule, 0.01 fails there too, in fabric (1.45) and vial (1.37), with a macro pair MSE ratio of 0.81.
- The synthetic hinge is the unstable metric. With the same seed and settings, a category mean differs by up to 6.9 times between the two setups (walnuts B1). Which categories pass the 1.25 limit therefore depends on run-to-run noise as well as on the weight.

The tables above use the locked environment. #11 is a replicate.

## 288 × 288 check

The official SimpleNet script trains at 288 × 288. We reran the study at 288 as a sensitivity check: the same 72 paired and 24 single-view runs at the same commits, with only the input size changed, B2 fixed at 1.0 and no new weight search. All 96 runs finished and passed the audit. 256 stays the study setting, and neither size will be chosen from test scores.

- B1 lowers pair MSE against B0 in every category at 288 as well, by 5.5 times (sheet_metal) to 112 times (fruit_jelly). B2 lowers it further in every category, with a macro ratio to B1 of 0.35 (0.40 at 256).
- The score-spread result does not carry over. At 288, B1's spread is higher than B0's in five categories: by 3 to 13% in fabric, fruit_jelly, rice and vial, and by 2.8 times in can.
- 20 of the 24 category and variant pair MSE means are within 20% of their 256 values. The exceptions are rice B0 (0.66 times), can B1 (2.7 times), fabric B2 (1.56 times) and wallplugs B2 (0.59 times).
- 288 is less stable. Seven of the 96 runs had not settled by their final step, against three at 256: can B1 seed 1, fruit_jelly B2 seed 1, vial B2 seeds 0 and 1, and the single-view can seed 1, fruit_jelly seed 1 and vial seed 0. can B1 seed 1 spiked late and drives the can B1 row. All are kept under the fixed-final rule.
- A paired run took a median of 1.25 times as long at 288.

Normal validation at 288, mean over the three seeds:

| Category | Variant | Pair MSE | Score spread | Synthetic hinge |
|---|---|---:|---:|---:|
| can | B0 | 3.36 | 0.344 | 0.00744 |
| can | B1 | 0.109 | 0.964 | 1.37 |
| can | B2 1.0 | 0.0107 | 0.185 | 0.00249 |
| fabric | B0 | 0.217 | 0.259 | 0.00271 |
| fabric | B1 | 0.0181 | 0.292 | 0.0981 |
| fabric | B2 1.0 | 0.00958 | 0.202 | 0.026 |
| fruit_jelly | B0 | 8.85 | 0.424 | 0.0377 |
| fruit_jelly | B1 | 0.079 | 0.467 | 0.0199 |
| fruit_jelly | B2 1.0 | 0.0241 | 0.289 | 0.0475 |
| rice | B0 | 0.27 | 0.257 | 0.00222 |
| rice | B1 | 0.0216 | 0.267 | 0.000973 |
| rice | B2 1.0 | 0.00568 | 0.179 | 0.00836 |
| sheet_metal | B0 | 0.124 | 0.352 | 0.0247 |
| sheet_metal | B1 | 0.0225 | 0.338 | 0.0249 |
| sheet_metal | B2 1.0 | 0.0129 | 0.291 | 0.0282 |
| vial | B0 | 8.12 | 0.49 | 0.021 |
| vial | B1 | 0.102 | 0.503 | 0.0178 |
| vial | B2 1.0 | 0.018 | 0.349 | 0.295 |
| wallplugs | B0 | 0.15 | 0.334 | 0.0123 |
| wallplugs | B1 | 0.0179 | 0.325 | 0.00637 |
| wallplugs | B2 1.0 | 0.00841 | 0.251 | 0.00931 |
| walnuts | B0 | 0.119 | 0.286 | 0.005 |
| walnuts | B1 | 0.0152 | 0.278 | 0.0044 |
| walnuts | B2 1.0 | 0.0055 | 0.213 | 0.00415 |

## Limits

- This page reports normal-validation diagnostics. The completed public [image evaluation](ad2-test-evaluation.md) and [native localization](ad2-native-public-results.md) are separate.
- The full-run rule was written after the seed-3 search failed and after the 1.0 runs were known. It keeps the original safeguards, and its outcome selects no weight.
- sheet_metal has 19 normal-validation images, so its threshold and diagnostics rest on 19 images. Reporting gives macro averages over all eight categories and over the seven without sheet_metal.
- The 288 runs are in a separate release.

## Artifacts

The [study release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad2-study-2026-10-31) has the trained heads for the 72 paired runs (B2 at 1.0) and the 48 B2 runs at 0.01 and 0.1, the 24 single-view checkpoints, normal-validation outputs, run metadata, the weight search and the run audit. Dataset images and pretrained backbone weights are not included.

The [288 release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad2-288-2026-11-09) has the trained heads, normal-validation outputs, run metadata and audit for the 96 runs at 288 × 288. Their full training checkpoints and raw logs were not kept.
