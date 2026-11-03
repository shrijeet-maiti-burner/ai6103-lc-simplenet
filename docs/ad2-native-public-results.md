# native public-subset results

all 120 fixed-final heads were evaluated on the same annotated public cohort as the [image results](ad2-public-rtx-results.md): eight categories, five methods and seeds 0/1/2. maps came from the separately labelled rtx3060 contingency, using the locked scoring stack. this is neither the primary a40 audit nor an official private-server result.

continuous input-size maps were restored bilinearly to unchanged native masks. no test normalization, morphology, fitted pixel threshold or downsampling was added. pixel auroc retains all pixels; au-pro gives each eight-connected defect region equal weight and integrates to fpr 0.05, normalized by 0.05. every public lighting capture is included within its category; these are not per-light localization results. a complete eight-category sweep contains 4,454,464,384 native pixels.

the frozen matrix used the retained prototype external-memory engine, sha256 `e2b26148854ca8d45791631976d38817d1bd69140ae9e933d6a4fcad6a048729`. the later shared cli is a separately checked integration; it is not retroactively assigned to these runs. exact prototype snapshots and source/input hashes are retained in the [private native review artifact](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad2-native-2026-11-03). source revisions and metadata redactions are labelled explicitly.

## all eight categories

percent mean +/- sample sd over three per-seed category macros:

| method | pixel auroc | au-pro-0.05 |
|---|---:|---:|
| b0 | 78.70 +/- 1.00 | 11.94 +/- 0.68 |
| b1 | 79.80 +/- 1.16 | 13.02 +/- 0.67 |
| b2, lambda 0.01 | 80.57 +/- 1.61 | 13.15 +/- 0.87 |
| b2, lambda 0.1 | 79.11 +/- 0.96 | 13.57 +/- 0.97 |
| b2, lambda 1 | 77.37 +/- 4.16 | 13.47 +/- 0.52 |

b1 minus b0 au-pro is +0.98, +1.21 and +1.04 percentage points in seeds 0/1/2. for b2 minus b1, the respective seed differences are -0.27/+0.51/+0.16 (lambda 0.01), +0.65/+0.10/+0.92 (0.1), and +1.52/+0.63/-0.80 (1). these are descriptive paired differences; three seeds and correlated captures do not justify a strong significance claim.

## without sheet metal

this scope was declared because sheet metal has only 19 normal calibration images. it remains a sensitivity analysis alongside the full result.

| method | pixel auroc | au-pro-0.05 |
|---|---:|---:|
| b0 | 78.02 +/- 1.23 | 12.58 +/- 0.73 |
| b1 | 79.29 +/- 1.45 | 13.79 +/- 0.59 |
| b2, lambda 0.01 | 80.20 +/- 1.93 | 13.92 +/- 0.85 |
| b2, lambda 0.1 | 78.54 +/- 1.31 | 14.42 +/- 1.00 |
| b2, lambda 1 | 76.46 +/- 4.91 | 14.29 +/- 0.68 |

## interpretation and replay

augmentation improves this localization endpoint while losing image recall at the frozen operating point. au-pro measures region coverage over pixel false-positive rates; image recall measures detected defective images at a normal-calibrated threshold. improvement in one does not cancel deterioration in the other. all b2 weights failed the existing normal-data safeguards and remain ablations. none is selected from this table, and no overall robustness or state-of-the-art improvement is established.

`scripts/build_native_report.py` checks complete category/method/seed coverage, metric-file hashes, unchanged evaluator hashes, scorer metadata, checkpoint/config identities and identical images/masks/pixel populations across heads. it then regenerates both tables and all per-seed/category values in `native-evidence.json`. the compact artifact supports aggregation replay without images, maps or weights. metric recomputation additionally requires the original saved maps and native dataset.

the engine preserves exact float32 ties, including signed zero. independent toy checks against the official utility and a 370-million-pixel sparse/external comparison verify the implementation; they do not certify predictive quality. resource details and the later shared integration are in [localization](localization.md).

archive sha256: `9e18c2f8808c775cc563258dcbd6f2cef73e92b6a8830edfe1418dc6a0e4448a` (945,985 bytes). a fresh release download passed crc and all 617 payload hashes; both tables and `native-evidence.json` replayed byte-for-byte.
