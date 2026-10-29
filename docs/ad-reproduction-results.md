# MVTec AD single-view reproduction

Single-view SimpleNet (upstream `351a2b8`) through our wrapper at `22fed6a`, all 15 MVTec AD categories, seeds 0, 1 and 2: 45 runs. Each number is the fixed final checkpoint, meta-epoch 40. Every run also scored the test set after each meta-epoch so we can compare with upstream's checkpoint rule. That trajectory is a disclosed diagnostic and selects nothing. Our first test exposure was 27 October, after the settings were frozen.

## Recipe

Same as the [single-view baseline](baseline-training.md): WRN50-2 layers 2 and 3, resize 329 and center crop 288, batch 8, 40 meta-epochs of 4 discriminator epochs, noise SD 0.015, one A40. Two differences from the paper:

- We train on 80% of `train/good`; the other 20% is the normal validation holdout. Upstream trains on all of it.
- Upstream `SimpleNet.train` keeps the epoch with the best test image AUROC. We keep the last epoch.

Image AUROC uses the max patch score. Pixel AUROC is computed inside the 288 crop with masks resized nearest and cropped the same way, as upstream does. Paper values are Table 1 of arXiv:2303.15140v2.

## Results

Image / pixel AUROC (%). Fixed final is mean ± SD over the three seeds. "Best test epoch" is what upstream's rule would report: per seed, the epoch with the highest test image AUROC (ties broken by pixel AUROC), then averaged.

| Category | Paper | Best test epoch | Fixed final | Final image per seed |
|---|---|---|---|---|
| carpet | 99.7 / 98.2 | 99.40 / 98.00 | 99.22 ± 0.37 / 97.93 ± 0.13 | 99.44, 98.80, 99.44 |
| grid | 99.7 / 98.8 | 99.47 / 98.60 | 72.07 ± 25.09 / 68.62 ± 27.42 | 69.92, 48.12, 98.16 |
| leather | 100.0 / 99.2 | 100.00 / 99.14 | 100.00 ± 0.00 / 98.80 ± 0.04 | 100.00, 100.00, 100.00 |
| tile | 99.8 / 97.0 | 99.82 / 93.71 | 99.03 ± 0.56 / 96.97 ± 0.46 | 98.88, 99.64, 98.56 |
| wood | 100.0 / 94.5 | 100.00 / 93.25 | 99.42 ± 0.40 / 93.34 ± 0.59 | 99.82, 99.04, 99.39 |
| bottle | 100.0 / 98.0 | 100.00 / 98.10 | 99.97 ± 0.05 / 98.05 ± 0.07 | 99.92, 100.00, 100.00 |
| cable | 99.9 / 97.6 | 99.34 / 97.23 | 98.73 ± 0.41 / 97.00 ± 0.32 | 99.16, 98.67, 98.35 |
| capsule | 97.7 / 98.9 | 97.83 / 98.87 | 97.17 ± 0.46 / 98.84 ± 0.03 | 97.21, 97.61, 96.69 |
| hazelnut | 100.0 / 97.9 | 99.96 / 97.84 | 99.89 ± 0.11 / 97.93 ± 0.10 | 99.79, 99.89, 100.00 |
| metal_nut | 100.0 / 98.8 | 99.97 / 98.74 | 99.90 ± 0.13 / 98.57 ± 0.02 | 99.95, 99.76, 100.00 |
| pill | 99.0 / 98.6 | 98.33 / 98.41 | 97.74 ± 0.59 / 98.41 ± 0.12 | 98.42, 97.35, 97.44 |
| screw | 98.2 / 99.3 | 97.20 / 98.94 | 95.75 ± 0.56 / 99.07 ± 0.08 | 96.39, 95.51, 95.35 |
| toothbrush | 99.7 / 98.5 | 99.81 / 98.50 | 98.80 ± 0.16 / 98.51 ± 0.03 | 98.89, 98.89, 98.61 |
| transistor | 100.0 / 97.6 | 99.94 / 96.62 | 99.74 ± 0.46 / 96.49 ± 0.31 | 100.00, 99.21, 100.00 |
| zipper | 99.9 / 98.9 | 99.83 / 98.86 | 99.66 ± 0.11 / 98.91 ± 0.02 | 99.53, 99.74, 99.71 |

Macro averages. The fixed-final SD is across the three per-seed macros.

| | Paper | Best test epoch | Fixed final |
|---|---|---|---|
| All 15 | 99.57 / 98.12 | 99.39 / 97.65 | 97.14 ± 1.65 / 95.83 ± 1.85 |
| 14 without grid | 99.56 / 98.07 | 99.39 / 97.59 | 98.93 ± 0.15 / 97.77 ± 0.03 |

The paper's own average is 99.6 / 98.1, which matches the first row.

## What the numbers say

- 11 of 15 categories are within 1 image point of the paper at the fixed final epoch. Pixel AUROC is within 0.6 points everywhere except grid, wood (−1.16) and transistor (−1.11).
- Grid is the only failure. Two of three seeds collapse in the last meta-epochs (69.92 and 48.12), and upstream's rule hides it by picking an earlier epoch (99.47). The cause and the training-loss jump are in the grid diagnostics of #4.
- Without grid, upstream's test-epoch rule is worth about 0.46 image points on average (99.39 against 98.93). The epochs it would pick range from 1 to 39.
- Screw, pill and cable stay below the paper even with test-epoch selection (−1.00, −0.67 and −0.56 image points). One untested explanation is that we train on 20% fewer images; seeds and library versions are other candidates.
- Tile shows that picking by image AUROC can cost localisation: the best-test-epoch pixel AUROC is 93.71, against 96.97 at the final epoch.

## Checks

- All 42 new runs: commit `22fed6a` with a clean tree, upstream `351a2b8`, torch 2.6.0+cu124 on an A40, 40 trajectory rows whose last row equals the summary, one manifest per category, seeds 0 to 2 exactly once.
- Reloading each saved checkpoint reproduces its normal-validation scores and maps exactly (maximum difference 0). The fast per-epoch scorer matches upstream `_predict` exactly.
- Grid comes from the three 28 October trajectory runs, whose final epochs equal the original grid runs.
- Seed 2 and the bottle runs shared the GPU three at a time. A rerun of toothbrush seed 0 under sharing gave a byte-identical checkpoint and identical per-epoch scores, and the three bottle checkpoints are byte-identical to the `train_baseline.py` reruns in [baseline results](baseline-results.md).

## Cost

One run on an A40 with no other job: median 14.9 training minutes (4.8 for toothbrush to 24.8), peak VRAM 1.66 GiB including scoring, for the 26 serial runs (seeds 0 and 1 except bottle). Seed 2 and bottle times were measured while sharing the GPU and are not comparable.

## Run and outputs

[`scripts/reproduce_ad.py`](../scripts/reproduce_ad.py) trains one category and seed. It reads the normal manifest `data/manifests/ad-<category>.jsonl` and the MVTec AD test split:

```bash
uv run --frozen --extra cuda python scripts/reproduce_ad.py screw 0 results/ad-repro-screw-seed0
```

`LC_DATA`, `LC_MANIFESTS` and `LC_DEVICE` change the dataset root, manifest folder and device. The committed script is the one that produced these runs, reformatted for lint; its code parses to the same syntax tree and only the usage docstring changed. It has no CPU test.

Each run writes `checkpoint.pt`, `validation_normal.npz` (with the crop geometry), `trajectory.jsonl`, `scores.npz` and `summary.json`. The [reproduction release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad-reproduction-2026-10-29) has every run's summary, trajectory and per-epoch scores, the table and the checks. Checkpoints are not released.
