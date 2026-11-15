# Score-field validation result

Normal-validation selection only. k=4 is the only grid that passes the standard-deviation floor on all 48 runs. Its worst relight false-positive rate is 1.0 against a target of 0.05, a gap of 0.95. Anomaly sensitivity and held-out generalization have not been established. This is not a demonstrated detection improvement. No public-test or private-test image, label, or mask was used.

## Overview and motivation

Released B0 and B1 anomaly maps respond to illumination as well as to defects. A smooth change in brightness can move many patches together, and the image score used by this study is the maximum patch. A broad shift can therefore raise the image score even when no small region stands out.

The hypothesis is that a broad illumination effect is mostly a low-frequency component of the anomaly map. If that component is estimated and subtracted, a localized anomaly response should remain in the residual, while a map that only rose or fell smoothly should lose much of its illumination-driven score. The hypothesis is not a result. This document records one validation-only test of it. The test does not show that the residual is robust, and it does not show that defects are found more reliably.

The correction is applied to anomaly maps from frozen released heads. It does not retrain the adapter or the discriminator, and it does not change the released checkpoints.

## Method

Let \(S \in \mathbb{R}^{H \times W}\) be one anomaly map. Higher values are more anomalous. \(H\) and \(W\) are the map dimensions, not the RGB image dimensions. For these 256×256 heads the map grid is the model's patch grid.

A candidate scale is an integer \(k \in \{4, 8, 16\}\). The low-frequency field is built in three steps, matching `low_frequency_field` in `src/lc_simplenet/score_field.py`.

1. Downsample with adaptive average pooling to a \(k \times k\) grid. PyTorch cell \((i, j)\) covers rows \([\lfloor i H / k \rfloor, \lceil (i+1) H / k \rceil)\) and columns \([\lfloor j W / k \rfloor, \lceil (j+1) W / k \rceil)\). Each coarse value is the mean of \(S\) inside that cell. Call the coarse grid \(C \in \mathbb{R}^{k \times k}\).
2. Upsample \(C\) back to \(H \times W\) with bilinear interpolation, `align_corners=false` and `antialias=false`. Output pixel \((y, x)\) is placed at the source coordinate

\[
\left((y + 0.5) \frac{k}{H} - 0.5,\; (x + 0.5) \frac{k}{W} - 0.5\right)
\]

and that coordinate is clamped inside the coarse grid, so the sample is not read from outside it. Call the upsampled field \(U\).

3. The residual and the image score are

\[
R = S - U, \qquad s = \max_{y,x} R_{y,x}.
\]

\(U\) is the smooth field the hypothesis attributes to broad illumination. Subtracting it removes that field and leaves the part of \(S\) that the coarse grid cannot represent. The image score is still a maximum, so one strong residual patch is enough to flag the image. A map whose variation lives only in the coarse cells has a small residual and a small \(s\).

The same operations are applied independently to the regular-light map and to each relit map. Nothing in this step reads a label.

## Configuration selection protocol

The frozen rule is `configs/score-field-selection.json`. That file was not edited after the run. It still records `selected_downsample_size` null and `selection_status` `protocol_frozen_not_executed`. Its SHA-256 is `2dcf5d4c1771df60ec49a4296362e9f868c829e58d2dd8275e11bf57d23b3096`. The numbers below come from the separate result file, not from rewriting the protocol.

The matrix is the formal 256 study's B0 and B1 heads:

- Eight categories: can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts.
- Arms B0 and B1. B2 is not an input.
- Seeds 0, 1, and 2.
- These are 48 identities. `select_downsample_size` requires the set to match exactly. A missing run, a duplicate, or an unexpected category, seed, or arm is rejected before any \(k\) is chosen.

Each run supplies anomaly maps for normal validation images only. An image id must be `<category>/validation/good/<file>`. Paths whose components include `test`, `test_public`, `test_private`, `test_private_mixed`, or `ground_truth` are refused.

Each image is scored once under regular light and once under each of seven frozen relights. The four corners are

\[
\mathrm{clip}(b \cdot x^{\gamma}, 0, 1)
\]

with \((b, \gamma) \in \{(0.8, 0.8), (0.8, 1.2), (1.2, 0.8), (1.2, 1.2)\}\). The three spatial gains use pixel centers \(u = (x + 0.5) / W\) and \(v = (y + 0.5) / H\):

\[
\begin{aligned}
g_{\mathrm{horizontal}} &= 0.8 + 0.4 u, \\
g_{\mathrm{vertical}} &= 0.8 + 0.4 v, \\
g_{\mathrm{spotlight}} &= 0.8 + 0.4 \exp\left(-\frac{(u - 0.5)^2 + (v - 0.5)^2}{2 \cdot 0.25^2}\right).
\end{aligned}
\]

The spatial views are \(\mathrm{clip}(x \cdot g, 0, 1)\) with no extra gamma. Stored order: `corner_0.8_0.8`, `corner_0.8_1.2`, `corner_1.2_0.8`, `corner_1.2_1.2`, `horizontal`, `vertical`, `spotlight`.

For one run and one \(k\), the threshold is fit on the corrected regular-light image scores only:

\[
\tau = \text{higher empirical quantile of } \{s_i\} \text{ at } 1 - 0.05.
\]

An image is a false positive when its score is strictly greater than \(\tau\). Released scores and released thresholds are not rewritten. The target false-positive rate is 0.05. Regular-light FPR is recorded and is not part of the objective.

The standard-deviation check uses the population standard deviation, over every patch in the run, of the uncorrected regular-light maps and of their residuals. If the uncorrected standard deviation is at most \(10^{-8}\), the ratio is 0. Otherwise

\[
\rho = \frac{\mathrm{std}(R)}{\mathrm{std}(S)}.
\]

The run passes \(k\) when \(\rho + 10^{-12} \ge 0.5\). A size is eligible only when every one of the 48 runs passes. The floor stops a grid that erases too much of the map's variation, including variation a defect might occupy. It is not a measure of detection accuracy.

Among eligible sizes, choose the \(k\) whose worst absolute relight-FPR gap from 0.05 is smallest. The worst gap is the maximum of \(|\mathrm{FPR} - 0.05|\) over all 48 runs and all seven relights. A tie keeps the smaller \(k\). If no size is eligible, the selected size is null and this protocol does not authorize a later test-set score.

No other selection criterion is implemented.

## Implementation architecture

The residual is not applied while the maps are generated. Generation writes the anomaly maps. Selection reads those maps and applies \(R = S - U\).

| Piece | Responsibility |
|---|---|
| `src/lc_simplenet/score_field.py` | Residual, seven relights, validation-id checks, per-run FPR, and the 48-run selection rule. |
| `src/lc_simplenet/validation_maps.py` | Score regular light, require agreement with the released score within \(10^{-3}\), then score the seven relights. Write one compressed `.npz` outside the released run. |
| `scripts/generate_validation_score_maps.py` | One released head. Reads `heads.pt`, `run.json`, and `validation-scores.json`. Loads only the image ids in the score file. |
| `scripts/select_score_field.py` | Load completed `.npz` files and write the selection JSON. It does not open a dataset directory. |
| `configs/score-field-selection.json` | Frozen protocol text. It is not updated by the selector. |
| `scripts/run_can_score_field_pilot.py` | Six can runs only. Provenance, locked-runtime, backbone, encoder, and `heads.pt` checks, then an all-or-nothing regular-light gate before any can relight. It does not run the selector. |
| `scripts/run_can_score_field_pilot.sbatch` | NTU TC2 request for that can pilot: `MGPU-TC2`, account `stud`, 4 CPUs, 30G, `gpu:1`, one hour. The GPU name is checked in Python after allocation. There is no Slurm GPU constraint. |
| `tests/test_score_field.py` | Residual geometry, relight bounds, identity gate, and the selection rule on synthetic maps. |
| `tests/test_validation_maps.py` | The \(10^{-3}\) gate and the refusal to write a test path or an existing file. |
| `tests/test_can_score_field_pilot.py` | Pilot safeguards, including JSON path conversion. No GPU and no real head. |
| `provenance/can-pilot-code.json` and `.sha256` | Historical can-pilot source snapshot. The sidecar is the manifest's own SHA-256; later code changes deliberately fail those historical hashes. |
| `provenance/score-field-validation-execution.json` | Execution record for the 48-run validation job. It is not the protocol file. |

`generate_validation_score_maps.py` accepts:

```text
--run-dir   released run directory (required)
--root      dataset root (required)
--out       output .npz, must not already exist (required)
--device    cpu or cuda (default cpu)
--threads   torch thread count (default 4)
```

The dataset root must contain `<category>/validation/good/<file>` for every id in `validation-scores.json`. The script refuses an output inside the run directory.

`select_score_field.py` accepts:

```text
--maps   one or more .npz paths (required)
--out    selection JSON, must not already exist (required)
```

`run_can_score_field_pilot.py` accepts only `--device cuda` and requires `--threads 3`:

For a new execution, first commit source changes and generate a new manifest:

```bash
uv run --frozen --extra cpu python scripts/write_source_provenance.py --out results/score-field-source.json
```

Pass that file and its `.sha256` sidecar to the pilot. The Slurm wrapper requires explicit `SCORE_FIELD_PROVENANCE` and `SCORE_FIELD_PROVENANCE_SHA256` values. Do not overwrite the historical can-pilot manifest or present a new source snapshot as the source of job 53275.

```text
--source-root
--provenance
--provenance-sha256
--heads-root
--data-root
--out-dir          must be an empty directory
--weights          wide_resnet50_2-95faca4d.pth
--device cuda
--threads 3
```

There is no script that loops over all 48 runs. The can job script does not generate the other 42 categories. A full matrix is one generator invocation per identity, with `--device cuda` when a GPU is used, then one selector invocation. The generator's default is CPU and 4 threads. The can pilot is the only entry point that enforces the locked A40 stack, and it refuses any head other than the six can runs.

## external artifacts and recomputation

the 48 validation `.npz` maps and the original `selection.json` are external artifacts. neither is committed, and this repository records no download location for them. a fresh checkout is therefore insufficient to recompute the recorded result. the execution record is a summary of the supplied artifacts, not a self-contained rerun.

the original selection file has SHA-256 `179f30dd96b0408e0759ede3b5b4156042ad6a9878ca10c9c7d0d6e6733447bb` and contains 215,662 bytes. retain that file when comparing a rerun; this digest does not establish that its input maps are locally available.

after obtaining all 48 external validation maps, replace the placeholder directory below with their actual location. the shell expands the glob into individual filenames; the selector does not download or locate the artifacts.

```bash
export SCORE_FIELD_MAPS=/absolute/path/to/external-validation-maps
uv run --frozen --extra cuda python scripts/select_score_field.py \
  --maps "$SCORE_FIELD_MAPS"/*.npz \
  --out results/score-field-selection-recompute.json
```

That command needs the CUDA extra's interpreter only because `uv run --extra cuda` is the locked environment used for the study. The selector itself is array arithmetic. The output path must not already exist, and no path component may be a forbidden test name.

## Results

Counted from the transferred `selection.json`; its original machine-specific directory is omitted.

| Metric | Result |
|---|---:|
| Model identities | 48 |
| Selected k | 4 |
| k=4 eligible | Yes |
| k=8 eligible | No |
| k=16 eligible | No |
| k=8 runs below std floor | 9 |
| k=16 runs below std floor | 18 |
| k=4 residual std ratio | 0.603–0.945 |
| Worst relight FPR gap | 0.95 |
| Relight conditions with FPR=1 | 54 / 336 |
| Runs with worst-case gap 0.95 | 18 / 48 |

`public_test_used` is false. The exact k=4 ratios are 0.6033458729021821 to 0.9445323687139722. The table rounds them to three decimals. k=8 fails 9 runs and k=16 fails 18, all with \(\rho < 0.5\). Those two sizes also have worst gap 0.95. The std floor is what removes them, not a better relight FPR.

The gap 0.95 is \(|1.0 - 0.05|\). On those runs a frozen relight flags every normal validation image. At k=4 this happens in 18 of 48 runs, and 54 of the 336 relight conditions (48 runs times 7 relights) have FPR 1. The smallest per-run worst gap at k=4 is 0.0292, on walnuts B1 seed 0. That spread is reported so the worst case is not mistaken for the only case. The selection rule uses the worst case.

Regular-light FPR at k=4 stays between 0 and 0.0488. That is what the quantile rule does: \(\tau\) is chosen on the regular-light residual scores so that the regular-light validation FPR sits near the target. A low regular-light FPR after that fit is not evidence that a different illumination stays near 0.05. The relight columns are the check, and they do not.

## Interpretation

k=4 is the only candidate that keeps at least half of each run's uncorrected patch variation. Selecting it does not establish that k=4 improves anomaly detection. The eligibility floor says the residual still varies. It does not say the remaining variation marks defects, and the relight FPR says the image-level decision does not hold under the seven synthetic illuminations.

The relighting result is a robustness failure of this correction on normal validation images: the worst false-positive rate is 1.0 against a target of 0.05. No public or private test image was used. Anomaly sensitivity and held-out generalization have not been established.

## Reproducibility and provenance

SLURM job 53275 on 2026-11-11 exited `0:0` in `00:13:24` and generated 48 validation maps. The execution record is `provenance/score-field-validation-execution.json`.

| Item | Value |
|---|---|
| Selection JSON | External artifact named `selection.json`; not committed |
| Selection SHA-256 | `179f30dd96b0408e0759ede3b5b4156042ad6a9878ca10c9c7d0d6e6733447bb` |
| Maps | 48 external `.npz` files; no repository location or download link recorded |
| Provenance sidecar | `cfade5ec7459d3e1ffdb34253c4f2a57938d8f8974f691ed39b29ff77a7f6f2a` |
| Git HEAD at execution | `1f4de6afc1911a3ad44c03a33cbaeb6296aff66c` |
| SimpleNet | `351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b` |
| `uv.lock` SHA-256 | `3d3b9dd4995622d26fb3077f8000255f974a03c715ebe764a64eeedac3e53aa2` |

The transferred result directory contains `selection.json` and the 48 maps. It does not contain `environment.json` or `runtime.json`. No SLURM log was available locally. Node, partition, GPU name, clock times, and the exact thread count used by job 53275 are unavailable. The can pilot script is a separate six-run gate. It is not a record of job 53275.

The selection JSON and the raw maps stay outside Git.

## Limitations and next experiments

This run answers only the validation-only selection question. It does not answer whether the frozen residual helps on images the threshold was not fit to.

A later evaluation, if one is run, has to keep k=4 frozen. The first held-out check would need image-level detection, pixel-level localization where the evaluation setup supports it, defect sensitivity, and behavior under illumination changes that were not the seven synthetic maps. Those quantities are not measured here. Test labels and test lighting must not be used to pick a different \(k\) or to change the floor, the target rate, or the relight set.

No improvement on those checks is claimed, because those checks have not been run.
