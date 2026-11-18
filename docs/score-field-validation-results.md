# Score-field validation

A low-frequency correction was tested on normal-validation maps from all 48 B0/B1 category/seed runs. Only k=4 passed the score-spread floor everywhere, but its worst relight FPR was 1.0 against the 0.05 target. This correction did not solve lighting sensitivity. No defect or test image was used.

## Method and selection

For a raw anomaly map S, form a smooth field U by adaptive average pooling to k×k followed by bilinear upsampling with `align_corners=False`, `antialias=False`. The residual is `R = S - U`; image score is `max(R)`. Each map is processed independently. The frozen checkpoints and their original scores/thresholds remain unchanged.

Candidates are k ∈ {4, 8, 16}. For each candidate/run, fit a new higher-q95 threshold on regular-light normal residual scores and use strict exceedance. Eligibility requires `std(R) / std(S) >= 0.5` on every run, with a 1e-12 comparison tolerance; uncorrected spread ≤1e-8 gives ratio zero. Among eligible sizes, minimize the worst absolute relight-FPR gap from 0.05. Ties keep the smaller k.

The matrix is eight categories × B0/B1 × seeds 0/1/2. Missing, duplicate or unexpected identities are rejected. Only `<category>/validation/good/<file>` IDs are accepted.

Seven relights are fixed: four brightness/gamma corners in {0.8, 1.2}², horizontal and vertical gains from 0.8 to 1.2 at pixel centers, and a centered Gaussian spotlight with sigma 0.25 and gain `0.8 + 0.4 * exp(-radius² / (2 * 0.25²))`. All transformed RGB values are clipped to [0, 1]. Their exact ordering and operations are in [score_field.py](../src/lc_simplenet/score_field.py).

The [protocol JSON](../configs/score-field-selection.json) is immutable. Its null selected size and `protocol_frozen_not_executed` status describe the pre-execution record, not an unfinished selector. The result is a separate file.

## Result

| Quantity | Recorded result |
|---|---:|
| Complete model identities | 48 |
| Selected eligible size | 4 |
| k=8 / k=16 runs below spread floor | 9 / 18 |
| k=4 residual/original spread ratio | 0.603–0.945 |
| Worst absolute relight-FPR gap | 0.95 |
| k=4 relight groups with FPR=1 | 54 / 336 |
| k=4 runs attaining the worst gap | 18 / 48 |
| Regular calibration FPR range | 0–0.0488 |

The spread floor retains variation; it does not establish defect sensitivity. Low regular-validation FPR follows from fitting the threshold on those samples. Neither observation offsets the relight failure.

## Reproduce

Generate one map file per fixed identity:

```bash
uv run --frozen --extra cuda python scripts/generate_validation_score_maps.py --run-dir results/study-artifacts/paired/can-B0-seed0 --root data/mvtec_ad2 --out results/validation-maps/can-B0-seed0.npz --device cuda
```

The generator checks regular-light image-score replay within 1e-3 before producing seven relit maps. It refuses test paths, an existing output or an output inside the released run. Repeat for all 48 identities, then pass their paths to `scripts/select_score_field.py --maps ... --out results/score-field-selection.json`.

The six-run can pilot is a separate A40/runtime gate in `run_can_score_field_pilot.py` and its Slurm wrapper. It is not the full 48-run job. For a new pilot, create a fresh source manifest with `scripts/write_source_provenance.py`; do not rewrite the historical receipt.

## Provenance and limits

SLURM job 53275 on 11 November generated 48 maps and exited 0:0 after 13:24. [The execution record](../provenance/score-field-validation-execution.json) identifies source `1f4de6afc1911a3ad44c03a33cbaeb6296aff66c`, upstream `351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b` and the locked environment.

Original selection JSON: 215,662 bytes, SHA-256 `179f30dd96b0408e0759ede3b5b4156042ad6a9878ca10c9c7d0d6e6733447bb`. Protocol SHA-256: `2dcf5d4c1771df60ec49a4296362e9f868c829e58d2dd8275e11bf57d23b3096`.

The original selection and 48 maps are external artifacts without a published download location. The checkout supports regenerating them from released heads and separately acquired normal images; it cannot recount the original result from source alone. No environment/runtime JSON or SLURM log was supplied for that full job, so its GPU, node and exact thread count remain unverified. No held-out defect/localization result is claimed.
