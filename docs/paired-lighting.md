# Role C: paired lighting and consistency

## Current status

The completed AD and AD 2 training runner already implements the same relighting and objective as the Role C helpers. Athul reported numerical agreement on [PR #10](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/10#issuecomment-6552964794). `tests/test_role_c_runner_equivalence.py` now checks this directly against `PilotTrainer.train_step`, including loss components, discriminator gradients and BatchNorm buffers, plus relighting with identical sampled parameters at 256x256.

No replacement of the runner or retraining is needed for this PR. Preserve the code, environment and checkpoints recorded for the released runs. The helpers are a separately tested reference interface for the method. The runner's original `losses.py` remains unchanged; the additional helper reductions live in `paired_losses.py` so this PR does not alter that source hash.

## Confirmed settings

- AD 2: full-frame bilinear resize to **256x256**, no crop, identical for B0/B1/B2. Non-square inputs change aspect ratio under this existing policy.
- Brightness and gamma: independent uniform draws in **[0.8, 1.2]**, confirmed by C on 30 October.
- Form the pair after shared spatial preprocessing and before ImageNet normalization. Apply `clip(brightness * image**gamma, 0, 1)`, preserving pixel coordinates.
- B0 uses original + identical original; B1 uses original + relit without consistency; B2 uses the same pairing with consistency.
- Match data, initialization, noise policy, batch handling and update budget between B1 and B2.

The NumPy helper and the runner use different RNG implementations. Equal integer seeds alone do not generate identical pairs across these implementations. The equivalence test passes the same sampled parameters explicitly; do not swap generators mid-study.

See [study policy](../configs/ad2-study-policy.json) for the authoritative experiment setup and [helper config](../configs/paired-lighting.json) for the Role C interface settings.

## Objective and code

```text
Normal image -> shared 256x256 preprocessing -> original + relit
    -> identical normalization -> frozen extractor -> trainable adapter
    -> normal features and Gaussian-noise synthetic features
    -> one joint discriminator forward
    -> mean two-view hinge loss + lambda * normal patch-score MSE
```

The discriminator receives original normal, relit normal, original synthetic, and relit synthetic features, in that order. Larger raw scores mean normal. Consistency uses aligned normal patch scores before heatmap processing or score normalization, with gradients through both branches. Synthetic anomalies participate in discrimination only.

| File | Responsibility |
|---|---|
| `src/lc_simplenet/pairs.py` | Aligned copies, sampled lighting parameters and clipping fraction |
| `src/lc_simplenet/paired_training.py` | CHW normalization, joint discriminator forward and score diagnostics |
| `src/lc_simplenet/paired_losses.py` | Two-view hinge + normal-score consistency; FP32 reductions for half/bfloat inputs |
| `src/lc_simplenet/pilot.py` | Existing study runner, retained unchanged |

`normalized_pair` returns two CHW tensors. `paired_discriminator_loss` takes the discriminator, two adapted feature tensors, two corresponding noise tensors, margin and consistency weight. It returns differentiable loss terms and detached diagnostics. The caller owns noise generation and optimizer updates. Settings are validated before the stateful discriminator forward.

`clipped_fraction` counts channel values above one before clipping. The runner's `newly_clipped_fraction` additionally excludes pixels originally equal to one. These are different diagnostics and must not be conflated. Neither proves defect preservation.

## Weight outcome and interpretation

No AD 2 candidate passed all safeguards, including the full-length seed-mean check. Therefore there is **no selected B2 weight**. The test matrix reports 0.01, 0.1 and 1.0 as ablations against both B0 and B1. B0 versus B1 is the main comparison. The lambda=1.0 runs following the unsuccessful initial search remain a recorded deviation, not a successful selection.

The team's [normal-validation results](ad2-study-results.md) show that B1 lowers pair MSE relative to B0 in every category; B2 generally lowers it further but can narrow score spread or increase synthetic hinge. This is evidence about prediction consistency on normal images, not proof of preserved real-defect recall. Do not choose a weight or change safeguards after seeing test results.

## Scoring agreement

C agreed on [PR #8](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/8#issuecomment-6578117103) to D's clarified protocol:

- Fixed-final checkpoints; one threshold per model/category/seed calibrated on regular-light normal validation at a 5% target FPR, then frozen across lighting conditions. Report achieved validation FPR and sample size.
- Seeds 0–2: mean and sample SD per category; macro averages over all eight categories and seven without sheet_metal.
- Assess the 5-percentage-point recall tolerance on seed-mean macro recall separately for every lighting condition and both aggregates. Name each category exceeding the drop regardless of the macro result.
- B1 versus B0 is primary; each B2 weight is compared with both B0 and B1. No checkpoint, threshold or weight selection from test scores.
- Retain the 19-image sheet_metal caveat and prior exploratory fabric test exposure. Verify lighting-label mapping before per-condition reporting.

C's helper review does not block D's first end-to-end scoring check. Dataset transfer/availability is a separate A/D dependency.

## Verification

Verified on 2 November against current main: **168 tests passed**, including 13 runner-equivalence cases. Ruff, formatting, repository link/source checks and the environment doctor passed. The 256x256 architecture smoke check passed. These replace the earlier standalone-branch test counts.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_paired.py --output results/role-c/smoke-256.json
.venv/bin/python scripts/preview_lighting.py --image data/TRAINING_IMAGE.png
```

The architecture smoke check uses the pinned SimpleNet architecture, random weights and synthetic images at the configured 256x256 size. It verifies frozen-backbone gradients, trainable gradients, an optimizer update and exact saved-weight score restoration. `--image-size 64` is an optional smaller check, not the experiment setting. The preview without an image uses a clearly labeled synthetic texture. Generated outputs remain under ignored `results/`.

The equivalence tests use a small backend with BatchNorm to exercise the actual runner training step cheaply; they are not new dataset experiments. Completed study results belong to the team training runs referenced above.
