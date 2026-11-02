# Paired lighting: integration handoff

Role C supplies image-pair preparation and a paired-loss interface. The shared runner already implements the intended objective and has completed the locked AD 2 training study. These helpers are an independently checked interface, not a retroactive replacement for that run source. See the [study results](ad2-study-results.md).

## Input size

Use **256x256** for the team's AD 2 runs, with identical spatial preprocessing across B0, B1 and B2. This preserves compatibility with the team's existing training. `configs/paired-lighting.json` records `input_size: [256, 256]`. The pair helper does not resize: the data loader must supply 256x256 RGB images after the agreed resize/crop pipeline, then create both lighting views before normalization. Do not independently resize or crop the paired views.

The paired smoke check reads this size from the config by default. `--image-size 64` is a smaller diagnostic run, not the experiment setting. The formal AD 2 study uses full-frame 256x256 bilinear resizing without a crop; the faithful AD reference separately uses resize 329 and center crop 288.

## Image path

```text
Normal training image
        |
Shared crop / resize / spatial augmentation (once)
        |
HWC floating RGB in [0, 1]
        |
        +-- original copy ---------+
        |                          |
        +-- brightness and gamma --+
                                   |
                 Normalize both views identically
                                   |
                  Frozen extractor, trainable adapter
                                   |
              Add synthetic feature noise to each view
                                   |
                One joint discriminator forward
                                   |
        Mean two-view hinge loss + weighted normal-score MSE
```

`make_pair(image, brightness=..., gamma=...)` in `src/lc_simplenet/pairs.py` returns the original copy, relit image, parameters and fraction of channel values clipped above 1. `sample_pair` additionally takes an explicit NumPy generator and parameter ranges. Neither function crops, normalizes or selects final study settings.

The existing relighting rule is `clip(brightness * image**gamma, 0, 1)`. The helper preserves pixel coordinates, but this does not prove that small defects remain visible. Inspect real training examples with the data owner before freezing ranges. Brightness/gamma alone do not simulate every real lighting effect.

## Trainer interface

The data loader should return two views from the same spatially processed image, its sample identifier, brightness, gamma and clipping fraction. Convert HWC to CHW and apply the same ImageNet normalization to both views after pairing. Do not apply independent crops or flips afterward. Give workers distinct, recorded RNG streams; match their sampling order and seeds across B1/B2.

After feature extraction and adaptation, concatenate equal-size patch groups in this order:

1. Original normal features.
2. Relit normal features.
3. Synthetic anomalies from the original features.
4. Synthetic anomalies from the relit features.

Run the discriminator once and split its raw scores into those four groups. The trainer owns feature-noise generation and must use the same noise policy and batch handling across comparisons.

```python
from lc_simplenet.losses import paired_objective

# Each group contains the same number of corresponding patches.
normal_original, normal_relit, synthetic_original, synthetic_relit = scores.chunk(4)
terms = paired_objective(
    normal_original, normal_relit, synthetic_original, synthetic_relit,
    margin=margin, lambda_score=lambda_score,
)
terms["total"].backward()
```

The returned `total`, `discrimination` and `consistency` tensors retain their gradients. Detach only for logging. Larger raw scores mean normal. Consistency compares only corresponding normal patches, before heatmap processing or score normalization; both branches receive gradients. Freeze the extractor, not the adapter or discriminator.

| Variant | Pair | Consistency weight |
|---|---|---|
| B0 | Original and identical copy | 0 |
| B1 | Original and relit | 0 |
| B2 | Same pairing and randomness as B1 | Positive, selected on normal validation |

The implementation settings in `configs/paired-lighting.json` use independent uniform brightness and gamma in [0.8, 1.2], matching the team's reported bottle/grid settings. Check real training pairs before freezing these ranges across categories. The B2 weight remains undecided. Do not select it from test lighting or test labels.

## Checks completed on 26 October 2026

All 21 repository tests passed in the CPU environment, including the new checks for aligned pairs, unchanged inputs, clipping counts, repeatable sampling, zero identity consistency, lambda-zero hinge equivalence, invalid inputs, and gradient paths. A small toy network verifies a joint forward with BatchNorm and a frozen extractor. It is not a SimpleNet training run or a measured robustness result.

The brightness/gamma transform and basic raw-score MSE already existed in the shared repository. This contribution adds paired sampling with metadata, the combined two-view objective, integration guidance and additional tests.

## Remaining joint work

- With A: inspect licensed training images, confirm spatial preprocessing and agree the paired loader fields.
- With B: connect these functions to SimpleNet, preserve matching patch order, and run B0/B1/B2 smoke training and checkpoint reload checks.
- With D: inspect loss curves, raw-score spread, normal/synthetic separation and validation behavior before freezing settings. Measure real-defect recall only under the agreed evaluation protocol.
- Record GPU runtime and memory before committing to the full run budget. No improvement in false alarms or recall is established yet.

## Ready-to-connect interface (30 October)

`normalized_pair` in `src/lc_simplenet/paired_training.py` converts each pair to two ImageNet-normalized CHW tensors. Stack original tensors into one batch and relit tensors into another. The extractor and adapter must preserve image and patch order.

`paired_discriminator_loss` takes the discriminator, two adapted feature tensors and two corresponding noise tensors. It assembles the four groups, runs one forward, and returns loss terms plus detached score diagnostics. The caller retains control of noise generation, optimizer updates and the study configuration.

```python
from lc_simplenet.paired_training import paired_discriminator_loss

# original_batch and relit_batch: equally shaped normalized BCHW tensors.
# Freeze the backbone before this step; keep adapter/discriminator trainable.
features, _ = model._embed(torch.cat([original_batch, relit_batch]))
original_features, relit_features = model.pre_projection(features).chunk(2)
# Generate original_noise and relit_noise using the shared trainer's noise policy.
model.proj_opt.zero_grad()
model.dsc_opt.zero_grad()
terms, diagnostics = paired_discriminator_loss(
    model.discriminator, original_features, relit_features,
    original_noise, relit_noise,
    margin=model.dsc_margin, lambda_score=lambda_score,
)
terms["total"].backward()
model.proj_opt.step()
model.dsc_opt.step()
```

This example assumes `pre_proj > 0`, as in the checked architecture. The loop owner must set training mode, log detached losses and diagnostics, and preserve matched B1/B2 batches and random streams. Do not use the upstream test-based checkpoint selection for the study.

The diagnostics include normal and synthetic score means/spreads, their mean separation, and paired-score absolute difference. A falling consistency loss alone is insufficient: constant predictions could also produce it.

## Commands for review

From the repository root, with the CPU environment installed:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_paired.py --output results/role-c/smoke.json
.venv/bin/python scripts/preview_lighting.py
.venv/bin/python scripts/preview_lighting.py --image data/TRAINING_IMAGE.png
```

The preview without an image uses a synthetic texture with a narrow mark. It displays the original and four range corners, with clipping fractions. It demonstrates transforms only. Supply an actual training image to assess detail preservation; the default synthetic preview cannot establish this for MVTec.

The smoke script uses the pinned upstream SimpleNet architecture with a random-weight WideResNet50-2, synthetic inputs at the configured 256x256 size and CPU execution. It checks gradients, an optimizer step and exact restored scores after an in-memory checkpoint round trip. Its lambda=0.1 is a smoke setting, not the selected B2 weight. Generated files stay under ignored `results/`.

Unit verification on 30 October: all 25 tests passed, including matching B1/B2 BatchNorm statistics and raw-score diagnostics before an update. Full dataset training, pretrained behavior, GPU behavior and real-lighting robustness remain unverified here. A and B still need to connect the licensed data and shared training loop; C's components and executable checks are ready for that integration.

Architecture check on 30 October: passed on CPU. Adapter and discriminator gradients were nonzero, the backbone had no gradients, the adapter changed after its optimizer step, and reloaded adapter/discriminator weights reproduced evaluation scores exactly. This is implementation evidence only. Local output: `results/role-c/smoke.json`.

The shared runner applies `clip(b * x**gamma, 0, 1)` before normalization. Matching ranges alone does not imply matched numerical behavior: `pairs.py` uses NumPy transformations and a NumPy RNG, while the completed runner uses Torch transformations and independent Torch streams. Keep the original run source and provenance rather than substituting one sampler into finished experiments.

## Pre-publication review

Two independent code/method reviews checked the implementation. Fixed invalid objective settings mutating BatchNorm before rejection, and half-precision reduction overflow (including zero consistency weight). Added validation of public pair inputs. The final CPU suite passes all 34 tests. The CI CPU job also runs the paired SimpleNet smoke script.

Half/bfloat score reductions use FP32, while gradients remain connected to both branches. This protects the loss calculation; full mixed-precision training is not validated by the current CPU check.

## Agreed resolution and verification, 30 October

The team chose 256x256 to retain compatibility with training already underway. The Role C config now records this size; the pair/normalization helpers preserve the loader's dimensions. All variants must share the same loader resize/crop policy. Brightness and gamma remain in [0.8, 1.2].

All 35 unit tests passed, including an explicit configured-size alignment/normalization test. The actual SimpleNet architecture check also passed at 256x256: finite losses, frozen backbone, nonzero adapter/discriminator gradients, a successful optimizer update and exact score restoration after loading saved weights. The default paired CI smoke command now exercises this configured size.

## integration check, 2 november

`tests/test_runner_objective_parity.py` compares this helper with `PilotTrainer.train_step` on the same adapted features and independent per-view noise. It checks identity/relit controls, zero/positive consistency weights, two seeds, partial batches and a new epoch. Losses, gradients, BatchNorm buffers, optimizer updates and random-stream continuation are compared. Different float32 reduction groupings use explicit rounding tolerances; this is not a bitwise-equivalence or pretrained-accuracy claim.

The seed-3 AD 2 search admitted no consistency weight. All full-length B2 weights are disclosed ablations, and no weight passes the full-run safeguards. See the frozen [policy](../configs/ad2-study-policy.json). Historical test counts above describe their dated checks, not the current combined suite. No finished run is rerouted through the helper, and `pilot.py` remains unchanged by this parity check.
