# Paired lighting and consistency

The [paired runner](../src/lc_simplenet/pilot.py) and the reference helpers implement aligned relighting and the same two-view objective. Tests compare losses, discriminator gradients, BatchNorm buffers and identical sampled photometric parameters.

## Inputs

AD 2 uses full-frame bilinear resize to 256×256 without cropping. Brightness and gamma are independent uniform draws in [0.8, 1.2], applied before ImageNet normalization:

```text
normal image → shared preprocessing → original + clip(b * image ** gamma, 0, 1)
             → normalization → frozen extractor → trainable adapter
             → normal and Gaussian-noise synthetic features
             → joint discriminator → hinge loss + lambda * normal score MSE
```

B0 uses identity pairs; B1 uses relit pairs without consistency; B2 adds consistency. All other controls match. NumPy and Torch RNGs do not generate identical pairs from equal integer seeds; equivalence checks pass the same sampled parameters explicitly.

See the [study policy](../configs/ad2-study-policy.json) and [helper configuration](../configs/paired-lighting.json).

## Implementation

| File | Responsibility |
|---|---|
| `pairs.py` | Aligned copies, sampled parameters and clipping fraction |
| `paired_training.py` | CHW normalization, joint discriminator forward and diagnostics |
| `paired_losses.py` | Two-view hinge and normal-score MSE; FP32 reductions for half/bfloat inputs |
| `pilot.py` | Study trainer, RNGs, optimizers and checkpoints |
| `losses.py` | Original study reductions, retained for recorded-run compatibility |

The discriminator batch contains original normal, relit normal, original synthetic and relit synthetic features, in that order. Larger raw scores mean normal. Consistency uses aligned normal patches; synthetic features participate only in discrimination. `paired_discriminator_loss` validates settings before its stateful forward and returns differentiable terms plus detached diagnostics. The caller supplies noise and performs optimizer updates.

`clipped_fraction` counts values above one before clipping. The runner's `newly_clipped_fraction` excludes values already equal to one. Neither diagnostic establishes defect preservation.

## Results and checks

No AD 2 weight passed all normal-only safeguards. The 0.01, 0.1 and 1.0 test rows are ablations; there is no selected B2 weight. The unsuccessful initial search and later lambda-1.0 runs remain documented in the [study results](ad2-study-results.md). Normal score stability does not establish real-defect recall; the [public evaluation](ad2-test-evaluation.md) reports that loss.

```bash
uv run --frozen --extra cpu python scripts/smoke_paired.py
uv run --frozen --extra cpu python scripts/preview_lighting.py --image data/TRAINING_IMAGE.png
```

The smoke check uses random weights and synthetic 256×256 inputs. It checks gradients, an optimizer update and exact checkpoint restoration. The optional `--image-size 64` is a smaller software check. A supplied preview image receives the training resize before relighting; omitting it creates an explicitly synthetic illustration. Outputs remain under ignored `results/`.
