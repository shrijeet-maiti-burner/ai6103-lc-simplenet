# Project

## Question

Does paired relighting with patch-score consistency improve SimpleNet beyond lighting augmentation alone, while preserving defect recall and regular-light performance?

Scope: the revised [one-page proposal](../references/proposal.pdf), dated 2 October 2026 in the supplied chat. Working name: LC-SimpleNet. This is an empirical hypothesis; originality and improvements are unverified.

## Method contract

For a normal training image, apply shared spatial preprocessing once, then form `x` and `T(x)`. Initially `T` changes brightness and gamma only, before ImageNet normalization. Inspect clipping, shadows and fine details: mild parameter values do not by themselves prove defect preservation.

Let `E` be the frozen feature extractor and patch aggregation, `A` the trainable adapter, and `D` the discriminator. Upstream `D` gives larger scores to normal features. Inference uses anomaly scores `s = -D(A(E(x)))`.

For `N` corresponding normal patch positions:

```text
L_score = (1/N) sum_u [D(A(E(x)))_u - D(A(E(T(x))))_u]^2
L_total = L_discrimination + lambda_score * L_score
```

The sign cancels in the squared difference. Use raw patch scores before image aggregation, interpolation, smoothing or score normalization. Keep gradients through both adapter/discriminator branches; freeze only the extractor. The implementation helper checks identical shapes but cannot prove semantic alignment; the paired loader must preserve it.

Retain SimpleNet's normal-versus-Gaussian-feature-noise discrimination objective. For a raw normal score `d`, synthetic score `d_fake`, and margin `m`, its hinge terms are `mean(max(0, m-d)) + mean(max(0, m+d_fake))`. Average the discrimination losses over the two views. Synthetic anomalies remain training examples, not validation defects or evidence of real-defect recall.

Normal-only consistency can flatten useful scores. The discrimination objective is a safeguard to investigate, not a proof against collapse or loss of real-defect sensitivity. Log score spread, hinge losses, normal/synthetic separation and real-defect recall after freezing.

At inference use one image and the same network path. The design adds no inference branch; latency and memory parity still require measurement.

## Comparisons

| Variant | Views per optimizer step | Relighting | Consistency |
|---|---|---|---|
| B0 | original + identical original | no | 0 |
| B1 | original + relit | yes | 0 |
| B2 | original + relit | yes | positive weight selected on normal validation |

This is a proposed implementation detail, not wording supplied by the professor. B0 duplicates the original view to match image exposure and batch handling. Also reproduce the upstream single-view setting separately and document any difference. B1 and B2 must have identical data, augmentations, initial weights, seeds, noise policy, batch composition, BatchNorm handling, optimizer steps and tuning budgets; only the loss weight changes. Record both steps and wall time. Do not call matched steps equal compute.

Use one joint discriminator forward with a documented ordering for original/relit normal features and corresponding synthetic features. BatchNorm statistics depend on this composition. Compare B1/B2 under exactly the same implementation.

The primary contrast is B2 minus B1. B2 minus B0 alone cannot identify the benefit of consistency. Feature consistency, paired synthetic anomalies, additional lighting transforms and new architectures remain optional variants, not baseline scope.

## First feasibility gate

1. Obtain licensed data; select one AD reproduction category before examining its test results.
2. Replace test-based selection and normalization in a shared integration; preserve the original submodule.
3. Reproduce a one-category baseline with a fixed final checkpoint, saved maps, runtime and peak VRAM.
4. Inspect aligned brightness/gamma pairs and verify gradients reach the adapter and discriminator while the backbone remains frozen.
5. Run B0/B1/B2 smoke training, reload checkpoints, and verify repeatable raw scores. These are diagnostics, not benchmark results.
6. Measure AD 2 pilot cost and available labels/lighting metadata. Freeze selected categories and full three-seed budget before the main study.

A six-GB GPU is present locally, but full training feasibility is unmeasured. Do not select the full category scope from optimistic estimates.
