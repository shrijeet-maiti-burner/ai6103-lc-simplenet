# Method

The [revised proposal](../references/proposal.pdf) asks whether aligned relighting and patch-score consistency reduce lighting-induced false alarms while preserving defect recall. The original implementation is [SimpleNet](https://arxiv.org/abs/2303.15140), CVPR 2023.

For a normal image, apply shared spatial preprocessing once and form `x` and `T(x)`. Brightness and gamma are sampled independently from [0.8, 1.2]:

```text
T(x) = clip(brightness * x ** gamma, 0, 1)
```

Relighting occurs on raw RGB before ImageNet normalization. The frozen extractor and patch aggregation are `E`, the trainable adapter is `A`, and the discriminator is `D`. Larger raw discriminator scores mean normal; inference uses anomaly score `-D(A(E(x)))`.

```text
L_score = mean((D(A(E(x))) - D(A(E(T(x))))) ** 2)
L_total = L_discrimination + lambda_score * L_score
```

Consistency uses corresponding normal patches before aggregation, interpolation or smoothing. Both branches receive gradients. The original discrimination term is the mean normal hinge `max(0, margin - d)` plus synthetic hinge `max(0, margin + d_fake)`, averaged over both views. Gaussian feature noise generates synthetic training anomalies; it does not supply real-defect validation.

## Comparisons

| Arm | Pair | Consistency |
|---|---|---|
| B0 | original + identical original | none |
| B1 | original + relit | none |
| B2 | original + relit | normal patch-score MSE |

Duplicated B0 views match batch handling within this paired study. B1 and B2 share data, initialization, sampled transforms/noise, BatchNorm treatment, optimizers and update budget. B2 minus B1 isolates the added consistency term. The faithful [single-view reference](baseline-training.md) is a separate cohort with documented differences.

No AD 2 consistency weight passed all normal-only safeguards. We therefore report 0.01, 0.1 and 1.0 as ablations, without declaring a selected weight. See the [study results](ad2-study-results.md) for the unsuccessful search and deviations.

## Routed follow-up

The follow-up adds local/global synthetic anomaly supervision adapted from GLASS. R0 sends the normal consistency gradient through the adapter while using detached discriminator parameters and cloned buffers for the auxiliary forward. Classification still trains both modules. L0/L1/L2 separate synthesis, augmentation and full-gradient consistency from routing.

See the [objective](routed-consistency.md), [execution instructions](routed-reproduction.md) and [complete results](routed-results.md). The original AD 2 recall target fails; the separate blade mean-level benefit is limited by stronger controls, seed variation and background regressions. Neither score stability nor gradient isolation guarantees real-defect recall.

All deployed heads use one input image; training-only paired branches are absent at inference. Equal update counts do not establish equal compute.
