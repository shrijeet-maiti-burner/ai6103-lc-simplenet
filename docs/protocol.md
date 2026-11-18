# Evaluation protocol

Use the [normal-data contract](data-contract.md). Preserve data membership, preprocessing and source identity before training. The completed studies are recorded in the [evidence index](report-evidence.md); their settings are not selected retrospectively from test outcomes.

## Data and training

MVTec AD uses a fixed 20% normal validation holdout, grouping exact decoded-RGB duplicates. AD 2 uses its official normal train/validation folders and a separate per-category model. No private labels are inferred.

The paired AD 2 study uses eight categories, 256×256 full-frame bilinear resize, batch eight, 80 source passes, seeds 0/1/2 and fixed-final checkpoints. Match initialization, data order, photometric/noise streams, forward modes and optimization within a comparison. The [single-view reference](baseline-training.md) and [blade follow-up](routed-blade-reproduction.md) have separate budgets and geometry.

Use training and regular-light normal validation for settings. No checkpoint, weight, category, threshold or postprocessing may be selected from public labels, test lighting or server feedback. Prior exploratory fabric-test exposure and later public-reference exposure are disclosed in the results. A later follow-up does not make those public labels untouched.

## Frozen operating point

For every model/category/seed, fit the higher empirical 95% quantile of regular-light normal-validation image scores. An image is flagged only when `score > threshold`. Report calibration sample size and achieved FPR; small samples and ties make the quantile coarse. Calibration FPR is not held-out evidence.

Keep the same threshold across lighting conditions. Image score is the maximum raw negative-discriminator patch score.

```text
FPR    = flagged normal images / normal images
Recall = flagged defective images / defective images
```

Report counts as well as rates. Missing denominators yield unavailable values. Lower FPR alone is insufficient because an always-normal detector has zero FPR and zero recall.

The routed point targets, relative to B0, are a changed-light FPR reduction of at least 5 percentage points, recall loss of at most 5 points under changed and regular light, and regular FPR rise of at most 2 points. AD 2 retains both the matched B0 and previous frozen B0 anchors. All eight categories are primary; excluding sheet_metal, which has 19 calibration normals, is sensitivity analysis.

## Metrics

Report each seed, then unweighted category means within seed and mean/sample SD across seeds. Never substitute a pooled-image metric for a category macro. Lighting groups use [documented suffixes](ad2-lighting.md); a group absent from a category cannot silently become a complete eight-category macro.

Image AUROC measures ranking independently of the frozen threshold. Native pixel AUROC and AU-PRO are separate localization measures. AD 2 AU-PRO uses eight-connected regions, exact score ties, trapezoidal integration to FPR 0.05 and division by 0.05. Restore continuous full-frame maps with bilinear interpolation; evaluate cropped AD maps only against the corresponding resized/cropped masks. Do not stretch a crop across unseen pixels. See [localization](localization.md).

Three-seed SD describes training variation. Paired image intervals in the routed study condition on those fitted seeds and cluster exact RGB copies; they do not establish new-seed or physical-specimen independence. All domains, seed failures and stronger controls remain reported.

## Evidence

Retain configuration, source/upstream commits, dependency lock, input and checkpoint hashes, RNG state, runtime, complete raw scores/maps, calibration and metric code. A replay or software test establishes implementation behavior, not detector improvement.

Public-subset metrics are distinct from official private-server results, which have not been obtained. No latency or SOTA claim follows from the demo.

Sources: [proposal](../references/proposal.pdf), [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2), [versioned benchmark paper](https://arxiv.org/html/2503.21622v1).
