# Evaluation protocol

Status: draft operational protocol implementing the revised proposal. Record final choices in [decisions](decisions.md) before testing.

## Data and selection

- MVTec AD: reproduce on selected categories. Derive a fixed normal validation holdout from training data; persist the file list and hash. Do not use defect labels for the split or selection.
- MVTec AD 2: train separate per-category models on AD 2's own normal training images. Use official normal validation images. AD and AD 2 contain different categories; this is not direct transfer.
- Keep all views of a source image in its split. Record source IDs and acquisition/lighting groups. Never infer private labels from names or image appearance.
- The AD 2 public test subset includes labeled examples across lighting conditions. Use documented condition metadata for local per-lighting counts. If that mapping is unavailable, report the limitation instead of inventing groups.
- Private AD 2 ground truth requires the official server. Local FPR/recall and server outputs are separate evidence; publish only metrics actually returned or computed from known labels.

Use training and regular-light normal validation data for settings. Fixed final training duration is the initial checkpoint policy; do not choose the best checkpoint on public test AUROC. Normal-only validation cannot directly optimize defect recall or F1. Predeclare a small search budget, proxy, candidate ranges and tie rule. Never select settings using public test labels, test lighting, or evaluation-server feedback. Disclose any accidental exposure.

Freeze categories, data lists, seeds, transforms, loss weight, preprocessing, optimizer schedule, duration, image aggregation, postprocessing, calibration and metric implementation before final evaluation. Store the frozen configuration and Git commit. Subsequent changes define a new study and must disclose prior test exposure.

## Operating point

For each trained model/category/seed, calibrate an image threshold on regular-light normal validation scores. The proposed helper uses the higher empirical `(1-alpha)` quantile and decision `score > threshold`; choose `alpha` before testing. Report validation sample size and achieved FPR. Small samples and ties constrain its resolution. This is not a statistical guarantee on future data.

Reuse that exact threshold for regular and changed lighting. Use raw anomaly scores or a single validation-fitted transform. Never normalize each test condition independently or fit calibration to test min/max values. Pixel thresholds, if used, require a separate declared rule; image and pixel operating points are not interchangeable.

For each lighting condition report:

```text
FPR    = false-positive normal images / all normal images
Recall = true-positive defective images / all defective images
```

Show numerators and denominators. Missing denominators produce an unavailable rate, not zero. Lower FPR alone is insufficient: predicting every image as normal gives zero FPR and zero recall.

## Metrics and uncertainty

Primary application evidence: changed-light FPR and defect recall at the frozen regular-light threshold, compared with B1, alongside regular-light preservation. Choose a tolerated recall drop and decision rule before testing. Describe uncertain or mixed outcomes as such.

Also report image AUROC, localization quality and per-category results. For AD 2 use the official AU-PRO with FPR integration limit 0.05 and documented normalization/interpolation. The upstream PRO helper is not equivalent: it uses 0.30, dilation, and an approximate threshold grid. Keep AD PRO-0.30 separate. Do not quietly substitute pixel AUROC for regional localization quality.

The private same-light versus mixed-light gap is informative, but the mixed set contains both seen and unseen conditions. It is not a pure unseen-light score. A smaller gap can also result from worse regular-light performance; show both endpoints.

Use three independent training seeds `[0, 1, 2]` for each main method/category, identical across variants. Share the same data split; record initialization, shuffle, augmentation and noise RNG states. Report per-seed results and mean/sample SD (`ddof=1`). Seed variation is not sampling uncertainty. With three runs, avoid strong significance claims. If bootstrapping images, preserve paired lighting/acquisition groups.

Report category macro averages and category-level counts separately; do not present pooled-image metrics as a macro average. Save normal and defective examples selected using a fixed rule, including failures. Use consistent heatmap scales and thresholds for comparison.

## Cost and evidence

Each run directory must contain resolved config, Git/upstream SHAs, lockfile hash, data-manifest hash, seed/RNG state, hardware/runtime versions, logs, checkpoint hash, raw scores, maps, calibration rule/threshold and metric version. Mark development, frozen-public and private-server results explicitly. Keep large files locally; commit concise reviewed tables with artifact references.

Profile training separately from inference. Use warmup, synchronized GPU timing, repeated measurements and batch size 1 for inference. Record image size, repetitions, peak allocated VRAM, parameter count and hardware. Exclude upload/UI time from network latency and describe preprocessing consistently.

Private submissions should be planned after freezing, with an access check and an archived submission hash/server response. Server access and available per-lighting outputs have not yet been tested.

Sources: [AD 2 dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2), [versioned benchmark paper](https://arxiv.org/html/2503.21622v1), and [selected proposal](../references/proposal.pdf). The proposal intentionally forbids test-based selection more strictly than the paper's optional public-test estimation workflow.
