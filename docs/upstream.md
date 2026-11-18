# Upstream integration

[DonaldRR/SimpleNet](https://github.com/DonaldRR/SimpleNet/tree/351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b) is pinned at `351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b` and remains unchanged. Our integration lives in `src/lc_simplenet/`.

| Original behavior | Our treatment |
|---|---|
| `train()` evaluates test labels after every meta-epoch and selects the highest test AUROC | Train to a fixed final horizon; no test-based checkpoint selection |
| Evaluation normalizes from test-score/map extrema | Save raw scores and calibrate only on regular normal validation |
| Legacy PRO uses removed NumPy/pandas APIs, dilation and FPR limit 0.30 | Use a separately checked exact AD 2 evaluator with limit 0.05 |
| `run.sh` uses resize 329, crop 288, 40 meta-epochs × four discriminator passes | Retain that geometry for the AD reference; document the separate 256px AD 2 adaptation |
| Original loader lacks our persisted AD holdout and AD 2 folder handling | Use hash-pinned normal manifests and explicit split enforcement |
| Training-mode shape probe updates pretrained BatchNorm buffers | Faithful single-view reference retains the probe; paired controls all probe in evaluation mode |
| Prediction resizes an unused dense feature visualization | Reference helper omits that visualization and checks exact complete score/map parity |

The discriminator uses unbounded normality scores and BatchNorm. Pair order and forward modes are therefore part of the experiment. The paired runner makes one joint discriminator call; it does not replace raw normality scores with sigmoid probabilities.

The single-view reference differs from paired B0 in initialization buffers, batch composition and budget. Its recovered A40 state has a separate [complete replay audit](backbone-state.md). Do not attribute every cross-runner difference to the proposed loss.

The source ZIP preserves the original upstream code and licence, omitting only author-tracked bytecode caches. Its manifest records the omitted paths and the unchanged upstream commit. The original upstream training entry point is preserved for provenance; use our documented commands for the project's fixed-final protocol.

Paired illumination robustness has prior art, including [PIAD](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_PIAD_Pose_and_Illumination_agnostic_Anomaly_Detection_CVPR_2025_paper.html) and [paired well-lit/low-light anomaly detection](https://academic.oup.com/jcde/article/12/5/41/8119431). Our empirical study does not establish a broad novelty or SOTA claim.
