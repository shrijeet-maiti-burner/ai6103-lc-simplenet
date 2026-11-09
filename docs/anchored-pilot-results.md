# fixed-teacher screening result

8 november 2026. the [fixed candidate](anchored-pilot.md) failed its predeclared screen on both training seeds. it is not promoted. confirmation inputs remain closed, and no real test images were scored for this follow-up. the original lighting-robustness goal remains unmet.

## complete comparison

all eight categories, seeds 0/1, five arms and 12,800 predictions completed. each category uses 32 fit, 32 calibration and eight development parents. four corner relightings reuse each parent. reported rates are macro means over categories and seeds, in percent. recall here measures procedural changes, not industrial defects.

| arm | clean fpr | clean synthetic recall | corner fpr | corner synthetic recall |
|---|---:|---:|---:|---:|
| frozen teacher | 6.25 | 69.79 | 42.97 | 80.34 |
| identity continuation | 10.94 | 77.60 | 50.39 | 87.70 |
| augmentation | 7.81 | 75.52 | 20.51 | 78.65 |
| anchor | 8.59 | 71.88 | 16.02 | 73.05 |
| candidate | 12.50 | 73.18 | 15.82 | 72.66 |

the candidate reduces corner false alarms relative to augmentation, but also loses synthetic recall and increases clean false alarms. adding consistency to the anchor changes mean corner fpr by only -0.20 points and recall by -0.39 points. the anchor control cannot substitute for a failed candidate.

| seed | corner fpr change vs augmentation | corner recall change vs augmentation | clean fpr change vs augmentation | categories with lower corner fpr | qualified |
|---|---:|---:|---:|---:|---|
| 0 | -6.25 points | -8.85 points | +4.69 points | 5/8 | no |
| 1 | -3.13 points | -3.13 points | +4.69 points | 3/8 | no |

both seeds fail the gaussian-separation safeguard as well. candidate/augmentation calibration hinge ratios exceed the fixed 1.25 limit for fabric seed 0 (2.40), fabric seed 1 (1.29), sheet_metal seed 1 (2.80) and walnuts seed 1 (1.47). the patch-standard-deviation floor passes; low score variance alone does not explain this failure.

five of the 960 procedural-change cases per seed are erased by image content or relighting. they remain in the denominator. seeds and relightings reuse parents; this screen does not provide independent real-defect observations or establish that every teacher-anchoring method must fail.

## verification and limits

- 325 cpu tests passed in github ci. the actual-size cuda smoke passed zero-weight update equivalence, frozen-teacher checks, checkpoint prediction replay and optimizer continuation, all bit-exact.
- a separate script reconstructed all 160 metric rows from saved predictions, recalculated thresholds and auroc, and reproduced the failed decision using exact count fractions.
- all 12,800 evaluation scores replayed bit-exactly from the 80 saved heads. all calibration maxima and normal-calibration safeguards also replayed exactly. checkpoint hashes agree between the replay and separate recount receipts.
- an actual confirmation-preparation invocation, supplied a nonexistent input path, returned the failed-screen refusal before creating a cache. no confirmation model-input decoding occurred.
- all 16 training receipts were retained. teacher fitting took 237.69 seconds in total; the four adaptation arms together took 1,044.78 seconds. full training/calibration/synthetic scoring took 1,431.28 seconds on the rtx 3060 laptop. preparation is excluded. this does not establish compute parity with the original study.

the prototype fits a new teacher on its 32 fit parents. development/confirmation exclude recorded local pilot and gradient-scoring parents, but earlier full-data models may have fitted those images; unrecorded teammate exposure is not claimed absent. this short, cached, 256px experiment cannot replace the original full-study reference or an independent real-lighting evaluation.

training source: `23a672a015c9634a6c3cb4977951a5959a15571c`. the upstream submodule stayed at `351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b`. later commits add separate audits and documentation without changing the frozen training files. raw artifacts remain local; the hashes below identify the retained records, not a publicly available artifact release.

| record | sha256 |
|---|---|
| protocol | `0b4e8b926d0dc96c999bfd77fb44e493059b5f31c605b9c82d57da0acfc54818` |
| complete metrics | `536bdfd862c1ed2937ea5eb9f5532a790b927a692cacc4624b9b4e0a15c807fd` |
| screening decision | `5e47d263e373e5e830ef3c7ec3754d3463c2bc35e229847c9c29aa3f510eed46` |
| separate recount | `b9f04c23ad1f95cc21ddbb9ff0127c5b701e621c248bdd186c14efd4c8512257` |
| full saved-head replay | `5e5e1b7f695dfbd87a4072dbdc583903f4d5faad665d65733f22e399089331fd` |

## next work

the [fit/calibration diagnosis](anchored-pilot-diagnosis.md) finds reduced gaussian separation on fit parents as well as calibration parents; it does not support treating overfitting as the established explanation.

retain this as a negative follow-up, without a coefficient sweep or confirmation scoring. b's original single-view portability check and d's authenticated private-evaluation access remain open. the existing dataset archive contains both private splits and matches its recorded mirror-lfs hash; original archive equality and server output fields remain separate checks. another loss experiment needs a distinct written hypothesis and an audited development/confirmation population before fitting. no setting may be chosen from public labels or private-server feedback.
