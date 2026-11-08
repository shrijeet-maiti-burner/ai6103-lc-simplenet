# fixed-teacher pilot

experimental follow-up to the [recovery plan](simplenet-recovery-plan.md). this changes the training loss within simplenet. there is no measured improvement yet, and a synthetic pass would not establish one on real defects.

## fixed comparison

the [policy](../configs/anchored-pilot-policy.json) fixes all coefficients, parent counts, steps, thresholds and numerical gates before fitting. each category/seed first fits one identity-pair teacher on 32 normal parents for 512 steps. four students copy that teacher and adapt for another 512 steps with fresh identical optimizers:

| arm | relighting | clean-response anchor | paired-score mse |
|---|---|---|---|
| identity | no | no | no |
| augmentation | yes | no | no |
| anchor | yes | yes | no |
| candidate | yes | yes | yes |

the fifth arm is the frozen teacher. every student uses the same parent order and post-adapter gaussian noise. batchnorm running statistics stay frozen during adaptation in every arm; affine parameters remain trainable. this is an explicit difference from the original full-study runner. the frozen teacher and student process the same four groups; the anchor uses only clean normal and clean-view gaussian scores. the teacher is absent at deployment.

the candidate must pass against all four references, separately for both screening seeds. the confirmation inputs remain unavailable to the model unless the complete screen passes. identity continuation is a matched diagnostic, not a replacement for the original full-study reference.

for a clean image, let `s` and `t` be the student's and teacher's raw normality scores. let `u` and `v` be their scores after adding the same gaussian vector to their respective adapter outputs. the added term is:

```text
anchor = 0.5 * (mse(s, stop_gradient(t)) + mse(u, stop_gradient(v)))
candidate_loss = two_view_hinge + 0.1 * mse(s_clean, s_relit) + anchor
```

the means include every patch. gaussian noise is shared between teacher and student, and across compared arms; the two image views have separate noise draws. teacher targets use its own adapter coordinates. preserving those responses does not constrain every genuine defect direction.

## run

use a cuda environment from [setup](setup.md), initialized upstream submodule, normal dataset and existing decoded-rgb audit. commit source before freezing a protocol. paths below are placeholders; inputs and generated artifacts stay local.

```powershell
python scripts/smoke_anchored.py --out results/anchored-smoke-001.json
python scripts/build_anchored_inputs.py --recorded-runs <prior-run-root> --rgb-audit-dir <audit-root> --out results/anchored-inputs-001
python scripts/anchored_pilot.py plan --out results/anchored-001 --normal-root <dataset-root> --reused-roles <prior-protocol.json> --exclusions results/anchored-inputs-001/exclusions.json --rgb-audit results/anchored-inputs-001/rgb-audit.json
python scripts/anchored_pilot.py prepare --out results/anchored-001 --normal-root <dataset-root>
python scripts/anchored_pilot.py train --out results/anchored-001
python scripts/audit_anchored_pilot.py --run results/anchored-001 --out results/anchored-001/independent-audit.json
python scripts/replay_anchored_pilot.py --run results/anchored-001 --out results/anchored-001/checkpoint-replay.json
```

only after a complete screening pass:

```powershell
python scripts/anchored_pilot.py prepare-confirmation --out results/anchored-001 --normal-root <dataset-root>
python scripts/anchored_pilot.py confirm --out results/anchored-001
```

the exposure inventory checks supplied recorded protocols. it cannot establish absence of unrecorded teammate activity or earlier full-data fitting. development and confirmation each use eight parents/category, disjoint by identity and decoded-rgb hash. existing fit/calibration roles are reused; the new teacher sees only the fit role. procedural changes include spots, scratches and donor blends, including erased changes. thresholds use the 95th percentile of 32 unmodified calibration-image maxima, with `method="higher"` and a strict `score > threshold` decision.

the four corner lights and three procedural changes reuse each parent. they are correlated observations, not additional independent images. both training seeds use the same parent split. this is a short mechanism screen, not an estimate of industrial defect recall.

retain protocols, hashes, checkpoints, optimizer states, all predictions and raw counts. source/cache changes, incomplete counts and failed replay are errors. failed qualification closes this candidate; it does not authorize a coefficient sweep, a control substitution or confirmation decoding. any qualifying result still needs frozen training and independent real-lighting evaluation. no public-test tuning.
