# why the anchor needs a different hypothesis

9 november 2026. this post-hoc check describes the [failed fixed pilot](anchored-pilot-results.md). it does not refit a model, change a threshold or reopen confirmation.

the check covers all eight categories, two seeds, five arms and both sets of 32 fit/calibration parents: 5,120 parent-arm records and 160 aggregates. each parent uses the same fixed feature-space gaussian perturbation across arms. heads run in eval mode with the original four-group batch shape. these are synthetic feature perturbations, not real defects.

means over categories, seeds and parents:

| arm | teacher-response mse, fit | mse, calibration | normal-minus-gaussian score, fit | score, calibration |
|---|---:|---:|---:|---:|
| frozen teacher | 0 | 0 | 0.8955 | 0.8670 |
| identity continuation | 0.5782 | 0.5822 | 0.8520 | 0.8300 |
| augmentation | 0.5506 | 0.5655 | 0.7616 | 0.7444 |
| anchor | 0.1906 | 0.2141 | 0.6764 | 0.6565 |
| candidate | 0.1980 | 0.2212 | 0.6734 | 0.6536 |

anchoring brings responses closer to the teacher but does not retain its mean gaussian separation. the smaller separation is already present on fit parents. the median calibration/fit anchor-mse ratio is about 1.06 for both anchored arms. these observations weaken a primarily fit-to-calibration overfitting explanation; they do not identify a causal mechanism or explain industrial defect recall.

there is also a basic limit to the objective. let the student's normal and gaussian response errors relative to the teacher be `e_n` and `e_g`, and let `L_anchor = E[e_n² + e_g²] / 2`. then the rms error in their score difference is at most `2 sqrt(L_anchor)`. an average fit loss of 0.198 permits a bound near 0.890, comparable to the teacher's mean separation of 0.896. this loose upper bound does not prove that separation was erased. it shows why a lower response mse is insufficient as a retention guarantee.

the next candidate therefore needs a distinct reason to preserve discrimination, with fixed controls and fresh audited development/confirmation roles. this result does not authorize a coefficient sweep, promotion of the anchor control or a real-defect claim.

## reproduce and limits

on the pilot cuda runtime, with the original run artifacts:

```powershell
python scripts/diagnose_anchor.py --run <pilot-run> --out results/anchor-diagnosis-001
```

the script verifies the frozen training source/policy, cache identities and checkpoint bindings. it reads only fit/calibration caches. calibration parents were unseen by this optimizer, but participated in prior development; they are not an untouched evaluation set. source integration happened after the initial diagnosis. its complete cuda execution reproduced all 5,120 parent-arm records and 160 aggregates with identical file hashes, and matched all 80 checkpoint hashes.

all 80 checkpoint hashes match the earlier complete replay receipt. a separate recount reproduced all 160 aggregates from the 5,120 records with maximum error 0.0. the retained private records are identified below; no public raw-artifact release is implied.

| record | sha256 |
|---|---|
| initial diagnosis protocol | `87ecf9fcb7478f01c31a6f58faa7c0d0e99d87756ba4aac7094d84b5a178c6c8` |
| parent-level records | `12d5862c3619a599cc1fc9fcd8a075b88a55bc704b263ffdaefc049ec19589a9` |
| aggregates | `1ff70101d05efe36937cfe5a378115814767e88b873ce4937ec579483b67c3de` |
