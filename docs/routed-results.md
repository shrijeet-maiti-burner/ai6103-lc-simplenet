# routed consistency: completed results

checked 18 november 2026.

**all 168 original AD 2 heads, target scoring and a separate raw-score recount are complete. the eight-category recall-preservation target does not pass.** the separate blade-lighting experiment meets its fixed point targets on three-seed averages against its reproduced B0; this is a limited result, not a general SimpleNet improvement.

the [aggregate evidence release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-results-2026-11-18) contains all category/domain and seed rows, comparisons, provenance hashes and a portable verifier. it excludes images, weights, per-image scores, personal notes and machine paths.

## method and controls

all comparisons retain their recorded preprocessing, budgets and normal-only thresholds. B0 is our fixed-budget reproduction, not the published paper's result. use these labels within each experiment; different datasets and budgets are not pooled.

| arm | role |
|---|---|
| B0 | original Gaussian/hinge training control |
| B1 | matched relighting, no consistency |
| B2 | relighting with the original coupled consistency term, weight 0.1 |
| L0 | local/global synthesis and supervision with identity pairs, no consistency |
| L1 | matched local/global supervision with relit pairs, no consistency |
| L2 | local/global supervision with coupled normal consistency |
| R0 | local/global supervision with normal consistency routed through the adapter only; no defect-lift loss |
| previous_B0 | B0 heads from the original 120-head AD 2 study (A40), kept as a second anchor |

global ascent and local focal supervision borrow from GLASS. the whole B0-to-R0 difference cannot be attributed to routing; R0 versus L1 isolates the added routed term more closely. the blade evidence also retains L3, R1, R2, P0 and P1 alternatives without selecting a new winner.

## original eight-category AD 2 result

all 1,084 exposed public images; eight categories; three seeds; seven newly trained arms. 80 source passes, batch 8, 256 x 256 full-frame inputs, all fixed-final checkpoints. the scorer replayed every normal-calibration patch exactly before applying the saved higher-q95 threshold with strict exceedance. sheet_metal has only 19 calibration normals; all eight categories remain primary.

rates and AUROC below are category-macro means across three seeds, in percent. AUROC is the threshold-free image AUROC within changed light. the downloadable CSV retains counts and each seed; the summary JSON also retains the seven-category sensitivity, both baseline anchors and paired intervals.

| arm | regular FPR | regular recall | changed-light FPR | changed-light recall | changed-light AUROC |
|---|---:|---:|---:|---:|---:|
| B0 | 4.03 | 27.78 | 46.66 | 66.92 | 63.21 |
| B1 | 7.78 | 26.67 | 45.45 | 61.68 | 61.80 |
| B2 | 6.51 | 26.11 | 42.69 | 60.94 | 63.22 |
| L0 | 5.65 | 24.17 | 20.02 | 37.62 | 67.85 |
| L1 | 5.93 | 23.89 | 21.53 | 38.56 | 67.64 |
| L2 | 6.00 | 25.28 | 21.08 | 38.86 | 68.19 |
| R0 | 6.69 | 26.11 | 21.43 | 39.62 | 68.33 |
| previous_B0 | 6.01 | 29.17 | 49.51 | 68.17 | 63.15 |

R0 versus matched B0:

- changed-light false alarms fall **25.23 percentage points**, passing the required 5-point drop.
- changed-light recall falls **27.30 points**, exceeding the allowed 5-point loss. paired 95% interval: **[-31.14, -23.56] points**.
- regular-light false alarms rise **2.66 points**, exceeding the allowed 2-point rise.
- regular-light recall falls **1.67 points**, within the 5-point allowance.

the original target also fails against previous_B0. removing sheet_metal does not reverse the recall failure. the large recall loss is already present in L0 without consistency; the result does not establish consistency-gradient conflict as its cause.

versus L1, R0 changes changed-light FPR by **-0.10 points** and recall by **+1.06 points**; both corresponding intervals include zero. the regular recall increase is **+2.22 points**. this is not evidence of a broad routing improvement.

threshold-free, R0 does not rank changed-light images worse than B0 on the category macro: AUROC **68.33** versus **63.21** (L0 67.85). can and fabric fall (48.6 to 41.0 and 64.5 to 47.4); the other six categories rise. the recall loss happens at the fixed normal-only operating point, not through a lower macro ranking. this does not permit a different threshold; thresholds stay normal-only.

B0, B1 and B2 here use the original study's recipe and seeds (80 source passes, batch 8, seeds 0-2, fixed-final checkpoints), retrained on L40S instead of A40 and alongside the new arms with a shared feature encoder. the cause of the gap to previous_B0 was not isolated; per seed, the new B0's changed-light FPR is 5.10, 2.53 and 0.91 points lower. the relighting effect is also smaller in this rerun: B1 minus B0 changed-light FPR is **-1.21 points** (per seed -1.09, -2.55, 0.00), against -6.0 in the original study. the recall loss repeats, **-5.24 points** against -7.5, and still exceeds the 5-point allowance. do not present the original 6-point false-alarm drop as stable across reruns.

## separate blade experiment

AeBAD_S contains one blade family. the fit uses 128 source normals, 64 normal calibration images, batch 4 and 160 source passes. 1,639 real test images cover illumination, same conditions, background and view changes. all 12 methods and all three seeds remain in the evidence (59,004 image decisions). these three-seed means are not ensemble predictions.

| arm | illumination FPR | illumination recall | same-condition FPR | same-condition recall | illumination AUROC |
|---|---:|---:|---:|---:|---:|
| B0 | 55.56 | 46.46 | 14.78 | 23.53 | 42.15 |
| B1 | 33.78 | 37.21 | 10.72 | 27.38 | 56.75 |
| B2 | 17.78 | 31.82 | 6.52 | 25.34 | 60.06 |
| L0 | 28.89 | 61.62 | 13.62 | 42.48 | 71.27 |
| L1 | 24.00 | 52.19 | 9.28 | 32.46 | 70.54 |
| L2 | 26.22 | 54.88 | 9.13 | 35.15 | 70.95 |
| L3 | 17.33 | 30.98 | 5.07 | 16.70 | 66.75 |
| R0 | 22.67 | 52.36 | 10.43 | 32.17 | 70.77 |
| R1 | 26.22 | 55.05 | 11.01 | 37.84 | 70.29 |
| R2 | 22.67 | 53.03 | 11.16 | 33.62 | 70.96 |
| P0 | 28.89 | 38.55 | 10.87 | 29.12 | 55.37 |
| P1 | 24.44 | 34.68 | 5.80 | 18.08 | 59.70 |

the illumination/same-condition B0 point targets pass on means: illumination false alarms **55.56% to 22.67%**, recall **46.46% to 52.36%**; same-condition false alarms **14.78% to 10.43%**, recall **23.53% to 32.17%**.

limits that belong beside that result:

- the paired 95% interval for illumination recall change is **[-2.69, +14.14] points**. it supports the five-point tolerance, not a strictly positive recall gain. resampling conditions on these three trained seeds and exact-RGB image clusters.
- individual seeds do not all pass: seed 1 loses **8.59 points** of illumination recall; seed 2 raises same-condition FPR **3.91 points**.
- background-change recall falls **67.61% to 39.62%**. background and view rows remain available in full.
- L0 retains higher illumination recall (**61.62%**). versus matched L1, routing changes illumination FPR **-1.33 points**, interval **[-4.00, +1.33]**; recall **+0.17 points**, interval **[-1.68, +2.19]**. the incremental routing gain is uncertain.
- B0 ranks illumination images below chance (AUROC **42.15**), so the B0 targets compare against a weak baseline. threshold-free, R0 (**70.77**), L0 (**71.27**) and L1 (**70.54**) have similar point estimates.
- the original synthetic source screen and full source confirmation both failed. local records describe R0 as fixed before real-defect feedback and admitted as an explicitly unqualified exploratory candidate. this timing is self-recorded, not independently timestamped pre-registration; those failures have not been reclassified as passes.

the local exploratory amendment names R0, forbids fallback selection and is dated 16 november at 02:16 UTC. the retained scorer log starts with the protocol hash before its first domain output. these records are from the same experiment workflow. publishing the protocol hash in the 18 november handoff, after the 16 november result drafts, does not independently establish its pre-test timing. describe the study as exploratory with self-recorded pre-specification.

## interpretation and verification

the supported claim is a blade-lighting benefit against our B0 on three-seed averages, with seed and background limitations. the original eight-category test shows lower lighting false alarms with an unacceptable recall loss. neither result establishes a general improvement over SimpleNet, a gain over every stronger control, private-server performance or SOTA.

separate audit scripts recounted all raw patch maxima and flags within the same assistant-led experiment workflow; this was not a second-person raw-data audit. [athul's review](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/40#pullrequestreview-5954465344) separately checked the published checksums, aggregate tables, deltas, seed failures and sensitivity results. that review did not rerun inference or recount the private raw patch scores. exact-RGB source/test overlap, duplicate-target and conflicting-label checks passed; physical-specimen independence is not established. no target threshold was refitted. the AD 2 public labels had prior project exposure, so this is a frozen transfer comparison rather than an untouched independent target.

download and extract the release archive, then run:

```bash
python scripts/verify_routed_results.py routed-results-2026-11-18
```

the archive also contains the same standalone verifier (`python verify_results.py .` from the extracted folder). it verifies the published file hashes, full aggregate matrices, count-derived rates, macro means and original fixed point gates. it does not rerun inference, replay raw patches or recompute bootstrap intervals; hashes of the separately recounted original decisions are retained for provenance. the immutable release snapshot retains the earlier "independent audit" wording; the distinction above applies to those records too.

the method code is in [PR #37](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/37). [PR #39](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/39) provides both the [AD 2 source snapshot](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-reproduction-source-2026-11-16) and the separate [blade execution source](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-blade-source-2026-11-18). the blade export retains byte-identical recorded workers and matches all 512 source-image/texture and 64 replay members. athul approved both PRs and this results handoff; all three were merged on 18 november. the [code delivery release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/code-submission-2026-11-18) records the packaged source and extraction checks. source/preparation, mechanics and extraction checks do not certify a fresh full-budget GPU run through either portable wrapper.

the [report/video review drafts](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/review-drafts-2026-11-16) predate these completed results. the report and narration still need both outcomes, the stronger controls and the limits integrated before final review. this release is evidence, not a completed course submission.

the recorded blade walkthrough's [renderer](../scripts/build_blade_gallery.py) and [HTML template](../demo/blade-gallery.html) are shared source. run `python scripts/build_blade_gallery.py --evidence-dir <recorded-evidence-directory> --out results/blade-gallery` against the original audited recording layout, native test packets and score files. aggregate CSVs alone do not supply image cards. the generated gallery contains dataset-derived thumbnails and stays local; the silent review clip is already available in the draft release. this renderer replays recorded evidence, not live inference, and does not accept arbitrary uploads.

for the final code zip, combine the reviewed changes and include both execution-source archives as described in [the source-package handoff](source-package.md#final-routed-handoff).

## decision identities

| record | sha256 |
|---|---|
| AD 2 frozen protocol | `751098d10d402f10b10f61d6e5b39a768a74dad04e59af0c9df917063dd57c08` |
| AD 2 separately recounted decision | `39a9d5a932438f02d87657415f18bd0343aae34e85b60b3d46e78946cb9c7be6` |
| blade recorded real-test protocol (self-recorded timing) | `75c262274d9fa4295a7416ee306ef66106e83b18c2b689c0efd36c98bda45af6` |
| blade separately recounted decision | `29430423067cabc302e8d9690ba74ec05cc60ff7f90f5f48a94b20a920bc9a16` |
