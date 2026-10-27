# AD training policy

The [predeclared policy](../configs/ad-study-policy.json) bounds the AD component to six GPU-hours, three seeds and a complete B0/B1/B2 matrix. AD 2 feasibility and real-lighting evaluation remain separate; this plan cannot establish the proposal's real-lighting claim by itself.

## Cost and scope

Profile all AD categories using normal data only. Category choice reads image counts, dataset category type and measured time/memory, never validation anomaly scores or test results. Bottle is retained from the reproduction. Add the cheapest texture with at least 20 normal-validation images if the complete matrix fits. Include a 25% step-time margin and initialization/validation overhead, and reserve time for profiling and the normal-only weight search.

Use batch 8, resize 329 and center crop 288 throughout the paired controls. Prefer 80 source passes; the predeclared 40-pass fallback permits the object-plus-texture matrix under the budget. If both scopes fail, retain the longest feasible bottle-only matrix. Select the horizon before weight search. At 80 source passes, two views give the single-view reference's 160 input-view passes, but optimizer updates and measured compute still differ. At 40 passes the exposure is halved. All paired methods share the selected horizon.

## Normal-only weight choice

Before the main seeds `[0, 1, 2]`, use tuning seed 3 and 256 updates per trial. Candidate consistency weights are `[0.01, 0.1, 1.0]`; brightness/gamma ranges stay fixed at `[0.8, 1.2]`. Each category gets one B1 run and one B2 run per candidate. B1 is shared by all candidates, since reruns at the same seed are identical.

Evaluate all normal-validation images at the four fixed range corners. Select one shared positive weight using category-macro relative B2/B1 patch-score MSE, retaining at least half B1's patch-score standard deviation and no more than 1.25 times its normal/synthetic hinge loss. Choose the smallest weight within 1% of the best eligible proxy. If none qualifies, stop rather than changing the safeguards after seeing the results.

These diagnostics can detect some forms of collapse but cannot establish defect recall or defect preservation. The normal-validation target FPR is fixed at 0.05 with the existing higher-quantile/strict-exceedance rule; report its sample-size limits. D handles test evaluation later. No model, threshold, category or setting may be selected from test images, labels, lighting or server responses.

## Evidence

Freeze categories, per-category manifests, horizon, selected weight, source revision and hardware before main training. Keep initial/final RNGs, optimizers, resolved configuration, weight/backbone fingerprints, numerical runtime flags, source hashes, raw scores/maps and reload evidence. Paired checkpoint continuation requires matching source/runtime, including the CLI and image-processing versions. Keep generated runs and personal cluster details outside Git.

## Provisional decision, 27 October

B fixed the first AD matrix before any weight-search result: bottle and grid, B0/B1/B2, seeds 0, 1 and 2, 80 source passes (1,680 updates for bottle, 2,160 for grid), batch 8, resize 329 and crop 288, on A40 GPUs. Grid is the cheapest eligible texture by measured paired cost. The weight search uses tuning seed 3, one B1 run per category and one B2 run per candidate. Brightness and gamma stay at the runner defaults, 0.8 to 1.2, until C confirms them; any change reruns B1/B2. The bottle and grid single-view references use the existing fixed-final baseline. Results are in [the study results](ad-study-results.md).
