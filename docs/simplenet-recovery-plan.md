# simplenet recovery plan

8 november 2026. shrijeet asked to share this plan and start work. the original study stays frozen; this is a new follow-up. implementation has started, but the candidate has no measured improvement yet.

## goal

reduce false alarms under real lighting changes while preserving defect recall and regular-light performance. the [current public results](ad2-public-rtx-results.md) do not meet that goal: relighting reduces nonregular false alarms from 49.5% to 43.5%, but recall falls from 68.2% to 60.6%. localization improves, which remains a separate result.

aim for no recall regression. retain the previously agreed five-point tolerance in the reporting; do not loosen it after seeing results. a lower pair mse alone is not success.

## what we know

- the paired b0/b1/b2 matrix has valid saved-normal replay checks. the older faithful single-view ad2 reference has a separate reload/source-state problem; resolve it before claiming an improvement over that reference.
- [athul's validation-only projection experiment](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/issues/26#issuecomment-6719168403) failed its recall rule. its synthetic recall also failed to predict b1's real-public recall loss. synthetic tests can reject a candidate; passing them cannot establish defect retention.
- the current brightness/gamma model covers only part of the physical lighting variation. a generic gradient-conflict explanation and global intensity normalization are insufficient explanations.
- a stronger borrowed backbone is a reference. it does not establish an improvement to our simplenet method.

## next candidate

implementation and reproducible commands: [fixed-teacher pilot](anchored-pilot.md).

keep simplenet's frozen feature extractor, adapter, discriminator, discrimination objective and normal paired-score mse. add a training-only frozen baseline teacher that anchors the student's clean-normal and shared gaussian-corruption scores while the student adapts to relighting.

the hypothesis is that a fixed reference may limit loss of useful discrimination. it is not a guarantee: preserving synthetic responses need not preserve genuine defects. response preservation during adaptation is established work, including [learning without forgetting](https://arxiv.org/abs/1606.09282); do not claim that general principle is new.

the deployed student still takes one image. no teacher or extra relighting network is required at inference.

compare the same fitted teacher, student initialization and adaptation budget:

| arm | relighting | clean-response anchor | paired-score mse |
|---|---|---|---|
| identity continuation control | no | no | no |
| augmentation control | yes | no | no |
| anchor control | yes | yes | no |
| candidate | yes | yes | yes |

retain the frozen teacher as a reference. the candidate must be assessed against both controls; an anchor-only gain cannot be attributed to consistency.

## work checklist

### baseline and implementation

- [ ] resolve or explicitly bound the faithful single-view reload issue with b; do not relax replay tolerances.
- [ ] specify teacher/student tensor locations, fixed coefficients, forward modes, batchnorm behavior and noise sharing before fitting.
- [ ] use a teacher fitted only on pilot fit parents. a full-data checkpoint cannot support an unseen-parent claim on its own training images.
- [ ] test zero-weight control equivalence, frozen teacher parameters/buffers, gradients, continuation and prediction replay.
- [ ] keep the upstream submodule unchanged. put shared integration and tests in our package.
- [ ] record teacher cost and adaptation cost. matched steps are not a claim of matched compute with the old study.

### bounded pilot

- [ ] audit parent identities and previous model-scoring exposure before fixing fit, calibration, development and confirmation roles.
- [ ] fix a short eight-category, two-seed pilot, 256px geometry and one candidate before screening. keep the current augmentation family fixed to isolate the loss change.
- [ ] use normal data and fixed procedural changes only for development; never choose settings from public labels, test lighting or server feedback.
- [ ] write the numerical false-alarm, recall and category safeguards before training. require the candidate to qualify against both controls and the frozen teacher.
- [ ] use a separate confirmation role and training seed only if the complete screen passes. do not recycle exposed parents or repurpose the failed dinov3 candidate's sealed set.
- [ ] retain every attempt. a failed gate stops that candidate; no silent fallback or threshold change.

target: a go/no-go within 48-72 hours, subject to measured runtime. the next available experiment slot goes to this simplenet question.

### real evaluation

- [ ] check independent real-lighting data and official-server access now, including which outputs the server actually supplies.
- [ ] after a qualifying pilot, freeze all training, calibration and evaluation choices for eight categories and three seeds.
- [ ] compare with the original and relighting-only models. a weaker warm-start control cannot replace a stronger existing reference.
- [ ] retain normal-validation thresholds under every lighting condition. report category counts, false alarms, recall, auroc, localization and seed variation.
- [ ] obtain independent real-lighting evidence through the private evaluation or another previously unexamined dataset with a verified protocol. do not use its results for further tuning.
- [ ] label any once-only public follow-up as exposed. it cannot become an untouched test by changing its name.
- [ ] update the report and demo only after the result is supported. report regressions as well as gains.

## coordination

the [existing allocation](research/README.md) stays in place: shrijeet handles a/e and overall review, athul/umesh b, harsh c, aryaman d. these are proposed handoffs, not claims that anyone has completed them:

- a: parent/split audit and transform checks.
- b: replayable original reference and matched training runs.
- c: objective and control implementation, with source/configuration review.
- d: frozen evaluation and independent-test access.
- e: result-backed report, demo and presentation.

discussion: [#26](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/issues/26). datasets, weights, raw outputs and unrelated personal research stay out of git. sharing this plan does not publish the rest of the private research folder.
