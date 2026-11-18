# submission checklist

working due date: **23 november 2026**, confirmed by shrijeet on 4 november. the exact portal cutoff is not in the supplied announcement. use the [current requirements](requirements.md); earlier 8/15 november targets are superseded.

the final-submission screenshot supplied on 16 november confirms the rules below. it supplies no new deadline or report page limit. the [shared review drafts](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/review-drafts-2026-11-16) are for team review; the report and earlier video do not yet include the [completed routed follow-ups](routed-results.md).

## report: 30 points

- [x] original eight-category R0 training, scoring and independent audit complete; original target failure and the separate blade mean-level result retained in [the results handoff](routed-results.md).
- [ ] incorporate both completed outcomes, stronger controls, per-seed failures and test-exposure limits into the final report.
- [ ] one designated member submits the report; the others do not submit duplicates.
- [ ] first page contains all five official names, matriculation numbers and ntu email addresses.
- [ ] original simplenet, our changes, matched controls and protocol deviations are explained precisely.
- [ ] every result has a traceable artifact; seed variability, category failures and test exposure are disclosed.
- [ ] modern reference results, if added, use a declared protocol and distinguish our contribution from reused components.
- [x] obsolete a40-pending wording is removed from the draft after checking the [completed image audit](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/18#issuecomment-6605059690). native rtx and official private-server evidence remain distinct.
- [ ] final pdf has at most six body pages in the unchanged aaai layout, plus references; every page is visually inspected.

## presentation: 10 points

- [ ] one clear argument: problem, method, decisive comparison, insight and limitation.
- [ ] diagrams and labels remain readable at normal playback size; the demo demonstrates a measured point.
- [ ] recorded video is playable and no longer than five minutes, including titles and ending. aim below the limit to leave margin.
- [ ] check the exact duration: each additional minute costs two points; the announcement gives videos longer than ten minutes as a zero-mark example.
- [ ] speech and playback are at their recorded speed. the announcement deducts five points for acceleration.
- [ ] one speaker or an ai voice is acceptable; having every member speak earns no extra credit. q&a is not required.
- [ ] final uploaded file is a video rather than a slide deck or audio file.
- [ ] replace older material when adding the new blade segment; appending the whole silent clip to the 4:30 draft would exceed five minutes. the final narration and complete playback still need review.

## code and final handoff

- [ ] submit a source zip or an instructor-accessible repository link.
- [ ] a zip includes the pinned simplenet submodule files, environment specification, licences and documented entry points. `git archive` alone omits submodule contents.
- [ ] a fresh extraction passes setup, the documented smoke run and artifact checks. state external dataset/weight acquisition steps honestly.
- [ ] our reproducibility check: review [pr37](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/37) and [pr39](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/39), then verify the submitted implementation against the final report. the [AD 2 snapshot](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-reproduction-source-2026-11-16) and [separate blade source](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-blade-source-2026-11-18) now expose both fixed training/scoring recipes. source, input-parity and mechanics checks have passed; fresh full-budget GPU executions through the portable wrappers and final clean-package checks remain open.
- [ ] exclude credentials, signed download urls, personal research, datasets, weights and generated run outputs from the source package.
- [x] the [complete aggregate results and provenance](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-results-2026-11-18) have a downloadable verifier and full category/domain/seed rows. this checks aggregates, not model inference or raw-score replay.
- [ ] verify private repository/release access for the instructor and include the final report's exact evidence references.
- [ ] gallery selection handles missing false positives/negatives explicitly, retains repeated-object identities and has retrievable rendering code and source hashes.
- [ ] all members review their recorded contributions; complete peer evaluation when it opens.
- [ ] check the portal's date/time and upload receipts for the three required deliverables before the cutoff.

code carries no standalone marks, but missing or non-running code can incur a penalty. the runnable-method check above is our way of checking the handoff against that requirement; it is not a newly announced scoring category. the announcement says peer evaluation will be set up later, so its availability remains unconfirmed.
