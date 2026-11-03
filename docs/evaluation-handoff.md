# evaluation hand-off

the frozen study has 120 paired runs: b0, b1 and b2 at 0.01/0.1/1.0, eight categories and seeds 0/1/2. no weight is selected from test outcomes. [lighting definitions and the vial exception](ad2-lighting.md) are explicit.

## prepare locally

`scripts/prepare_study_artifacts.py` verifies every downloaded zip against the release's checksum manifests, then stages the paired run directories. `scripts/stage_ad2_public.py` verifies the pinned acquisition archive, stages original public images/masks only, and hardlinks existing normal train/validation images into the same root. it excludes annotated derivatives and private images. staging does not decode or score test pixels. keep all generated files under ignored `data/` or `results/`.

```powershell
uv run --frozen --extra cpu python scripts/build_public_conditions.py --root data/evaluation-ad2 --out results/evaluation/conditions.json
```

the inventory contains 1,084 original public images and 705 masks. vial has `shift_4`; retain that observed group and disclose its unverified physical setup. the public filenames, image-id hash and per-condition counts are saved with the mapping.

## score and aggregate

use d's `scripts/score_public_test.py` first on one released run with its saved validation-score guard. the agreed full scoring environment is the lock and a40; this hand-off does not silently change that choice. after the one-run check, score every fixed head without choosing a winner from its outputs.

```powershell
uv run --frozen --extra cuda python scripts/evaluate.py --runs results/study-artifacts/paired --out results/evaluation/final --conditions results/evaluation/conditions.json --categories can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts --seeds 0 1 2 --methods B0 B1 B2@0.01 B2@0.1 B2@1 --target-fpr 0.05 --recall-tolerance 0.05
```

`aggregate.json` computes unweighted category macros per seed, then mean/sample sd. it reports both all categories and without sheet_metal. the five-percentage-point recall rule is applied per condition, and every category-level violation is named. b2 is compared with both b0 and b1. no completed test data yields `not_evaluated`; a missing model/seed yields an incomplete macro. mismatched test image ids or labels across methods/seeds are rejected.

some physical groups do not exist in every category (fruit_jelly lacks shift_2/3; only vial has shift_4). the corresponding full-category macro is `not_applicable`, with absent categories listed. it is never silently computed over fewer categories and called macro-8.

the optional `--lighting-aggregates` additionally reports three explicitly defined pooled groups: `nonregular` (anything except regular), `exposure_changed` (under/overexposed), and `additional_light` (`shift_*`). these group definitions use metadata only; they do not select settings from outcomes. per-suffix rows remain available. pooled nonregular captures are not a pure unseen-light group, and shifted lamps differ physically across categories. record this additional analysis choice before scoring.

image operating points, pixel auroc and au-pro are distinct. d's image-score path is implemented. [native metrics and public-reference parity](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/17) are implemented in a separate review; the full native result matrix remains a completion gate. the public scorer's smoothed maps are at model-input size; restore continuous maps to native image size before comparing against native masks, or explicitly label a resized-mask analysis as a different geometry.

## provenance and prior exposure

retain thresholds, sample counts, raw scores/maps, model and scoring revisions, checkpoint hashes, runtime and exact image-id lists. disclose #9's earlier public-fabric evaluation on exploratory 100-step checkpoints. no subsequent metric may be used to choose thresholds, weights, checkpoints or categories. official private results require the evaluation server; public-subset tables are not private-benchmark results.

## current evidence

aryaman has completed acquisition and reported the primary a40 image matrix in [#18](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/18); no separate dataset hand-off is needed. all 150 published means/sds agree with the local matrix at their displayed precision. independent inspection of his raw artifacts and scoring-copy hashes remains pending. the [complete local rtx contingency](ad2-public-rtx-results.md) contains all 120 fixed heads and 16,260 predictions. it is independently recounted and separately labelled, with a material nonregular-light recall loss. do not select a weight from it or silently substitute it for a40/private-server evidence.
