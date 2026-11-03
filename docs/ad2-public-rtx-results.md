# frozen public-subset image results, 3 november 2026

separate local scoring contingency: rtx 3060 laptop, torch 2.6.0+cu124, source `b9e0a0357497037251653ef2e296fd932f003ed1`. aryaman has reported the primary a40 image evaluation in [#18](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/18). all 150 published means/sds match this run at the reported three-decimal rate precision; his raw artifacts and scoring-copy hashes have not been independently inspected. tc2 disk quota prevented local setup before scoring; no head was retrained or selected from these outcomes. this is the annotated public subset, not private-server evidence.

all 120 fixed-final heads (eight categories, five methods, three seeds) scored the same 1,084 images: 379 normal and 705 defective. total: 16,260 predictions. thresholds are higher empirical normal-validation 95th percentiles with strict `score > threshold`. every head passed its three saved-normal re-score checks. both declared category scopes and every weight are retained.

## image operating point

rates below are percentages: category macro within seed, then mean/sample sd over seeds 0/1/2. regular is the original light; nonregular pools exposure changes and additional lights, including vial's physically unverified shift_4. it is not a pure unseen-light population.

| scope | method | regular fpr | regular recall | nonregular fpr | nonregular recall |
|---|---|---:|---:|---:|---:|
| all 8 | b0 | 6.0 ± 2.5 | 29.2 ± 2.9 | 49.5 ± 4.1 | 68.2 ± 4.1 |
| all 8 | b1 | 6.4 ± 2.4 | 27.8 ± 1.3 | 43.5 ± 2.6 | 60.6 ± 1.6 |
| all 8 | b2@0.01 | 4.7 ± 2.1 | 26.4 ± 1.7 | 41.9 ± 3.6 | 60.4 ± 2.4 |
| all 8 | b2@0.1 | 7.5 ± 4.2 | 28.3 ± 0.8 | 41.5 ± 8.4 | 61.6 ± 2.6 |
| all 8 | b2@1 | 5.1 ± 1.9 | 26.4 ± 2.1 | 38.3 ± 5.5 | 57.4 ± 7.8 |
| without sheet metal | b0 | 6.9 ± 2.8 | 30.2 ± 3.1 | 48.7 ± 3.9 | 68.1 ± 4.4 |
| without sheet metal | b1 | 7.3 ± 2.8 | 27.9 ± 2.4 | 42.6 ± 1.8 | 59.4 ± 2.6 |
| without sheet metal | b2@0.01 | 5.4 ± 2.4 | 26.3 ± 1.1 | 40.2 ± 3.3 | 59.3 ± 1.8 |
| without sheet metal | b2@0.1 | 8.6 ± 4.8 | 28.6 ± 1.0 | 41.2 ± 6.1 | 60.9 ± 2.2 |
| without sheet metal | b2@1 | 5.8 ± 2.2 | 26.7 ± 2.5 | 37.3 ± 6.9 | 57.6 ± 9.2 |

b1 lowers the nonregular false-alarm macro by 6.0 percentage points, but loses 7.5 recall points against b0, exceeding the five-point tolerance. deltas use unrounded rates. all b2 weights also fail that recall rule against b0. the small weights preserve the nonregular macro against b1 but do not repair the b0 deficit or their normal-only safeguard failures. this is not a robustness improvement or a sota result.

## category recall violations

each cell names categories whose seed-mean recall drops more than five points. these are checks of fixed methods, not selection criteria. per-suffix counts and all comparisons remain in the local aggregate.

| candidate / reference | regular | exposure changed | additional light | nonregular |
|---|---|---|---|---|
| b1 / b0 | fruit jelly, wallplugs | fabric, fruit jelly, rice, vial, wallplugs, walnuts | fabric, fruit jelly, wallplugs | fabric, fruit jelly, rice, vial, wallplugs |
| b2@0.01 / b0 | fruit jelly, wallplugs | can, fabric, rice, vial, wallplugs, walnuts | fabric, fruit jelly, rice, wallplugs | fabric, fruit jelly, rice, vial, wallplugs |
| b2@0.01 / b1 | can, rice | rice | rice | rice |
| b2@0.1 / b0 | fruit jelly | can, fabric, fruit jelly, rice, vial | fruit jelly, rice, wallplugs | fabric, fruit jelly, rice, vial, wallplugs |
| b2@0.1 / b1 | rice | rice | rice, wallplugs | rice |
| b2@1 / b0 | fruit jelly, vial, wallplugs | can, fruit jelly, rice, vial, wallplugs | fabric, fruit jelly, rice, sheet metal, vial, wallplugs | fruit jelly, rice, sheet metal, vial, wallplugs |
| b2@1 / b1 | can, rice, vial | rice | rice, sheet metal, wallplugs | rice, sheet metal, wallplugs |

## audit and boundary

independent raw-score recounts and sklearn auroc checks reproduce every applicable aggregate within `2.23e-16`; checkpoint/source hashes, score/map ids and all native-public populations agree. thirty macro rows for groups absent from some categories remain `not_applicable`, rather than silently dropping categories.

fixed examples use the first sorted regular-light defective image in every category, seed 0, all five methods, with validation-fixed scales. the first two alphabetical categories are included in the report; the full eight-category gallery remains a generated local artifact. whole-image detected/missed decisions are kept separate from localization.

the [complete native pixel auroc/au-pro matrix](ad2-native-public-results.md) is now available separately for all 120 heads. independent primary a40 artifact/source audit and official private-server results remain pending. image-level detection and localization are evaluated separately.

| local evidence identifier | sha256 |
|---|---|
| frozen protocol | `ec53c0c7f9b4d45fcabe1790aeb5286709cca523d9a033c32d9baed54b00f0db` |
| aggregate | `491aed3e527448c4f10fdfabac7f66d3430cb0842a99ec2bb5b78a1867fca802` |
| independent audit | `328bb27d687719cb64ef2f14e410b7fbb1e6fd52dce184d0f6e2c7e319162711` |
| fixed examples | `1005e0a86d7e1acd6b9e732ca6adc6a617f7e0127d61a9cd4031fe6533c6b6a5` |

rebuild from retained scores/maps with `scripts/evaluate.py`, `scripts/build_public_report.py` and `scripts/build_public_examples.py`; see the [report instructions](../report/README.md). data, raw predictions, maps, weights, recordings and personal plans stay outside git.

## review records

[private team artifact](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad2-public-rtx-2026-11-03) contains the score/calibration records, all image aggregation outputs, conditions, per-run scoring metadata and explicitly labelled minimal run-identity projections. archive sha256: `fd25cac09fb10a380f062ecc20fe585d332d5351e8ecfa1eadb46696cc49b2ba`. all six packaged evaluation files reproduce byte-for-byte from these records. downloaded release bytes, zip crc and all 612 file hashes were checked. maps, image pixels, parameters and personal notes are excluded. the projections omit personal training paths and retain original-file hashes. this artifact audits the independent rtx cohort; it does not supply the missing primary a40 source provenance.
