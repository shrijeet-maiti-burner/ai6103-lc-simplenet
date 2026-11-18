# AD 2 public-test evaluation, 3 November

Evaluation of the 120 released runs (`dev-ad2-study-2026-10-31`) on the AD 2 public test sets. These are public-subset results, not private-benchmark results. No threshold, weight, checkpoint or category was chosen from them.

## Protocol

- Image score: maximum raw −D patch score (higher is more anomalous). An image is flagged when its score is above the threshold.
- Threshold: the higher empirical 95% quantile of that run's normal validation scores, so the target false-positive rate (FPR) is 5%. One threshold per category, method and seed, kept for every lighting condition.
- Recall is the share of defective images flagged; FPR is the share of normal images flagged. Counts and denominators are in `metrics.csv`.
- Macro: unweighted mean over categories for each seed, then mean and sample SD over seeds 0, 1 and 2. Reported over all eight categories and without sheet_metal.
- Recall rule (agreed on #8): recall is preserved in a condition when the candidate's macro recall is within 5 percentage points of the reference. B1 against B0 is the main comparison. Each B2 weight is compared with both B0 and B1. Every category more than 5 points below the reference is named.
- Lighting groups come from the documented file-name suffixes ([lighting definitions](ad2-lighting.md)). `shift_2`, `shift_3` and `shift_4` do not exist in every category, so their eight-category macro is not applicable; the tool does not compute it over fewer categories. The physical setup of vial's `shift_4` is unverified.
- Scoring: `scripts/score_public_test.py` on A40 GPUs, the locked environment, each run's `heads.pt`. The saved validation scores were re-derived for three images per run before test scoring. The encoder hash was checked against the checkpoint. The scoring metadata records the TC2 copy's commit as `b0b23be`; the copy had no git history, so that field is stale and is kept as recorded. `scoring-provenance.json` (below) gives the SHA-256 of the inference source actually run and each run's checkpoint, run-config, image-id and score-file identities. Evaluation used main at `2518602`, which identifies the aggregation, not the copied inference source.

## Results, eight-category macro (mean ± SD over three seeds)

Recall:

| Method | regular | underexposed | overexposed | shift_1 | all non-regular |
|---|---|---|---|---|---|
| B0 | 0.292 ± 0.029 | 0.525 ± 0.066 | 0.397 ± 0.034 | 0.825 ± 0.038 | 0.682 ± 0.041 |
| B1 | 0.278 ± 0.013 | 0.464 ± 0.017 | 0.261 ± 0.010 | 0.747 ± 0.038 | 0.606 ± 0.016 |
| B2 at 0.01 | 0.264 ± 0.017 | 0.464 ± 0.021 | 0.269 ± 0.038 | 0.722 ± 0.046 | 0.604 ± 0.024 |
| B2 at 0.1 | 0.283 ± 0.008 | 0.464 ± 0.027 | 0.283 ± 0.008 | 0.714 ± 0.064 | 0.616 ± 0.026 |
| B2 at 1.0 | 0.264 ± 0.021 | 0.478 ± 0.059 | 0.264 ± 0.013 | 0.683 ± 0.095 | 0.574 ± 0.078 |

FPR (target 0.05):

| Method | regular | underexposed | overexposed | shift_1 | all non-regular |
|---|---|---|---|---|---|
| B0 | 0.060 ± 0.025 | 0.362 ± 0.047 | 0.208 ± 0.030 | 0.634 ± 0.032 | 0.495 ± 0.041 |
| B1 | 0.064 ± 0.024 | 0.310 ± 0.045 | 0.064 ± 0.033 | 0.599 ± 0.080 | 0.435 ± 0.026 |
| B2 at 0.01 | 0.047 ± 0.021 | 0.291 ± 0.038 | 0.058 ± 0.021 | 0.561 ± 0.067 | 0.419 ± 0.036 |
| B2 at 0.1 | 0.075 ± 0.042 | 0.282 ± 0.022 | 0.088 ± 0.047 | 0.534 ± 0.123 | 0.415 ± 0.084 |
| B2 at 1.0 | 0.051 ± 0.019 | 0.275 ± 0.012 | 0.054 ± 0.025 | 0.501 ± 0.090 | 0.383 ± 0.055 |

Image AUROC:

| Method | regular | underexposed | overexposed | shift_1 | all non-regular |
|---|---|---|---|---|---|
| B0 | 0.703 ± 0.010 | 0.642 ± 0.029 | 0.685 ± 0.027 | 0.673 ± 0.010 | 0.631 ± 0.005 |
| B1 | 0.696 ± 0.016 | 0.671 ± 0.030 | 0.695 ± 0.031 | 0.678 ± 0.026 | 0.635 ± 0.007 |
| B2 at 0.01 | 0.698 ± 0.069 | 0.663 ± 0.053 | 0.681 ± 0.077 | 0.667 ± 0.028 | 0.634 ± 0.013 |
| B2 at 0.1 | 0.668 ± 0.063 | 0.668 ± 0.021 | 0.684 ± 0.081 | 0.662 ± 0.013 | 0.638 ± 0.021 |
| B2 at 1.0 | 0.689 ± 0.023 | 0.667 ± 0.022 | 0.694 ± 0.021 | 0.646 ± 0.027 | 0.636 ± 0.003 |

The [supplement](ad2-test-evaluation-supplement.md) gives the same tables for both declared scopes (eight categories and without sheet_metal), the pooled exposure-changed and additional-light groups, every comparison and the achieved validation FPR with counts. Without sheet_metal the drops against B0 are larger (B1: underexposed -7.9, overexposed -16.2, shift_1 -8.3, pooled non-regular -8.7 points), and the B2 weights stay within 5 points of B1 in every condition.

## Recall rule

Macro recall change in percentage points; `!` marks a drop of more than 5 points (recall not preserved).

| Candidate against reference | regular | underexposed | overexposed | shift_1 | all non-regular |
|---|---|---|---|---|---|
| B1 against B0 | −1.4 | −6.1 ! | −13.6 ! | −7.8 ! | −7.5 ! |
| B2 at 0.01 against B0 | −2.8 | −6.1 ! | −12.8 ! | −10.3 ! | −7.8 ! |
| B2 at 0.1 against B0 | −0.8 | −6.1 ! | −11.4 ! | −11.1 ! | −6.6 ! |
| B2 at 1.0 against B0 | −2.8 | −4.7 | −13.3 ! | −14.2 ! | −10.7 ! |
| B2 at 0.01 against B1 | −1.4 | 0.0 | +0.8 | −2.5 | −0.2 |
| B2 at 0.1 against B1 | +0.6 | 0.0 | +2.2 | −3.3 | +0.9 |
| B2 at 1.0 against B1 | −1.4 | +1.4 | +0.3 | −6.4 ! | −3.2 |

Categories more than 5 points below B0 for B1 (points of recall lost):

- overexposed: can −7, fabric −18, fruit_jelly −11, rice −7, vial −53, wallplugs −9, walnuts −9
- underexposed: fabric −9, fruit_jelly −11, rice −22, wallplugs −7, walnuts −7
- shift_1: fabric −22, fruit_jelly −24, rice −9
- shift_2: fabric −11
- shift_3: fabric −24, wallplugs −18
- regular: fruit_jelly −18, wallplugs −9

Names for the other comparisons are in `category_recall_checks` in `aggregate.json`.

## What this shows

- Regular-light public-test macro FPR is 4.7 to 7.5% against the 5% validation target. Achieved validation FPR and counts are reported separately in the supplement.
- Under the agreed rule, B1 does not preserve recall against B0 in underexposed, overexposed or shift_1 light, or in the non-regular pool. No B2 weight preserves recall against B0 across all reported conditions; B2 at 1.0 passes the underexposed comparison (a 4.7-point drop). In the eight-category macro B2 stays within 5 points of B1 except at weight 1.0 under shift_1 (-6.4) and in the pooled additional-light group (-5.5); without sheet_metal it has no exception in applicable conditions (shift_1 -4.8, additional light -3.4).
- B1 and B2 lower the FPR under lighting change (overexposed: 0.208 for B0, 0.054 to 0.088 for the others), but the recall falls with it. Image AUROC barely moves (0.63 to 0.64 over all non-regular images), so the change looks like a shift of the operating point, not better ranking of defective against normal images. That reading is an interpretation of these tables, not a tested claim.
- The frozen regular-light threshold flags 50 to 63% of normal images under shift_1 for every method.
- With three seeds and small public test sets, a single category's recall change of about 5 points may be noise. The macro SDs above show the spread.

## Limits

- This document is image-level. Pixel AUROC and AU-PRO for the same trained heads are in [the native results](ad2-native-public-results.md), which come from a separate RTX scoring of those heads, while this image evaluation used A40 scoring, so the two are not the same scoring outputs. They are pooled over lighting conditions, so localization is not split by light. Selected failure cases and per-category localization are in [the failure gallery](ad2-failure-gallery.md).
- Thresholds come from the regular-light normal validation images of each run and are frozen across lighting. A lighting-specific threshold was not evaluated.
- Three seeds and small public test sets: category-level differences of a few points may be noise, and no significance test was run.
- Categories with few normal validation images give coarse threshold quantiles (sheet_metal has 19), which is why the seven-category scope is reported as well.
- Public-subset results only. They are not private-server results and say nothing about the hidden test lighting.
- The fabric runs in #9 were scored on fabric's public test set with exploratory 100-step checkpoints. They are not part of this evaluation; that earlier test exposure remains a limitation.
- Vial `shift_4` is retained as observed, with its lamp setup unverified.
- Private-set results need the evaluation server.

## Artifacts

Stored outside Git in the [evaluation release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/ad2-eval-outputs-2026-11-03). ZIP SHA-256: `1c3df9cf83ad5494295b029c907dff695b447ef06b872e1baf68875a24eb318f`. Individual file SHA-256:

| File | SHA-256 |
|---|---|
| `aggregate.json` | `9f06c99a48c552f37da199fc44152e2f7830a932c964a80f8a845bd633b629a5` |
| `predictions.csv` | `e3752f45ac298e08d1ef1b62ac61e15d34a6ee3b8bff8613886b6451b0f1a614` |
| `metrics.csv` | `f6896643b264b2ab141ff1368269ae007c1bc52c1fd4343c0272129fe56d3a22` |
| `thresholds.json` | `f9419718a159ca4d98e4a789e473997b63430429edaac69d0c1d4db2d6ca1f43` |
| `conditions.json` | `9000f41b6d7dadc54ce68ca856d86b2221411480d9007fd6a275910e994c5823` |
| `scoring-provenance.json` | `537705618c74c15f82d2ac40658bf523840bd7fba530f001382e8631ad26fbbc` |

Inference source as run on TC2 (recorded in `scoring-provenance.json`):

| File | SHA-256 |
|---|---|
| `scripts/score_public_test.py` | `99e875a2a9503839b7caf55ba84efebd3c4811a9a4134fb38b01b019479f5d21` |
| `src/lc_simplenet/public_scoring.py` | `0a78de70658958577771f91a3c38b42b2871a7f071181bb74a94d5059b131fc0` |
| `src/lc_simplenet/pilot.py` | `bb71dc2f858bf1c52a76512a166aa8792d15738a500470623f2992c304a0dbb9` |
| `uv.lock` | `3d3b9dd4995622d26fb3077f8000255f974a03c715ebe764a64eeedac3e53aa2` |

Command with portable example paths, after `scripts/build_public_conditions.py` built `conditions.json` from the unpacked archive (1,084 public images):

```bash
python scripts/evaluate.py --runs results/ad2-study/paired --out results/ad2-evaluation \
  --conditions results/ad2-evaluation/conditions.json \
  --categories can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts --seeds 0 1 2 \
  --methods B0 B1 B2@0.01 B2@0.1 B2@1 --target-fpr 0.05 --recall-tolerance 0.05 --lighting-aggregates
```

The archive was the user-supplied `mvtec_ad_2.tar.gz` (SHA-256 `c0ded99ef32bfc8e352d52beb44515e5b292b8598cb963aadfa91ca0763505e4`).
