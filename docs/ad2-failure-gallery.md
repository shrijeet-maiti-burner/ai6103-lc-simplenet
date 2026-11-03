# AD 2 failure gallery and localization by category

Companion to [the evaluation](ad2-test-evaluation.md) and the [native localization results](ad2-native-public-results.md). Public-subset results only; nothing here chose a weight, threshold or checkpoint.

## Selection rule

The cases are chosen by score rank from `predictions.csv` at seed 0 and the regular-light threshold frozen for that run. The rule was fixed before the gallery images were rendered or viewed. The project had earlier public-test exposure (see the evaluation's Limits), so this does not mean the images were never seen.

- the two highest-scoring normal images that the threshold flags (false positives);
- the two lowest-scoring defective images that it misses (false negatives).

Ties break on `image_id` in ascending order. A category with fewer than two such images contributes fewer cases. One panel per category for B0 and B1 (16 panels, 60 cases). B2 weights are not shown. Seed 0 only, so a panel illustrates one run, not a rate.

## Reproduce

Retrievable here: [select_failure_gallery.py](../scripts/select_failure_gallery.py), [make_failure_gallery.py](../scripts/make_failure_gallery.py) and [ad2-failure-gallery-selection.json](ad2-failure-gallery-selection.json). The dataset images and the generated panels stay local.

| Input or file | SHA-256 |
|---|---|
| `predictions.csv` (in the [private evaluation release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/ad2-eval-outputs-2026-11-03), same hash as in the evaluation) | `e3752f45ac298e08d1ef1b62ac61e15d34a6ee3b8bff8613886b6451b0f1a614` |
| `ad2-failure-gallery-selection.json` | `df1b950153cfe7f06906f43e5edb3ff8203ae1659f59211bea62002b66906ca1` |
| `scripts/select_failure_gallery.py` | `2add5b50efe51e3f9439ee1ad0d1c72e46dea9ae487d39ff9258a8f8782b7fbb` |
| `scripts/make_failure_gallery.py` | `305f3c01ed35393cfa0354d879ea2f839db5696aba3e71c01fb2d9fc7f72b0ee` |

Running the selection script on that `predictions.csv` regenerates the selection file byte for byte. The selection was first made with an unsaved inline snippet; `select_failure_gallery.py` was written afterwards and checked against it. The rendering script is the one that produced the panels, except that its docstring was edited.

```bash
python scripts/select_failure_gallery.py predictions.csv ad2-failure-gallery-selection.json
python scripts/make_failure_gallery.py <mvtec_ad_2 root> <runs dir> <out dir> ad2-failure-gallery-selection.json
```

`<runs dir>` holds `<category>-<method>-seed0/test-maps.npz` for the 16 selected runs.

## Panel layout

Top row: the image, with the ground-truth defect outlined in cyan for false negatives. Bottom row: the saved input-size score map with its score against the threshold. The colour scale is set per panel for display only and is not used in any metric, so brightness compares within a panel and not across panels.

## Inspection status

Only one panel (vial, B1) has been checked by eye. The other 15 have not been inspected, so this document does not establish that they render correctly. The rendered PNGs are not in the repository and are not published; they are generated outputs.

## Localization by category

Pixel AUROC and AU-PRO pool every lighting condition of a category, so they cannot say which lighting drives a change. The per-category values are in `native-evidence.json` inside the published native evidence archive (SHA-256 `9e18c2f8808c775cc563258dcbd6f2cef73e92b6a8830edfe1418dc6a0e4448a`, see the [native results](ad2-native-public-results.md)), not in a separate file. The values below are from the A40 aggregation of the same 120 runs (mean over seeds 0 to 2); the eight-category macros agree with that archive, but I have not compared the per-category values against it yet.

| Category | AU-PRO B0 | AU-PRO B1 |
|---|---|---|
| vial | 0.085 | 0.184 |
| wallplugs | 0.023 | 0.074 |
| can | 0.026 | 0.008 |
| fruit_jelly | 0.352 | 0.322 |

- The B1 gain in the eight-category AU-PRO macro (+1.1 points on average over seeds) comes mostly from vial and wallplugs. Can and fruit_jelly fall, so the gain is uneven.
- In the vial B1 panel, both false positives are underexposed normal images with scores near 3.0 against a threshold of 0.82, and both false negatives are defects the map does not cover. This matches the operating-point picture in the image-level tables. One panel illustrates a case; it is not evidence of a cause.
