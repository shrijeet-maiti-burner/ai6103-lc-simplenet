# AD 2 failure gallery and localization by category

Companion to [the evaluation](ad2-test-evaluation.md) and the [native localization results](ad2-native-public-results.md). Public-subset results only; nothing here chose a weight, threshold or checkpoint.

## Selection rule

The cases are chosen by score rank from `predictions.csv` at seed 0 and the regular-light threshold frozen for that run. The rule was fixed before the gallery images were rendered or viewed. The project had earlier public-test exposure (see the evaluation's Limits), so this does not mean the images were never seen.

- up to two of the highest-scoring normal images that the threshold flags (false positives);
- up to two of the lowest-scoring defective images that it misses (false negatives).

Ties break on `image_id` in ascending order. A slot with no qualifying image stays empty and is not filled with a correctly classified one. fruit_jelly has no false positives for B0 or B1 at seed 0 (0 of 20 normal images flagged, checked as score above threshold), so its two panels show misses only. One panel per category for B0 and B1 (16 panels, 60 cases). Selected images can be repeated lighting captures of one object: both vial B1 misses are object 007 (regular and shift_2), so they are not two independent defect examples. B2 weights are not shown. Seed 0 only, so a panel illustrates one run, not a rate.

## Reproduce

Retrievable here: [select_failure_gallery.py](../scripts/select_failure_gallery.py), [make_failure_gallery.py](../scripts/make_failure_gallery.py) and [ad2-failure-gallery-selection.json](ad2-failure-gallery-selection.json). The dataset images and the generated panels stay local.

| Input or file | SHA-256 |
|---|---|
| `predictions.csv` (in the [private evaluation release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/ad2-eval-outputs-2026-11-03), same hash as in the evaluation) | `e3752f45ac298e08d1ef1b62ac61e15d34a6ee3b8bff8613886b6451b0f1a614` |
| `ad2-failure-gallery-selection.json` | `df1b950153cfe7f06906f43e5edb3ff8203ae1659f59211bea62002b66906ca1` |
| `scripts/select_failure_gallery.py` | `6a645b5a7df8f013a8bc5404e0808ff4ffd3cbe844674ce144cf27e328256751` |
| `scripts/make_failure_gallery.py` | `c0ba199fdcb7540c60af114e1739625ec75ca6d0b568eb2fde34c2cfe41dfd89` |

Saved score maps used for the panels (seed 0, from the released paired runs `dev-ad2-study-2026-10-31`):

| Run | `test-maps.npz` SHA-256 |
|---|---|
| `can-B0-seed0` | `bfc5d9a5738d6a58766c15271a64e0f129f52b84ae35b5a2d32365a31f52b3c6` |
| `can-B1-seed0` | `ad5cea5742b4e7e96bc4bb849e06908d5979b38f8bddd92280995fe78320bbe4` |
| `fabric-B0-seed0` | `93f7421f6bdbcb40ea1ba3c0d8fe3f6dcb2d3b520b2950fb727762b891fce458` |
| `fabric-B1-seed0` | `63d14550a9658103eaf2757707e7eeb5a008ee54972294753d464376e5b20220` |
| `fruit_jelly-B0-seed0` | `9ffc6e7553e19d914972a8b26b69ee1f4dc2d8e49fe6c0416945411c52a2ca4b` |
| `fruit_jelly-B1-seed0` | `58e2d5c2e485bd6f97bf4c0860f76f1531390910eb05fc8c378551d326a13d64` |
| `rice-B0-seed0` | `939828827e2c88925ed84459069f8958ef49ac2a891f0d27023a29e13783e83f` |
| `rice-B1-seed0` | `9c184934e779fde3db7b731a48aede0a35de744bf95d97f7bfc6543a1fbc9855` |
| `sheet_metal-B0-seed0` | `516fcbdc260101263b0c4805bd84ea231c6fbe6199803fa608c6af7d0f3cd601` |
| `sheet_metal-B1-seed0` | `cfd4d2bdf7b201f8ce3bdedeea333e251d2eceba3de4f463d28975f787e351b4` |
| `vial-B0-seed0` | `46bae03abf33e8b67fcc9ae86656663d67a4f3fcea65542a06d994e43468f429` |
| `vial-B1-seed0` | `b893502f71172a59e9dfd16a4de467a3a4a50e1aa819deb36a8a895c16bcf06b` |
| `wallplugs-B0-seed0` | `2c8fe8136e74819d7119e820e83f2396c67b7a4f32176ff1ae46f2cffccc6841` |
| `wallplugs-B1-seed0` | `e603422d6918020be83020aeeebd84eac293e27a03c390c20f8cdf896e6a9fdd` |
| `walnuts-B0-seed0` | `ea9343d38007e78eb4b4e60c162d38a5f5fdfd59f15b0117ad107c7b93219932` |
| `walnuts-B1-seed0` | `6ff6b7c678ddb4642003068f469de9b1fc4951cf4c8aa0c9f94be108f816e0d7` |

Running the selection script on that `predictions.csv` regenerates the selection file byte for byte. The selection was first made with an unsaved inline snippet; `select_failure_gallery.py` was written afterwards and checked against it. The rendering script is the one that produced the panels, except that its docstring was edited and the file was reformatted with ruff (line wraps only) to pass the repository lint. The selection script was reformatted the same way and its output hash is unchanged.

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
