# AD 2 failure gallery and localization by category

Companion to [the evaluation](ad2-test-evaluation.md) and the [native localization results](ad2-native-public-results.md). Public-subset results only; nothing here chose a weight, threshold or checkpoint.

## Selection rule

The cases are chosen by score rank from `predictions.csv` at seed 0 and the regular-light threshold frozen for that run. The rule was fixed before the gallery images were rendered or viewed. The project had earlier public-test exposure (see the evaluation's Limits), so this does not mean the images were never seen.

- up to two of the highest-scoring normal images that the threshold flags (false positives);
- up to two of the lowest-scoring defective images that it misses (false negatives).

Ties break on `image_id` in ascending order. A slot with no qualifying image stays empty and is not filled with a correctly classified one. fruit_jelly has no false positives for B0 or B1 at seed 0 (0 of 20 normal images flagged, checked as score above threshold), so its two panels show misses only. One panel per category for B0 and B1 (16 panels, 60 cases). Selected images can be repeated lighting captures of one object. In 9 of the 30 groups of two, both images are the same object, so they are not two independent examples: can B0 false positives (object 004, shift_1 and shift_2), fruit_jelly B1 misses (003, overexposed and regular), sheet_metal B0 misses (006) and false positives (003), sheet_metal B1 misses (006), vial B0 misses (007) and B1 misses (007), wallplugs B0 misses (009), walnuts B0 misses (005). B2 weights are not shown. Seed 0 only, so a panel illustrates one run, not a rate.

## Reproduce

Retrievable here: [select_failure_gallery.py](../scripts/select_failure_gallery.py), [make_failure_gallery.py](../scripts/make_failure_gallery.py) and [ad2-failure-gallery-selection.json](ad2-failure-gallery-selection.json). The dataset images and the generated panels stay local.

| Input or file | SHA-256 |
|---|---|
| `predictions.csv` (in the [private evaluation release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/ad2-eval-outputs-2026-11-03), same hash as in the evaluation) | `e3752f45ac298e08d1ef1b62ac61e15d34a6ee3b8bff8613886b6451b0f1a614` |
| `ad2-failure-gallery-selection.json` (LF newlines, as committed) | `38fc64d32cf4bdb6009d825f72c5426ec517129bd8e5c0b79e7e6ab5b0084f1d` |
| `scripts/select_failure_gallery.py` | `66b460a54ef431588a1fed5fb2001782e801bf5f8c332955aa00a6c8072ed717` |
| `scripts/make_failure_gallery.py` | `e19b1b83c137d980077a2e0bf9aab1522a4678ec24046f49d72fa1d35458dcdd` |

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

Running the selection script on that `predictions.csv` regenerates the committed selection file byte for byte; the script writes UTF-8 with LF newlines. The selection that drove the panels was first made with an unsaved inline snippet on Windows, and `gallery_selection.json` in the release zip is that original file. It holds the same 60 cases and values but has CRLF newlines (SHA-256 `df1b950153cfe7f06906f43e5edb3ff8203ae1659f59211bea62002b66906ca1`), so it differs byte for byte from the committed file. `select_failure_gallery.py` was written afterwards and checked against it by value. The rendering script is the one that produced the panels, except that its docstring was edited and the file was reformatted with ruff (line wraps only) to pass the repository lint. The selection script was reformatted the same way, and its newline handling was then made explicit.

```bash
python scripts/select_failure_gallery.py predictions.csv ad2-failure-gallery-selection.json
python scripts/make_failure_gallery.py <mvtec_ad_2 root> <runs dir> <out dir> ad2-failure-gallery-selection.json
```

`<runs dir>` holds `<category>-<method>-seed0/test-maps.npz` for the 16 selected runs.

## Panel layout

Top row: the image, with the ground-truth defect outlined in cyan for false negatives. Bottom row: the saved input-size score map with its score against the threshold. Each map is normalized separately by `imshow`, with no shared colour limits, so every case has its own scale. Brightness cannot be compared across cases, including the four cases in one PNG. The scale is for display only and is not used in any metric. The rendered panels are unchanged; only this description was corrected.

## Inspection status

All 16 panels were opened and read on 4 November 2026. What this checks and what it does not:

- Every panel renders four cases (fruit_jelly two), with the titles, scores and thresholds matching the selection file. The image row keeps the image's aspect ratio and the map row is a square, because the maps are at the 256 x 256 input size. The cyan ground-truth outline appears on every false negative and sits on a plausible defect. Several outlines are only a few pixels wide (can, rice, wallplugs) and are hard to see at panel size.
- It does not check pixel alignment of map and mask. The outline is drawn from the native mask over the native image, and the map is shown separately, so no overlay is made.
- Across panels the false positives are mostly shift_1 to shift_3 captures with concentrated hot spots (walnuts, can), or a bright band at the image edge (wallplugs B0, shift_3). The missed defects have maps with scores near or below zero and no clear peak at the defect. These are observations of 30 selected cases at one seed, not rates.

The rendered PNGs are generated outputs and are not committed to Git. They are in the private [A40 localization release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/ad2-a40-localization-2026-11-04) as `ad2-failure-gallery-B0.zip` (SHA-256 `29e34c4e419026c1f2d4ea66692d22a2907dd7263adae3edb8ddbc4b021634db`) and `ad2-failure-gallery-B1.zip` (`6bcb32672d3dba11e114a24768d82b65887cebb72ea6a2af8f4d6e718e05f960`), each with a `MANIFEST.sha256`.

## Localization by category

Pixel AUROC and AU-PRO pool every lighting condition of a category, so they cannot say which lighting drives a change. The values below are the A40 per-category means over seeds 0 to 2, from `native-aggregate.json` in `ad2-a40-native-outputs.zip` (SHA-256 `4215228599f223de2e825deb4537674979a189c017b11af97cbca6bea460c4ef`) in the same release, which also holds the 120 per-run native JSONs and `aggregate_native.py`. Each native JSON records `test_maps_sha256`, and the 16 gallery maps in the table above match those records, so the panels show the maps the metrics were computed from. The RTX cohort is separate: its per-category values are in `native-evidence.json` inside the published native evidence archive (SHA-256 `9e18c2f8808c775cc563258dcbd6f2cef73e92b6a8830edfe1418dc6a0e4448a`, see the [native results](ad2-native-public-results.md)). The eight-category macros agree between the two cohorts; I have not compared the per-category values. The A40 native maps themselves are not in either release, so recomputing the A40 metrics needs the saved maps and the dataset.

| Category | AU-PRO B0 | AU-PRO B1 |
|---|---|---|
| vial | 0.085 | 0.184 |
| wallplugs | 0.023 | 0.074 |
| can | 0.026 | 0.008 |
| fruit_jelly | 0.352 | 0.322 |

- The B1 gain in the eight-category AU-PRO macro (+1.1 points on average over seeds) comes mostly from vial and wallplugs. Can and fruit_jelly fall, so the gain is uneven.
- In the vial B1 panel, both false positives are underexposed normal images with scores near 3.0 against a threshold of 0.82, and both false negatives are defects the map does not cover. This matches the operating-point picture in the image-level tables. One panel illustrates a case; it is not evidence of a cause.
