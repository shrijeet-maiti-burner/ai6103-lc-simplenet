# AD 2 failure gallery and localization by category

Companion to [the evaluation](ad2-test-evaluation.md) and the [native localization results](ad2-native-public-results.md). Public-subset results only; nothing here chose a weight, threshold or checkpoint.

## Selection rule

Fixed before any image was viewed, from `predictions.csv` at seed 0 and the regular-light threshold frozen for that run:

- the two highest-scoring normal images that the threshold flags (false positives);
- the two lowest-scoring defective images that it misses (false negatives).

One panel per category for B0 and B1 (16 panels). The chosen images and scores are in `gallery_selection.json`. B2 weights are not shown. Seed 0 only, so a panel illustrates one run, not a rate.

## Panel layout

Top row: the image, with the ground-truth defect outlined in cyan for false negatives. Bottom row: the saved input-size score map with its score against the threshold. The colour scale is set per panel for display only and is not used in any metric, so brightness compares within a panel and not across panels.

The panels are generated run outputs and are kept out of Git, as the working rules require. They are rendered by `make_gallery.py` from the selection file and the saved maps. The rendered PNGs are not in the repository yet; they will go to the private evaluation release once the team agrees.

## Localization by category

Pixel AUROC and AU-PRO pool every lighting condition of a category (see the native results for the method), so they cannot say which lighting drives a change. Mean over seeds 0 to 2, from `native-aggregate.json` (SHA-256 `c03c8de294bfd03ef9e6ad953ae74fb274b9d1f4515b5971445803e56a401f11`, outside Git):

| Category | AU-PRO B0 | AU-PRO B1 |
|---|---|---|
| vial | 0.085 | 0.184 |
| wallplugs | 0.023 | 0.074 |
| can | 0.026 | 0.008 |
| fruit_jelly | 0.352 | 0.322 |

- The B1 gain in the eight-category AU-PRO macro (+1.1 points on average over seeds) comes mostly from vial and wallplugs. Can and fruit_jelly fall, so the gain is uneven.
- In the vial B1 panel, both false positives are underexposed normal images with scores near 3.0 against a threshold of 0.82, and both false negatives are defects the map does not cover. This is the same operating-point picture as the image-level tables: the frozen regular-light threshold is exceeded by many normal images under changed light. One panel is an illustration, not evidence of a cause.
