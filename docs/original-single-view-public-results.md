# original single-view ad2 reference

9 november 2026. all eight categories and seeds 0/1/2 were evaluated with the released fixed-final single-view heads. this is a separate original-reference cohort on the already exposed annotated public subset. it does not replace the matched [paired-study comparison](ad2-public-rtx-results.md).

## recipe and replay

the original runs use 40 meta-epochs, four discriminator epochs per meta-epoch, batch eight, imagenet-v1 wide-resnet50-2 features and the upstream adapter/discriminator. ad2 images are stretched to the full 256px square frame. this is the documented ad2 adaptation, not the original paper's mvtec-ad geometry or test-selected checkpoint rule.

the [backbone-state audit](backbone-state.md) recovered the original a40 reconstruction. all 24 heads, 906 saved normal predictions, complete maps and saved-threshold decisions replayed bit-exactly before public scoring. the complete normal replay was repeated for every head during scoring, again with zero error. the laptop's failed complete replay remains retained; its outputs were not used here.

all 24 heads scored the same 1,084 images: 379 normal and 705 defective, giving 3,252 predictions. each head uses its original saved normal-validation higher empirical q95, with strict `score > threshold`, unchanged across conditions. no head, category, threshold or postprocessing was selected from these outcomes.

## image operating point

percentages: category macro within seed, then mean/sample sd across three seeds.

| condition | fpr | defect recall | image auroc |
|---|---:|---:|---:|
| all public images | 40.01 ± 2.38 | 54.94 ± 4.19 | 62.07 ± 0.97 |
| regular | 7.05 ± 1.79 | 23.61 ± 3.37 | 65.77 ± 3.03 |
| nonregular | 46.48 ± 2.58 | 61.15 ± 4.37 | 61.71 ± 0.85 |
| exposure changed | 25.34 ± 1.33 | 39.58 ± 4.81 | 61.45 ± 3.07 |
| additional light | 61.55 ± 4.47 | 75.90 ± 4.27 | 62.59 ± 1.05 |

nonregular pools exposure changes and additional lights, including vial's physically unverified `shift_4`; it is not a pure unseen-light population. every category/seed/suffix count and score remains in the retained results. groups absent from some categories do not enter an all-category macro by silently dropping categories.

these rates describe the original reference. its backbone initialization, batch treatment and training budget differ from the paired study. comparing it with b1 cannot isolate augmentation or consistency, and cannot reverse b1's failed recall rule against its matched b0. there is still no demonstrated overall lighting-robustness improvement or sota result.

## native localization

the complete 24-head native matrix is running with the frozen exact float32 evaluator, au-pro fpr limit 0.05, eight-connected regions and no score quantization. partial category values are not reported as a complete result.

## evidence and reproduction

an independent implementation reconstructed all 237 condition rows using sorted calibration scores, direct counts and pairwise positive/negative auroc. it reproduced every image metric and macro within `1.12e-16`; each seed has exactly 379 normal and 705 defective predictions. transport sha256, zip crc, ordered image identities, checkpoint hashes and complete map dimensions also passed.

scoring used the verified source archive at `a05eb9e41fca0106abf72aae0ff632248c1e0c42` plus the inference helper from `c8c23ab`, on the original a40 runtime. the documented [filesystem scoring command](backbone-state.md#score-the-frozen-public-cohort) repeats full normal replay before opening public input. the actual remote run used the retained streaming worker; these are distinct execution paths.

| retained private record | sha256 |
|---|---|
| protocol, frozen before this reference's public input discovery | `ca3062f76c6a940880131d14d175ffad6163e383ff2f039415f8500bedb871c4` |
| complete input manifest | `53a680dbca0cae3e10814f548826ce6ee67fc4a1ed8499bdb4b6af33a549a8e3` |
| complete image results | `4cea435beff510ddfd1686e1852281f67c028787670fd3c9b21f37a7af0cfc50` |
| collection audit | `1962266fd1fcbc36586ed9f222635cd4bbb8317b6e747021f69e069c7431f991` |

data, checkpoints, raw predictions and maps remain outside git. these hashes identify retained local evidence, not a public artifact release. official private-server evaluation remains separate.
