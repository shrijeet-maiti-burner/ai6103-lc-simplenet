# Second-stack AD 2 replicate

Author: umeshgjh

This note records the normal-data manifests for a finished second-stack AD 2 run. The 72 run directories are not in Git. They are published in [`dev-ad2-replicate-2026-11-02`](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad2-replicate-2026-11-02), built from commit `ab0313d03c5769f004febf54b82dd1b0169ba2ae`.

The numeric comparison is already in [the AD 2 study results](ad2-study-results.md), under "Second environment (#11)". This note does not add another results table and does not replace `configs/ad2-study-policy.json`.

## What the run is

This is a separate second-stack replicate. It is not the locked-stack study and not the `dev-ad2-study-2026-10-31` release.

Seventy-two runs completed: eight categories (`can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs`, `walnuts`), variants B0, B1, and B2, and seeds 0, 1, and 2.

B2 uses λ = 0.01. B0 and B1 use λ = 0. The run does not select a consistency weight. On the seed means, λ = 0.01 fails the 1.25 hinge cap in fabric and vial, so it is not an eligible weight.

## Stack

The 72 `run.json` records give Python 3.11.16, torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900, and an NVIDIA A40. The lock at the recorded commit pins torch 2.6.0+cu124.

## Runtime differences inside the matrix

Three can B0 runs used 4 threads and GPU memory fraction 0.8. Seeds 0 and 1 also left the image cache off. The other 69 runs used 2 threads, fraction 0.25, and the cache.

`sheet_metal` has 19 normal-validation images.

## Manifests

The manifests are `references/ad2-normal-manifests/<category>.json`. Each audit file records the manifest SHA-256. Those hashes are the `manifest_sha256` values in the 72 run records.

## What the package does not contain

No checkpoint file is included. The recorded checkpoint hashes were not checked. Every `run.json` has `verify_replay` set to false. Every `result.json` has `replay` set to null.

The published release redacts cluster home paths in the 72 `run.json` files to `<local-path>`. `artifact-manifest.json` keeps the original hash of every file. The release notes state that the other 474 files are byte-identical to the source commit. This is not an unredacted copy of the `run.json` files.

Do not pool these runs with the locked-stack study. Nothing here shows that the two stacks produced the same weights, that a B2 weight was selected, or that any test image was scored.
