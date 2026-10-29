# Bottle baseline training results

27 October 2026. Development runs of the fixed-final single-view AD reference, not B0/B1/B2 results. Only normal validation was scored; test accuracy has not been measured.

All runs used 167 training / 42 normal-validation images, resize 329 then center crop 288, batch size 8 and 40 meta-epochs with four discriminator-training epochs each. Hardware was an NVIDIA A40, with Python 3.12.15, torch 2.6.0+cu124 and torchvision 0.21.0+cu124. See the [baseline guide](baseline-training.md) for the full recipe and reload limits.

| Seed | Training seconds | Peak training VRAM (GiB) | Max reload score / map difference | Checkpoint SHA-256 | `summary.json` SHA-256 |
|---|---:|---:|---|---|---|
| 0 | 836.6 | 1.66 | 0.0 / 0.0 | `3f3452bd46b6a9a4429c10af59f35b4fdc7cfcb4922578ccfd0b70120753d656` | `1556e25a396e5e7f7f396a9e56642b87e54d1f1280bb1e1cbfb08b7bbe214c02` |
| 1 | 829.3 | 1.66 | 0.0 / 0.0 | `3838c54465566c9bfd655df2744e09f80b5c65a9550c275e12f26e56fd4f0d8f` | `052e675ee03ce1683aebcbfdc5302f16e996ab2a467b651d5d6ece8c5589ad08` |
| 2 | 860.2 | 1.66 | 0.0 / 0.0 | `19704c5f61a4968bb81ea2fb03f13b35ab788662c3e854f2a5e8069199a1f166` | `ec17e6dfbbc88a038b421e72d03596e8be77d38e0d0b6dc4d15a6ba06c2a95a6` |

Training time excludes final validation and reload. Peak VRAM is peak allocated training memory.

All three seeds were rerun on 29 October at `22fed6a` from a clean tree, with upstream `351a2b8`, because seed 0's original 27 October run came from `4c91c64`, a commit that only existed before a rebase. Each rerun's checkpoint has the same SHA-256 as the 27 October checkpoint, and its validation scores and maps match exactly, so the released checkpoints stand. The three grid single-view runs used in the AD study (#4) were rerun the same way and also match exactly. The reruns' summaries and validation outputs are in the [rerun release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad-baseline-reruns-2026-10-29).

## Evidence and artifacts

- Each run logged all 3,360 optimizer updates with finite loss and separation diagnostics.
- Checkpoint hashes were verified; learned head values differ across seeds.
- All runs use the same manifest validation membership and save finite maps of shape `(42, 288, 288)`.
- Source records were clean, and fresh-model image scores and maps reloaded exactly. Local artifact copies match the remote hashes.
- The rerun `validation_normal.npz` files also store the geometry: `resize_hw` [329, 329], `crop_box_tlbr` [20, 20, 308, 308] in the resized canvas, and `map_hw` [288, 288].

Manifest SHA-256: `04aadeaba78dd2e93b58166651c255807d05642f5469cc5dc554422e0b05060f`.

Lockfile SHA-256: `3d3b9dd4995622d26fb3077f8000255f974a03c715ebe764a64eeedac3e53aa2`.

The [development artifact release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad-bottle-baseline-2026-10-27) contains the three checkpoints, summaries, normal-validation scores/maps, pinned normal manifest and audit. Its checksums cover the archive and individual files. Dataset images, pretrained backbone weights, machine-specific logs and personal notes are excluded. Binary artifacts remain outside Git.

Archive: `ad-bottle-single-view-seeds-0-1-2.zip`; SHA-256: `c76bf420812c254bd76b55c51f6c1409b918d85cf64aafc830844bf9bec8eb6e`.

## Evaluation handoff

D handles evaluation at the evaluation stage. The fixed checkpoints and normal-validation outputs are ready for that work.

Reconstruct the pinned upstream pretrained backbone and its original BatchNorm shape probe before loading the saved adapter/discriminator. Scores are raw anomaly scores. Maps observe the center crop, with `[20, 20, 308, 308]` coordinates in the resized canvas; ground-truth masks need matching geometry. Keep model, preprocessing, calibration and metric choices fixed before test scoring, with no test-based normalization or selection.

The 20% normal holdout differs from upstream's full `train/good` recipe. Checkpoints omit optimizer and RNG states and do not support continuation. These development runs establish training, artifact and reload behavior, not detection accuracy or published-benchmark parity.
