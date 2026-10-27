# Bottle baseline training results

27 October 2026. Development runs of the fixed-final single-view AD reference, not B0/B1/B2 results. Only normal validation was scored; test accuracy has not been measured.

All runs used 167 training / 42 normal-validation images, resize 329 then center crop 288, batch size 8 and 40 meta-epochs with four discriminator-training epochs each. Hardware was an NVIDIA A40, with Python 3.12.15, torch 2.6.0+cu124 and torchvision 0.21.0+cu124. See the [baseline guide](baseline-training.md) for the full recipe and reload limits.

| Seed | Training seconds | Peak training VRAM (GiB) | Max reload image-score difference | Training source |
|---|---:|---:|---:|---|
| 0 | 982.4 | 1.66 | 0.0 | `4c91c64` |
| 1 | 922.9 | 1.66 | 0.0 | `e94f57c` |
| 2 | 889.0 | 1.66 | 0.0 | `e94f57c` |

Training time excludes final validation and reload. Peak VRAM is peak allocated training memory. Seed 0 predates the image-mode rejection and documentation updates; baseline, training runner, image reader and lockfile blobs match across all three runs.

## Evidence and artifacts

- Each run logged all 3,360 optimizer updates with finite loss and separation diagnostics.
- Checkpoint hashes were verified; learned head values differ across seeds.
- All runs use the same manifest validation membership and save finite maps of shape `(42, 288, 288)`.
- Source records were clean, and fresh-model image scores reloaded exactly. Local artifact copies match the remote hashes.

Manifest SHA-256: `04aadeaba78dd2e93b58166651c255807d05642f5469cc5dc554422e0b05060f`.

Lockfile SHA-256: `3d3b9dd4995622d26fb3077f8000255f974a03c715ebe764a64eeedac3e53aa2`.

The [development artifact release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad-bottle-baseline-2026-10-27) contains the three checkpoints, summaries, normal-validation scores/maps, pinned normal manifest and audit. Its checksums cover the archive and individual files. Dataset images, pretrained backbone weights, machine-specific logs and personal notes are excluded. Binary artifacts remain outside Git.

Archive: `ad-bottle-single-view-seeds-0-1-2.zip`; SHA-256: `c76bf420812c254bd76b55c51f6c1409b918d85cf64aafc830844bf9bec8eb6e`.

## Evaluation handoff

D handles evaluation at the evaluation stage. The fixed checkpoints and normal-validation outputs are ready for that work.

Reconstruct the pinned upstream pretrained backbone and its original BatchNorm shape probe before loading the saved adapter/discriminator. Scores are raw anomaly scores. Maps observe the center crop, with `[20, 20, 308, 308]` coordinates in the resized canvas; ground-truth masks need matching geometry. Keep model, preprocessing, calibration and metric choices fixed before test scoring, with no test-based normalization or selection.

The 20% normal holdout differs from upstream's full `train/good` recipe. Checkpoints omit optimizer and RNG states and do not support continuation. These development runs establish training, artifact and reload behavior, not detection accuracy or published-benchmark parity.
