# Single-view AD baseline

[`scripts/train_baseline.py`](../scripts/train_baseline.py) runs the pinned upstream SimpleNet discriminator training loop and keeps the fixed final checkpoint. It reads only the normal training and validation splits from a hash-pinned manifest. It never calls upstream `train()` or its test-based checkpoint selection and normalization.

## Run

Initialize the submodule and select the CUDA environment in [setup](setup.md). Build an AD [normal manifest](../manifests/README.md), preserving the fixed 20% validation holdout. Replace the hash below with the manifest SHA-256 from its audit file:

```bash
uv run --frozen --extra cuda python scripts/train_baseline.py \
  --manifest data/manifests/ad-bottle.jsonl \
  --manifest-sha256 "<64-character manifest SHA-256>" \
  --root data/mvtec_ad --category bottle \
  --out results/ad-bottle-seed0-meta40 --seed 0 --meta-epochs 40
```

Use seeds 0, 1 and 2 with the same manifest, settings, source and environment. Give every run a new output directory. For a CPU check, use `--extra cpu --device cpu`; full training cost on CPU is unmeasured. Dataset files, checkpoints and run outputs remain ignored by Git.

## Recipe and scope

- Upstream commit `351a2b8`, WRN50-2 pretrained weights, layers 2/3, embedding dimension 1,536, patch size 3.
- Batch size 8, bilinear resize 329, center crop 288, then ImageNet normalization. This adapter supports square AD sources; the normal reader accepts RGB and grayscale images. AD 2 preprocessing remains a separate decision.
- 40 meta-epochs, four discriminator-training epochs per meta-epoch, Gaussian feature-noise standard deviation 0.015, discriminator hidden dimension 1,024, two layers and margin 0.5.
- The unchanged upstream optimizers train the adapter and discriminator. The backbone stays frozen during training. Its upstream shape probe runs once in training mode before freezing and updates BatchNorm buffers; retain that behavior when reconstructing this reference.

This single-view reference is separate from the paired B0/B1/B2 controls. Differences in input geometry, BatchNorm handling, batch composition and update count must be recorded when comparing runners. The primary consistency comparison remains B2 minus B1 under matched settings.

The normal validation holdout also leaves fewer training images than upstream's full `train/good` recipe. Report this difference when comparing with published AD results.

## Outputs and checks

Each new run directory contains:

| File | Contents |
|---|---|
| `checkpoint.pt` | Schema-2 final heads/settings, optimizer/scheduler states, final RNGs and provenance |
| `initial-rng.pt` | Initial Python, NumPy, Torch and applicable CUDA RNG states |
| `run.json` | Resolved settings, runtime versions/flags, source hashes, pretrained-weight hash and frozen-backbone fingerprint |
| `validation_normal.npz` | Normal-validation image IDs, raw max-patch anomaly scores and smoothed 288-by-288 crop maps |
| `summary.json` | Seed, settings, split counts, source/upstream revisions, dirty flag, manifest/lock/checkpoint hashes, runtime versions, training time, peak training VRAM and reload score difference |
| `logs/` | Upstream training diagnostics in TensorBoard format |

The runner reconstructs a fresh model, reloads the saved heads and reports the maximum absolute normal-validation image-score difference. Schema-2 runs preserve optimizer/scheduler states and initial/final RNGs without changing training settings. The CLI does not implement continuation. Reloading still requires the same pretrained backbone, shape-probe behavior, settings and compatible runtime; the frozen backbone weights are not duplicated in the checkpoint.

The published three-seed bottle development archive predates schema 2 and contains heads/settings only. Its original files and hashes remain unchanged. Both formats retain the `state` and `settings` keys for inference loading.

Scores are unnormalized anomaly scores, not probabilities. Maps observe only the center crop. Evaluation must resize and crop masks with the same geometry; stretching these maps to the native image would invent the missing field of view. No defect labels, test metrics or operating threshold are produced by this command.

[`tests/test_baseline.py`](../tests/test_baseline.py) checks upstream input parity, rejection of rectangular sources, and synthetic training/reload parity. Passing those checks establishes implementation behavior, not detector accuracy. See the [development results and artifact handoff](baseline-results.md). D handles evaluation at the evaluation stage, using fixed checkpoints and the [evaluation protocol](protocol.md).
