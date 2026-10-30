# Pilot training

This development runner connects the pinned normal manifest to SimpleNet's frozen embedding, trainable adapter and discriminator. It implements the [proposal controls](project.md#comparisons). It does not reproduce the upstream training schedule or establish detector accuracy.

## Run

Initialize the submodule and select the CPU or CUDA environment in [setup](setup.md). Create a local JSON file below `results/`; replace the manifest hash with the full hash of your own normal manifest:

```json
{
  "dataset": "mvtec_ad",
  "category": "bottle",
  "manifest_sha256": "<64-character manifest SHA-256>",
  "variant": "B2",
  "lambda_score": 0.1,
  "seed": 0,
  "batch_size": 2,
  "image_size": [288, 288],
  "brightness": [0.8, 1.2],
  "gamma": [0.8, 1.2],
  "weights": "imagenet_v1"
}
```

The ranges, weight and short duration here are smoke settings, not selected study hyperparameters. B0 and B1 require `lambda_score=0`. Keep all other fields and the manifest identical within each category/seed comparison. Use a separate AD 2 manifest and model for an AD 2 category.

```powershell
uv run --frozen --extra cuda python scripts/train_pilot.py --config results/local-config.json --manifest data/normal.jsonl --root data/mvtec_ad --output results/pilot-b2 --steps 8 --device cuda --threads 4 --gpu-memory-fraction 0.35 --verify-replay
```

Choose CPU with `--extra cpu --device cpu`; omit the CUDA memory limit. The default pretrained backbone may download the official torchvision V1 checkpoint. `weights=random` explicitly permits a synthetic architecture check; it is not a pretrained baseline. The output directory must be new and below this checkout's ignored `results/`. Dataset paths, configurations and outputs stay local.

for square ad images, add `"ad_resize": 329` to use the upstream resize 329 / center crop 288 geometry with `"image_size": [288, 288]`. the default `ad_resize=null` keeps full-frame resizing. choose the same geometry for all paired controls before running them. this option rejects rectangular sources and ad 2; it does not freeze the ad 2 preprocessing decision.

## Training contract

- one category, pinned manifest, training split only. every loaded image is checked against its recorded byte hash. pil bilinear resizing and the optional ad center crop precede all photometric operations. there is no padding.
- WRN50-2 V1, layers 2/3, upstream patch size 3 and 1,536-dimensional aggregation; a 1,536-dimensional linear adapter and two-layer discriminator with hidden size 1,024. The backbone is put in evaluation mode before upstream shape probing, and remains frozen, including BatchNorm buffers.
- B0 uses original/identity, B1 original/relit, B2 original/relit plus raw normal-patch MSE. Brightness/gamma are sampled per image before ImageNet normalization: `clip(brightness * pixels ** gamma, 0, 1)`.
- Independent Gaussian noise, standard deviation 0.015, is added after the adapter to both views. One discriminator call receives normal original, normal second view, synthetic original, synthetic second view. Both hinge terms average across both views; margin is 0.5. Consistency has gradients through both normal branches.
- Adapter AdamW: learning rate 0.0001, weight decay 0.01. Discriminator Adam: learning rate 0.0002, weight decay 0.00001. Both use PyTorch's default betas and epsilon. Constant learning rates; the upstream scheduler is unused.
- Dedicated CPU RNGs control data order, transforms and feature noise. All controls consume the same random draws. The last partial batch is retained. Steps are optimizer updates, not epochs; duplicate views do not double the distinct source-image count.
- The final step count is fixed before a run. No upstream training, test-driven checkpoint selection, test loading or test min/max normalization is called.

the optional ad crop can match the upstream input tensors; it does not make this runner an exact single-view reproduction. the remaining differences include paired discriminator batch composition, source batch size, sampling/noise streams and the fixed update horizon. use b2 minus b1 to isolate consistency under one shared configuration. matched steps do not imply equal compute.

there is also a backbone batch-norm difference. upstream constructs the backbone in training mode and probes its shape with a ones image, updating running statistics once before freezing. the paired runner puts the backbone in evaluation mode before that probe and preserves the pretrained buffers. all paired controls share this behavior. b's faithful upstream reference retains the original probe; record that distinction when comparing it with the paired controls, rather than attributing every difference to relighting or consistency.

the allocation remains b for baseline/training, c for the extension, and shrijeet for data, report/demo and overall review. this runner supplies shared integration and replay checks. agree the maintenance hand-off between b and c in the meeting; the implementation does not change anyone's allocation.

## Checkpoints and outputs

`run.json` records configuration, source/upstream revisions, source fingerprints, lock/manifest/weight hashes, runtime, device and split counts. `train.jsonl` records source-image IDs, component losses, raw score spread, clipping and synchronized step time including loading. Failed runs retain `failure.json` and completed logs.

`final.pt` contains the adapter/discriminator, optimizers, sampler position, all RNG states, configuration and frozen-backbone fingerprint. It omits frozen weights; reconstruct the same backbone when loading. Files are written atomically and never overwritten. For a cold continuation, use the identical configuration/runtime and `--resume results/old-run/final.pt --steps 16 --output results/new-run`. The step count is the total horizon, not the number of additional updates. Continue only within an exploratory pilot; changing a frozen final horizon defines a new study.

`--verify-replay` runs one diagnostic update beyond the fixed horizon, reloads the final checkpoint, and requires exact equality of that update's metrics, heads, BatchNorm buffers, optimizers, sampler and RNGs. It then restores the final checkpoint before validation. The extra update is excluded from training timing; total runtime and memory include the diagnostic. This is same-process replay; the tests additionally check cold reconstruction. Exact equality across different devices, libraries or machines is not claimed. Loading rejects changed configuration, training sources, runtime, device type or backbone.

`--cache-images` keeps each decoded, resized image in memory after its first hash-checked read and hands out copies. Decoding and resizing are deterministic, so training and validation results are unchanged; a test and a CPU run with and without the flag give identical records, checkpoints and validation scores. It matters for large AD 2 images, where re-decoding every step dominates step time. Memory grows by one resized image per sample.

validation saves raw `-D` patch maps and max-patch image scores for every normal validation image. each score record includes native size, resized size, model-input size, `crop_box_xyxy` and `map_coordinates`. crop coordinates are left/top inclusive and right/bottom exclusive in the resized canvas, using torchvision's center rounding; 329 to 288 gives `[20, 20, 308, 308]`.

the first map is bilinearly interpolated to model-input size and smoothed with `sigma=4` input pixels. full-frame runs then restore native size and save `first-validation-native-map.npy`. cropped runs save `first-validation-crop-map.npy` at crop size and report `native_map_shape=null`; no full native map is fabricated. evaluation must resize and crop ground-truth masks with the same geometry. raw patch maps and per-image geometry permit reconstruction of the remaining observed maps. scores are unnormalized and are not defect probabilities.

The threshold is the higher empirical `(1-alpha)` normal-validation quantile, with strict `score > threshold` and default `alpha=0.05`. Results include calibration counts and achieved FPR. This reuses the calibration samples: it is not held-out false-alarm evidence. Defect recall is unavailable. Neither synthetic-feature separation nor normal-only calibration can establish defect sensitivity, shifted-light performance, official metric parity or state of the art.

Timing excludes the first two training steps from its reported median, but includes them in total training time. The runner also records process peak RSS and CUDA peak allocated/reserved memory; initialization, replay and validation can contribute to these peaks. Eight-step timing is a feasibility observation, not a throughput benchmark. Full-resolution AD 2 cost remains a separate measurement.
