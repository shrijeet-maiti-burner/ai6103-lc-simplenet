# Normal data and score-map geometry

Status: implemented source and geometry helpers; trainer integration is pending. B/C/D must use one explicit input/map contract before experiments. This document does not freeze study categories, resolution, padding or augmentation settings.

## Build and pin the normal data

Follow [the manifest commands](../manifests/README.md) after acquiring the datasets from [MVTec](https://www.mvtec.com/research-teaching/datasets). Keep the archives, images, generated manifests and audits under ignored `data/`. Preserve source/version/checksum and available acquisition-group metadata.

`NormalManifestDataset` reads one category and split from a manifest pinned by SHA-256. It rejects non-normal labels/masks, unsafe/test paths, duplicate IDs, declared cross-split RGB content and AD 2 folder/split disagreement. Each loaded image must match its frozen file hash, dimensions, source mode and frame/orientation assumptions.

the builder and reader both accept only `RGB` and `L` sources. unsupported modes fail during inventory with the mode and filename, before a manifest is written.

```python
import json
from pathlib import Path
from lc_simplenet.data import NormalManifestDataset

manifest = Path("data/manifests/ad-bottle.jsonl")
audit = json.loads(manifest.with_suffix(".audit.json").read_text())
# Copy this hash into the frozen run config before training.
validation = NormalManifestDataset(
    manifest, Path("data/mvtec_ad"),
    dataset="mvtec_ad", category="bottle", split="validation",
    expected_sha256=audit["manifest_sha256"], output_size=None,
)
sample = validation[0]
```

AD validation files physically remain in `train/good`; the manifest field controls access. AD 2 uses its official normal validation folder. No test directories are discovered or read. Normal data cannot validate real-defect recall.

## Input and batch fields

| Field | Meaning |
|---|---|
| `image` | float32 NumPy HWC RGB in [0, 1] |
| `image_id` | Original relative path |
| `dataset`, `category`, `split` | Source and selected membership |
| `label` | Integer 0, normal |
| `original_size` | int64 `[native_height, native_width]` |
| `image_size` | int64 `[output_height, output_width]` |

Native geometry is the default. Explicit `output_size=(height, width)` uses full-image PIL bilinear resizing; it stretches aspect ratio if the requested ratio differs. The reader adds no crop, padding, augmentation or normalization. Grayscale sources become three identical RGB channels. It returns no fake masks.

Pair order: shared geometry -> HWC brightness/gamma transform -> stack/permute BCHW -> ImageNet normalization -> frozen backbone -> trainable adapter/discriminator. Torch's default collation returns **BHWC**. The trainer must permute explicitly and use the same image identity/geometry for both branches. The source does not replace upstream's trainer interface automatically.

## Restore continuous prediction maps

Use `lc_simplenet.geometry.restore_score_map` for full-image resizing with optional bottom/right padding:

```python
from lc_simplenet.geometry import restore_score_map

native_scores = restore_score_map(
    patch_scores,                          # finite 2D floating-point array
    original_size=(native_h, native_w),
    image_size=(model_input_h, model_input_w),
    valid_size=(before_padding_h, before_padding_w),  # omit without padding
)
```

It upsamples the score grid to model-input dimensions, removes bottom/right padding, then resizes the valid map to native dimensions. Interpolation is PIL bilinear in float32. Scores retain their sign and scale; there is no test-dependent normalization or clipping.

This helper does not reverse crops, rotations or arbitrary spatial augmentations. Padding must have been added only at the bottom/right, with the original valid image at the top-left. Map restoration cannot undo changes to model features caused by padding. Future discrete ground-truth masks need separate verified handling; do not interpolate them with this continuous-score helper.

the upstream ad reference uses resize 329 followed by center crop 288. its maps and ground-truth masks must use the same crop coordinates. stretching a cropped map to native size does not restore its missing field of view. the full-frame helper above makes no claim of crop support; ad 2 geometry remains a separate decision.

## Integration checks

Tests cover split isolation, byte/manifest drift, paths/labels, grayscale and batch layout, explicit resize metadata, padding removal before native resizing, border signals and invalid maps. Training still needs a shared resolved configuration, deterministic paired RNG, validation-only checkpoint/calibration rules and official evaluation parity. Keep the original [protocol](protocol.md) and [upstream audit](source-review.md#upstream-code) in force.
