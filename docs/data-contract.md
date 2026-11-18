# Normal data and score-map geometry

The loader enforces normal-image split membership and records geometry explicitly. AD 2 study inputs are full-frame 256×256; the separate AD reference uses resize 329 and crop 288.

## Build and pin the normal data

Follow [the manifest commands](data.md) after acquiring the datasets from [MVTec](https://www.mvtec.com/research-teaching/datasets). Keep the archives, images, generated manifests and audits under ignored `data/`. Preserve source/version/checksum and available acquisition-group metadata.

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

pair order: shared geometry -> brightness/gamma on raw rgb in [0, 1] -> imagenet normalization -> frozen backbone -> trainable adapter/discriminator. torch's default collation returns **BHWC**; the trainer must explicitly convert to **BCHW** before the backbone. relighting may operate on the raw hwc array or raw bchw tensor, always before normalization. both branches must share image identity and geometry. the source does not replace upstream's trainer interface automatically.

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

This helper does not reverse crops, rotations or arbitrary spatial augmentations. Padding must have been added only at the bottom/right, with the original valid image at the top-left. Map restoration cannot undo changes to model features caused by padding. The native evaluator reads binary masks separately; do not interpolate them with this continuous-score helper.

the upstream ad reference uses resize 329 followed by center crop 288. its maps and ground-truth masks must use the same crop coordinates. stretching a cropped map to native size does not restore its missing field of view. the full-frame helper above makes no claim of crop support; AD 2 uses the documented full-frame 256×256 resize.

## Integration checks

tests cover split isolation, byte/manifest drift, paths/labels, grayscale and batch layout, resize metadata, padding removal before native resizing, border signals and invalid maps. the runner adds explicit crop coordinates, deterministic paired rng, fixed final checkpoints and normal-only calibration. the [native evaluator](localization.md) passed numerical checks against the official utility; this is implementation parity, not private-server certification. the frozen public study is reported in the [evidence ledger](report-evidence.md). keep the [protocol](protocol.md) and [upstream audit](upstream.md) in force.
