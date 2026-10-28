# MVTec AD 2 data audit

Procedure for a user-provided official archive. This document does not freeze AD 2 geometry, categories, or hyperparameters. Machine-specific paths stay in ignored `data/audits/` files, not here.

Expected image counts below are from Heckler-Kram et al., [arXiv:2503.21622v1](https://arxiv.org/html/2503.21622v1), Table IV. They are **paper-reported**, not on-disk measurements. Compare them with `scripts/audit_mvtec_ad2.py` output tagged `source=files`.

Work from branch `ad2-fabric-prep` (based on `feat/b-study-runner`). Dataset archives, extracted images, generated manifests, filled audit JSON, checkpoints and `.env` stay out of Git.

## Acquire and extract

Download from [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) under CC BY-NC-SA 4.0. This repository does not fetch the archive.

Use uv **0.13.0** (`pyproject.toml` `required-version = "==0.13.0"`). If PATH `uv` is newer, invoke a 0.13.0 binary instead of replacing the global install; see [setup](setup.md#uv-0130-pin).

```bash
export UV="${UV:-uv}"   # or $HOME/.local/uv-0.13.0/uv
export MVTEC_AD2_ARCHIVE=data/fabric.tar.gz
export MVTEC_AD2_ROOT=data/mvtec_ad2

"$UV" sync --frozen --extra cpu
"$UV" run --frozen --extra cpu python scripts/prepare_mvtec_ad2_archive.py
"$UV" run --frozen --extra cpu python scripts/audit_mvtec_ad2.py --root "$MVTEC_AD2_ROOT" --out data/audits/ad2-tree.json
"$UV" run --frozen --extra cpu python scripts/build_normal_manifest.py --dataset mvtec_ad2 --root "$MVTEC_AD2_ROOT" --category fabric --out data/manifests/ad2-fabric.jsonl
"$UV" run --frozen --extra cpu pytest
"$UV" run --frozen --extra cpu python scripts/smoke_ad2.py \
  --manifest data/manifests/ad2-fabric.jsonl \
  --root "$MVTEC_AD2_ROOT" \
  --device cpu \
  --weights random
```

The pinned fabric normal-manifest SHA-256 is `6f9c589c1b45a0af94e77c3261eb6623236a42c1284538314bea344e63def2f5` (387 train / 43 validation). Rebuilds must match it. `--inspect-only` lists archive members without writing files. Extraction refuses path traversal, links and special files. A second run with the same archive SHA-256 and a matching acquisition record skips extraction; a changed archive or a conflicting tree stops with an error instead of overwriting. The builder refuses to replace an existing manifest.

`scripts/prepare_mvtec_ad2_archive.py` writes `data/audits/ad2-acquisition.json` by default. That file is gitignored.

## Table IV expected counts

| Category | Train | Val | TEST pub (good/bad) | TEST priv | TEST priv,mix | Native W×H |
|---|---:|---:|---|---:|---:|---|
| can | 412 | 46 | 162 (72/90) | 321 | 321 | 2232×1024 |
| fabric | 387 | 43 | 156 (66/90) | 314 | 314 | 2448×2048 |
| fruit_jelly | 263 | 37 | 80 (20/60) | 255 | 255 | 2100×1520 |
| rice | 313 | 35 | 132 (42/90) | 277 | 277 | 2448×2048 |
| sheet_metal | 137 | 19 | 114 (24/90) | 142 | 142 | 4224×1056 |
| vial | 291 | 41 | 140 (35/105) | 276 | 276 | 1400×1900 |
| wallplugs | 293 | 33 | 150 (60/90) | 232 | 232 | 2448×2048 |
| walnuts | 432 | 48 | 150 (60/90) | 228 | 228 | 2448×2048 |

Private good/bad splits in the paper are **not** local labels. The audit counts private files and does not infer defect membership from names.

On the fabric archive, duplicate file SHA-256 groups were only:

- Public masks: the same mask bytes repeated across lighting names for one defect index (`regular`, `overexposed`, `underexposed`, `shift_*`). That matches a shared annotation under changed illumination.
- Private RGB: some files in `test_private` are byte-identical to files in `test_private_mixed`. The mixed set is documented as containing both seen and unseen lighting; identical bytes do not assign private labels.

No duplicate file hashes involved `train/good`, `validation/good`, public RGB, or a second category. Those groups are not a reason to alter the normal manifests.

## Geometry (pending team agreement)

The [AD baseline](baseline-training.md) uses upstream `Resize(329)` then center crop 288. That path **rejects non-square natives**, so fabric 2448×2048 cannot use it unchanged. The [paired pilot](pilot-training.md) `ad_resize` option is **MVTec AD only**. Shared `PilotConfig.image_size` still defaults to 288×288 for bottle/grid; this branch does not change `baseline.py` or `pilot.py`.

Heckler-Kram et al. (arXiv:2503.21622v1) resize images to **256×256** unless a method says otherwise, and train SimpleNet for **160 epochs**. That is the proposed AD 2 protocol if the team agrees to match the paper baseline: full-frame PIL bilinear resize to 256×256, which **stretches** non-square natives such as fabric 2048×2448 (H×W). It is **not frozen** in training code.

**Recorded CPU smoke** (`scripts/smoke_ad2.py`, a pipeline check, not a scientific result): native fabric 2048×2448 was stretched to **288×288**, then ImageNet-normalized. Default `--image-height/--image-width` remain 288 so that check is reproducible. Passing 256 is optional and still not a training recipe. `geometry_frozen` stays false.

Do not start B0/B1/B2 until the team agrees. If 256×256 full-frame stretch is chosen:

1. Use the same resize for train, validation and inference (`output_size=(256, 256)`, `ad_resize=None`).
2. Restore continuous score maps with `restore_score_map(..., original_size=native_hw, image_size=(256, 256))` and no `valid_size` (no pad, no crop). That inverts the same full-frame stretch.
3. Discrete ground-truth masks need a **separate nearest-neighbor** full-frame resize to 256×256, not `restore_score_map` (bilinear would create fractional labels). Evaluate masks and maps in the same coordinate system.
4. Do not apply the AD 329→288 crop to AD 2; `restore_score_map` cannot undo a crop.
5. Implement that geometry in an AD 2-only config, not by changing shared bottle/grid defaults.

Until then, keep pad-versus-stretch as an open choice. The smoke aspect-ratio field uses `h * target_w != w * target_h`.

## Normal manifests

Reuse [`scripts/build_normal_manifest.py`](../scripts/build_normal_manifest.py). Do not add a second builder. It inventories only `train/good` and `validation/good`. Test directories are not walked.

## NTU GPU readiness (before B0/B1/B2)

Do not launch the three-variant fabric matrix until geometry is agreed. After copying data and checking out `ad2-fabric-prep`:

1. **Environment:** Python 3.12.15, uv 0.13.0, `git submodule update --init --recursive`, then `"$UV" sync --frozen --extra cuda`. Do not mix `--extra cpu` and `--extra cuda`.
2. **CUDA:** `"$UV" run --frozen --extra cuda python -m lc_simplenet doctor --require-cuda`. Driver must support the pinned PyTorch 2.6.0 / torchvision 0.21.0 CUDA 12.4 wheels.
3. **Pretrained weights:** ImageNet WRN50-2 V1 (`wide_resnet50_2-95faca4d.pth` via torch hub). The CPU smoke used **random** weights and did not download this file. Confirm the hub cache or a one-time download on the cluster before the CUDA smoke.
4. **GPU memory:** Fabric natives are 2048×2448; the provisional smoke used 288×288. Peak VRAM for fabric is unmeasured. Record `torch.cuda.max_memory_allocated` from the CUDA smoke before any full run. Do not assume bottle A40 numbers apply.
5. **Minimal CUDA smoke only:**

```bash
"$UV" run --frozen --extra cuda python scripts/smoke_ad2.py \
  --manifest data/manifests/ad2-fabric.jsonl \
  --root "$MVTEC_AD2_ROOT" \
  --device cuda \
  --weights imagenet_v1
```

Stop there. No B0/B1/B2, no three-seed training, no test-label evaluation.
