# MVTec AD 2 data audit

Procedure for a user-provided official archive. This document does not freeze AD 2 geometry, categories, or hyperparameters. Machine-specific paths stay in ignored `data/audits/` files, not here.

Expected image counts below are from Heckler-Kram et al., [arXiv:2503.21622v1](https://arxiv.org/html/2503.21622v1), Table IV. They are **paper-reported**, not on-disk measurements. Compare them with `scripts/audit_mvtec_ad2.py` output tagged `source=files`.

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

The pinned fabric normal-manifest SHA-256 is `6f9c589c1b45a0af94e77c3261eb6623236a42c1284538314bea344e63def2f5` (387 train / 43 validation). Rebuilds must match it. `--inspect-only` lists archive members without writing files. Extraction refuses path traversal, links, special files, and overwrites. The builder refuses to replace an existing manifest.

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

## Geometry (open team decision)

The [AD baseline](baseline-training.md) uses upstream `Resize(329)` then center crop 288. That path **rejects non-square natives**, so fabric 2448×2048 cannot use it unchanged. The [paired pilot](pilot-training.md) `ad_resize` option is **MVTec AD only**.

**Provisional CPU smoke only** (`scripts/smoke_ad2.py`, not a training recipe): native fabric **2048×2448** (H×W) was full-frame PIL bilinear **stretched** to **288×288**, then ImageNet-normalized. No 329→288 crop. `geometry_frozen` is false. This does **not** freeze AD 2 input geometry and does not change `baseline.py` or `pilot.py`.

Do not start B0/B1/B2 until the team picks one of:

1. Aspect-preserving resize and pad, with a recorded valid region for map restoration.
2. Explicit full-frame stretch to square, documented as a change of native aspect.

`restore_score_map` restores continuous scores after full-frame resize or bottom/right pad. It cannot undo a crop. Discrete ground-truth masks need a separate rule.

## Normal manifests

Reuse [`scripts/build_normal_manifest.py`](../scripts/build_normal_manifest.py). Do not add a second builder. It inventories only `train/good` and `validation/good`. Test directories are not walked.

## NTU GPU readiness (before B0/B1/B2)

Do not launch the three-variant fabric matrix until geometry is agreed. After copying data and checking out `research/ad2-fabric-prep`:

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
