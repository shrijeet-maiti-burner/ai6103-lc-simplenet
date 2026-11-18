# Data and manifests

Acquire [MVTec AD / AD 2](https://www.mvtec.com/research-teaching/datasets), [AeBAD](https://github.com/zhangzjn/AeBAD) and [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) separately under their dataset terms. Images, masks, weights and generated files are excluded from Git and the code ZIP. Preserve the licence and archive checksum with each local copy.

The usual layout is `data/mvtec_ad/<category>/train/good/` or `data/mvtec_ad2/<category>/{train,validation}/good/`. Command-line paths override these examples; the tools also accept explicitly exported `MVTEC_AD2_ROOT` and `MVTEC_AD2_ARCHIVE`. No .env loader is used.

## AD 2 acquisition audit

Inspect or extract a user-provided archive:

```bash
uv run --frozen --extra cpu python scripts/prepare_mvtec_ad2_archive.py --archive data/mvtec_ad_2.tar.gz --inspect-only
uv run --frozen --extra cpu python scripts/prepare_mvtec_ad2_archive.py --archive data/mvtec_ad_2.tar.gz --root data/mvtec_ad2
uv run --frozen --extra cpu python scripts/audit_mvtec_ad2.py --root data/mvtec_ad2 --out data/audits/ad2-tree.json
```

Supply `--expected-sha256` when the acquisition hash is known. Extraction rejects unsafe paths and links, records the archive identity, and refuses conflicting existing files. The tree audit compares observed counts with Table IV of the versioned AD 2 paper; paper counts are expectations, not inferred private labels.

The study used all eight AD 2 categories: can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs and walnuts. Normal inputs use full-frame bilinear 256×256 resizing, without crop or padding. This stretches non-square images; native dimensions remain in prediction records. The earlier fabric-only smoke runs are historical development checks, not the final study.

## Normal manifests

```bash
uv run --frozen --extra cpu python scripts/build_normal_manifest.py --dataset mvtec_ad --root data/mvtec_ad --category bottle --out data/manifests/ad-bottle.jsonl
uv run --frozen --extra cpu python scripts/build_normal_manifest.py --dataset mvtec_ad2 --root data/mvtec_ad2 --category fabric --out data/manifests/ad2-fabric.jsonl
```

Repeat `--category` to inventory more categories. The builder reads only normal sources, never test folders. AD keeps a 20% normal holdout with split seed 0, grouping identical decoded RGB content. AD 2 preserves official train/validation membership and rejects exact RGB overlap across those splits.

Rows record category, split, original relative path, normal label, absent mask, native dimensions/mode, file SHA-256 and decoded-RGB SHA-256. The adjacent audit records counts, parameters and manifest hash. Existing outputs are not overwritten. The committed study manifests are in [references/ad2-normal-manifests](../references/ad2-normal-manifests/).

AD validation files physically remain in `train/good`; the [manifest-enforced reader](data-contract.md) controls their membership. Exact pixel hashes cannot establish physical-object or acquisition-group independence.

## Public evaluation and follow-up inputs

AD 2 public labels come from `test_public/good` and `test_public/bad`; private ground truth is server-only. Preserve the [lighting suffix mapping](ad2-lighting.md), including vial's physically unverified shift_4. The public cohort contains 1,084 images: 379 normal and 705 defective.

The routed [AD 2](routed-reproduction.md) and [blade](routed-blade-reproduction.md) commands stage exact native input identities from their bundled execution manifests. Both require the fixed 256 DTD donors. Blade fitting uses 128 normal images, 64 calibration images and 64 procedural-development normals, separately from its 1,639 real test images.

Keep calibration and test inputs separate. The recorded experiment results have [prior public-test exposure and other limits](routed-results.md); do not use those outcomes to choose new settings.
