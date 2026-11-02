# Normal-data manifests

The builder inventories only normal source images. Dataset archives, images, generated manifests and audits stay in ignored `data/`. No dataset is included.

```powershell
uv run python scripts/build_normal_manifest.py --dataset mvtec_ad --root data/mvtec_ad --category bottle --out data/manifests/ad-bottle.jsonl
uv run python scripts/build_normal_manifest.py --dataset mvtec_ad2 --root data/mvtec_ad2 --category fabric --out data/manifests/ad2-fabric.jsonl
```

For AD 2, `--root` may be omitted when `MVTEC_AD2_ROOT` is set. See [the AD 2 audit notes](../docs/ad2-data-audit.md).

Categories are command examples, not frozen study choices. Repeat `--category` to inventory more categories. The default AD holdout is 20%, using split seed 0; grouping identical decoded RGB content keeps copies together. AD 2 retains official `train/good` and `validation/good` membership. Exact RGB content crossing splits fails the audit rather than silently changing the source split. Test directories are not traversed.

Rows contain dataset/category, manifest split, original relative path, normal label, absent mask, native width/height/mode, file SHA-256 and decoded RGB SHA-256. The adjacent `.audit.json` records counts, parameters, duplicate groups and the manifest hash. Existing outputs are not overwritten.

AD validation rows still point into the original `train/good` directory. Use the manifest-enforced reader described in [the data contract](../docs/data-contract.md); reading the entire directory includes the holdout. Exact pixel hashes do not establish physical-object/acquisition-group identity. Record provenance and available grouping metadata separately.
