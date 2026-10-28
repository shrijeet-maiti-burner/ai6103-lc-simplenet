# Local data

Download MVTec AD and AD 2 from [MVTec](https://www.mvtec.com/research-teaching/datasets) under their dataset terms. AD 2 uses CC BY-NC-SA 4.0. Dataset images, masks and archives are ignored by Git.

Place local datasets at `data/mvtec_ad/` and `data/mvtec_ad2/`, or set `MVTEC_AD2_ROOT` / `MVTEC_AD2_ARCHIVE` (see `.env.example`). No data is included or downloaded by setup commands.

```bash
export MVTEC_AD2_ARCHIVE=data/fabric.tar.gz
export MVTEC_AD2_ROOT=data/mvtec_ad2
uv run python scripts/prepare_mvtec_ad2_archive.py
uv run python scripts/audit_mvtec_ad2.py --out data/audits/ad2-tree.json
```

Before training, record dataset version/archive hashes, category names, relative image paths, split membership, source/acquisition IDs and lighting metadata provenance. AD 2 folder and condition semantics must be verified against its official utilities. Do not repurpose the AD loader by changing only the root path. See [the AD 2 audit notes](../docs/ad2-data-audit.md).
