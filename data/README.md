# Local data

Download MVTec AD and AD 2 from [MVTec](https://www.mvtec.com/research-teaching/datasets) under their dataset terms. AD 2 uses CC BY-NC-SA 4.0. Dataset images, masks and archives are ignored by Git.

Place local datasets at `data/mvtec_ad/` and `data/mvtec_ad2/`, or configure an external root in the future loader. No data is included or downloaded by setup commands.

Before training, record dataset version/archive hashes, category names, relative image paths, split membership, source/acquisition IDs and lighting metadata provenance. AD 2 folder and condition semantics must be verified against its official utilities. Do not repurpose the AD loader by changing only the root path.
