# ad2 second-stack replicate

72 completed normal-only runs: eight categories, b0/b1/b2 at lambda 0.01, seeds 0–2. this is an independent software-stack replicate, separate from the locked main study. no test scores, checkpoints or replay verification are supplied.

raw outputs and generated normal manifests are in the [checksummed artifact release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad2-replicate-2026-11-02), outside source control. [summary.csv](summary.csv) retains 24 category/method rows, each averaging three seeds. values are pair mse, normal patch-score std and synthetic hinge; none is defect recall.

## findings and limits

b1 lowers normal pair mse versus b0 in every category, by 5.915–132.066 times. b2 at 0.01 does not pass the main study's full-length safeguard: fabric's synthetic-hinge ratio versus b1 is 1.45205 and vial's is 1.36922, both above 1.25. these proxies neither establish nor refute real-defect retention. no weight is chosen from test outcomes.

all runs use python 3.11.16, torch 2.11.0+cu128 and an a40. the main study uses the locked python 3.12 / torch 2.6+cu124 environment. every replicate run records a dirty source tree and no replay. source hashes remain in the artifact metadata, but the dirty/untracked training state is not fully reconstructible from the named git commit. exact independent reproduction is therefore not established.

the three can b0 runs used four threads and a 0.8 gpu-memory fraction; seeds 0 and 1 had the decoded-image cache off, while seed 2 used it. the remaining 69 runs used two threads, caching and a 0.25 fraction. retain that distinction for any runtime comparison. it is not evidence of training-speed improvement.

## retrieve and audit

```powershell
gh release download dev-ad2-replicate-2026-11-02 --pattern 'ad2-replicate-2026-11-02.zip*' --dir results/replicate-release
Get-FileHash results/replicate-release/ad2-replicate-2026-11-02.zip -Algorithm SHA256
Expand-Archive -LiteralPath results/replicate-release/ad2-replicate-2026-11-02.zip -DestinationPath results/replicate-release/unpacked
```

expected zip sha256: `f55166201ed52521f6b92402f2dfa40abc348c12a2ecac07d82e0e5b7a3511b9`. compare before extraction. the archive contains 546 original artifact/manifests files plus `artifact-manifest.json`, which records each original and shared file hash.

72 historical `run.json` files have absolute home-directory paths replaced with `<local-path>`. all remaining 474 files, including every validation score, diagnostic, result, log and patch-map archive, retain their original bytes. the complete unredacted source snapshot at `ab0313d03c5769f004febf54b82dd1b0169ba2ae` is preserved privately, with sha256 `b853425ebc5ca02953d2eec2ea98c3577d61083be9dba23620f65dc5fce23a96`. redacted metadata is labelled and is not an executable resume configuration. historical checksum files refer to their original packages; use the new artifact manifest for the shared archive.

generated manifests are under `references/ad2-normal-manifests/` inside the archive. historical cluster helpers still refer to that path; restore them locally if inspecting those helpers. they also depend on local task queues and site modules, so they are not a portable rerun command. use the standard trainer and frozen policy for a new run, record the runtime, and keep outputs ignored.

data attribution: [mvtec ad 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2), cc by-nc-sa 4.0. method: [simplenet](https://github.com/DonaldRR/SimpleNet). no dataset images or model weights are in this release.
