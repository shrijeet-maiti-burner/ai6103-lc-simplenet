# robustad normal-training inventory

9 november 2026. the [author release](https://huggingface.co/datasets/AmazonScience/RobustAD/tree/82c642b2d18be60f2dab104f8dd4dea1f4ccfacb) provides a separate source/target-domain robustness benchmark under cc by 4.0. pcb and metal parts support localization/classification; piled bags is a count-based classification task. keep that task difference explicit.

the training folders contain anomalies. the pinned training metadata has 741 records; the normal-only selection contains 540, excluding all 201 labelled anomalies:

| category | normal training images | excluded training anomalies |
|---|---:|---:|
| metal parts | 180 | 51 |
| pcb | 180 | 75 |
| piled bags | 180 | 75 |

## reproduce the metadata audit

```powershell
python scripts/audit_robustad_metadata.py --fetch-source results/robustad-metadata-001 --out results/robustad-normal-manifest-001.json
```

this retrieves only the pinned repository inventory, three training metadata files and author card. it downloads no image or mask pixels and scores no model. the audit verifies the revision, exact metadata hashes, explicit normal labels, training paths, absence of a defect mask and author lfs sizes/hashes. piled bags' classification-only metadata omits the mask field; pcb/metal parts must provide an explicit null mask for normal records. duplicate identities and inconsistent source records fail.

for existing metadata, use `--source <directory>` instead. it must contain `repository.json` and `<category>-train-metadata.jsonl` for `MetalParts`, `PCB` and `PiledBags`. outputs never overwrite an existing manifest. failed downloads remain available for inspection.

## readiness and limits

the local normal-only acquisition verified all 540 native files against author lfs identities: 1,300,708,942 bytes. its separate pixel audit found no exact native-rgb duplicate groups and no exact native-rgb matches against the available ad2 inventory. these checks do not establish physical-object or capture independence. coarse perceptual hashes grouped similar silhouettes and cannot resolve capture identity.

the metadata command reproduces the selected file identities; it does not reproduce the separate pixel audit. no target-domain test pixels or masks were acquired for that preparation. no robustness result follows from acquiring normal data. this is not a drop-in ad2 loader: input geometry, task scope, fitting/calibration roles, capture limitations and a sealed evaluation protocol still need to be fixed for each new experiment. use no target labels or results to select settings.

the data, acquisition receipts and manifests stay outside git. retain the author card and attribution with any authorized dataset copy. this shared tool publishes the selection rule, not personal experiment plans or raw artifacts.
