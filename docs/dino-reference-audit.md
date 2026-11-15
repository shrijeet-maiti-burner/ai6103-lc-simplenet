# dinov2 reference artifact audit

9 november 2026. independent recount of athul's [reference release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-dino-reference-2026-11-09), following the [reported results](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/issues/26#issuecomment-6730455089). these remain exposed-public references under the shared recovery plan. this audit does not promote a method or demonstrate that our lighting-consistency loss improves simplenet.

## completed checks

- archive sha256 matches the release: `13e3ebe214a7ae350edcec7af638173f9b7b6cbe05802cb68d741403fed4260d`. all 311 members have unique safe paths, zip crc passes, and all 310 internal file hashes match. the checksum manifest is covered by the archive hash.
- all eight categories are retained: two knn banks, six simplenet-v1 arm/seed combinations and nine simplenet-v2 combinations, on 1,084 public images per combination. the 18,428 saved image scores independently reconstruct 1,343 condition rows, including every false-positive/true-positive count, denominator, rate and pairwise/tie-aware image auroc. maximum metric difference is zero.
- every released script matches the script digest recorded in its category results. no training-script identity mismatch was found.
- the 16,260 simplenet image maxima match their saved maps after float16 rounding. this binds quantized maps to the reported image scores; it does not recover full-precision maps.
- native result records have complete status, matching image counts, eight-connected regions and fpr limit 0.05. their category macros are reconstructed from the released metrics. native pixels were not evaluated again.

the shared command was run against the actual archive. it also exposes each condition's contributing categories so a suffix absent from some categories cannot be presented as an eight-category result.

## limits that affect the report

trained heads were not saved. neither the released outputs nor this recount provides cold checkpoint replay. the simplenet files also omit raw normal-validation scores, so their calibration quantiles cannot be independently reconstructed; image metrics here use the recorded frozen thresholds. knn includes its normal-validation scores and its higher-q95 thresholds were reconstructed exactly.

v2 saves only the max-rule test rows. its top-1-percent sensitivity table cannot be independently recounted from those rows. the published score maps are float16, and native evaluation uses those quantized maps with different resolution/geometry and postprocessing from the wrn study. cross-method native numbers therefore do not isolate a backbone substitution.

v2 was designed after v1's public outcomes were seen. its written protocol and complete reporting do not make the public set untouched. physical-lighting generalization and the original consistency contribution still require separate evidence. the release's reported v2 spotlight/gradient arm fails recall preservation against its own unrelit control; an improvement against a weaker or different reference cannot remove that result.

## reproduce

download the reference zip from the private team release using an authenticated account, then run:

```powershell
python scripts/audit_dino_reference_archive.py --archive <reference.zip> --out results/dino-audit-001.json
```

the expected archive hash is pinned by default. the command reads the archive without extracting or executing its scripts, independently recounts the primary image metrics, and refuses to overwrite a receipt. archive-integrity checks remain active under `python -O`. raw outputs, maps and generated audit receipts stay outside git.
