# native public localization

`scripts/evaluate_public_localization.py` evaluates d's saved, input-size smoothed maps against unchanged native masks. it uses pil bilinear interpolation to restore continuous maps, rejects cropped coordinates, and checks image ids, label folders, dimensions and binary masks. it applies no test-fitted normalization, morphology or pixel threshold.

```powershell
uv run --frozen --extra cpu python scripts/evaluate_public_localization.py --run-dir results/public-evaluation/runs/can-B0-seed0 --root data/evaluation-ad2 --out results/public-evaluation/native/can-B0-seed0.json --fpr-limit 0.05
```

pixel auroc and au-pro use all native pixels. connected defect regions have equal weight and eight-neighbour connectivity. exact score ties move together. au-pro integrates the pro curve by trapezoids, interpolates at fpr 0.05 and divides by 0.05. missing classes/regions give unavailable metrics. native-size interpolation restores coordinates, not lost feature detail.

## memory and numerical parity

for a native category that exceeds the sparse histogram budget, use the disk-backed engine:

```powershell
uv run --frozen --extra cpu python scripts/evaluate_public_localization.py --run-dir results/public-evaluation/runs/can-B0-seed0 --root data/evaluation-ad2 --out results/public-evaluation/native/can-B0-seed0.json --engine external --histogram-mib 64
```

this preserves the same float32 scores, ties, native masks and integration rule. upper score bits partition temporary files; all remaining bits are retained. one partition is reduced at a time, using 24 mib of persistent count/region-weight histograms. the histogram budget is not a bound on total process memory: native image decoding, connected-component labels, sorting and reduction buffers also need memory. temporary storage uses four bytes per normal pixel and twelve per defective pixel. check free disk space before running a full category.

the default work directory is the output filename with `.buckets` appended; `--work-dir` selects another new directory. existing directories are rejected. successful evaluation removes only the bucket files created by that accumulator. failed attempts retain their temporary files for inspection; no recursive deletion is performed. disk-write failures invalidate the accumulator rather than returning partial metrics. output records the engine source hash and temporary byte count.

the external engine was checked against the sort reference on continuous/tied arrays, signed zero, subnormal values, unequal region sizes and different partition widths. a separate 40-case check against the official utility gave maximum au-pro difference `1.86611e-9`; pixel auroc differed from scikit-learn by at most `6.66134e-16`. these checks use fpr limit 0.05.

one real can/B0/seed-0 run covered 162 images and 370,262,016 native pixels. external and sparse results differed by `4.44090e-16` in pixel auroc and `6.24501e-17` in au-pro. the external process used 262,418,432 bytes peak rss, 1,481,170,848 bytes temporary disk and 95.99 seconds in this local verification. the previous sparse histogram arrays alone used 1,789,550,308 bytes; its full process peak was not measured. this is one numerical/resource check, not a throughput comparison or detection improvement. verified external source sha256: `e5094f190863811127bcfb94ba539c254040b291c67fae8a554dce565bf34501`.

ad 2 categories contain roughly 255–782 million public native pixels. retaining and sorting every image together can exceed the tc2 memory allocation. the cli processes one image at a time and accumulates sparse histograms keyed by the exact ordered float32 bits, storing only occupied bins within each block. the blocks group identical stored values without quantizing scores. signed zero is one tie. normal/defect counts are uint64 and region weights accumulate in float64.

`--histogram-mib` bounds persistent histogram arrays (default 2048 mib); pending image updates can temporarily duplicate those arrays, and one image's labels/sort needs additional memory. exceeding the budget fails explicitly instead of silently downsampling. a native can run exceeded the original 1024 mib budget, which was retained as a failed resource attempt; the metric definition stayed unchanged. the small-array sort implementation in `localization.py` remains an independent reference; it also accepts float64 scores. the streaming path rejects float64 input instead of silently rounding it.

on 3 november 2026, 40 synthetic continuous/tied cases were compared with mvtec's downloaded ad evaluation utility v1.0 at integration limit 0.05. maximum au-pro difference was `4.60085e-9`, and maximum pixel-auroc difference versus scikit-learn was `2.22045e-16`. the official utility stores fpr in float32; our reference uses float64. the streaming path is additionally checked against the sort reference, including signed zero and unequal region sizes. these are implementation checks, not dataset performance or private-server certification.

reference archive sha256: `dfcda7d67eee25316ec6ae5042c0b1684a4cabf33b2346be351e2ce36013f220`. pro source sha256: `80feff3b5c96023a93ec3a28494583c7732f639547f572982952bb6d0469a29d`. integration source sha256: `96ed3135f23f49a75a767891dd1e3c975973590af31cad0cc39f655f6d76fef2`. the utility is retained locally; no third-party source was vendored or modified.

the output records input score/map hashes, ordered image-id hash and each defective image's native mask hash. existing outputs are never overwritten. evaluate every declared method/seed; no model, threshold or postprocessing setting may be chosen from these outcomes. local public-subset metrics and the [official ad 2 private server](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) are separate evidence.

sources: [mvtec ad evaluation code](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), [ad 2 metric definition](https://arxiv.org/html/2503.21622v1). the v1.0 utility's default 0.30 integration limit is replaced explicitly by the ad 2 limit 0.05; the upstream approximate/dilated pro helper is not used.
