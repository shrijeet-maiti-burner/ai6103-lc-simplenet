# Lighting inspection

a local inspection demo for the released ad 2 study. it compares b0, b1 and a labelled b2 ablation on one normal-validation or locally supplied image under a controlled lighting change. it does not select a model or estimate held-out defect recall.

## run

use the [locked environment](setup.md). download the `dev-ad2-study-2026-10-31` release zips and both `.sha256` manifests into an ignored folder. stage only paired artifacts:

```powershell
uv run --frozen --extra cuda python scripts/prepare_study_artifacts.py --archives results/release-ad2-20261031 --out results/study-artifacts
uv run --frozen --extra cuda python scripts/serve_demo.py --runs results/study-artifacts/paired --root data/mvtec_ad2 --device cuda
```

open `http://127.0.0.1:18763`. `--device cpu` works with the cpu environment. the server binds loopback only, uses no external scripts/fonts, and sends no data to external services. stop with ctrl+c. choose another `--port` if occupied; it does not take over an existing listener. no public hosting is configured.

## controls and evidence

- category, seed, normal-validation image, b2 weight, brightness/gamma and overlay opacity. the default is the first sorted fabric validation image at seed 0 and b2 weight 0.1. this is a presentation default, not a selected configuration.
- live inference uses the released heads and a shared frozen embedding with the recorded encoder fingerprint. a saved normal-validation image guards the model path: score errors above `1e-3` reject inference. dataset views are also checked against their own saved reference scores.
- live mode accepts one local, upright, single-frame rgb or grayscale png/jpeg (16 mib and 16 megapixels maximum; no transparency). select its category first. the image stays in process memory, receives an unknown label and is identified by its bytes, not its filename. the saved-normal replay guard remains separate from the uploaded image. exports contain the resized view and model outputs; keep them local. offline playback cannot inspect new uploads.
- each model retains its own validation-calibrated threshold. displayed sigma margins use that model's validation image-score standard deviation; they are neither probabilities nor uncertainty intervals.
- the score-units control rescales displayed scores, thresholds and reference changes together by 1/16, and measured patch mse by 1/256. flags, standardized margins and fixed-scale maps remain unchanged. mse uses scientific notation so small positive values do not appear as zero. this is an analytic change of units applied to real predictions, not another trained model or a quality improvement. exports retain original measurements and add the display-conversion factors.
- maps use per-model second/98th percentiles of saved validation patch scores as fixed display limits. map colors cannot be compared as equal raw scores across independently trained models. changing overlay opacity changes visualization only.
- a nine-point sweep uses brightness/gamma values `{0.8,1.0,1.2}`. every tile is measured. its purpose is to inspect synthetic-light sensitivity on one image. a flag is a false alarm only for a known normal-validation input; an upload has no verified defect label.
- prediction exports contain image/checkpoint/backbone hashes, transform, score, threshold, calibration counts and replay error. newly computed, cached and recorded outputs are explicitly distinguished. timing is the original measured view computation, including one or two shared encoder passes, three heads and map rendering. decoding, lazy model loading and the separate upload guard are excluded; it is not benchmark latency.
- b2 weights are labelled ablations because no tested weight passed the study's normal-only safeguards.

## offline presentation fallback

record the first sorted normal-validation image and first seed of every category, at all available b2 weights and the fixed nine lighting settings:

```powershell
uv run --frozen --extra cuda python scripts/record_demo.py --out results/demo-recording.json
uv run --frozen --extra cpu python scripts/serve_demo.py --recording results/demo-recording.json
```

the recording contains genuine measured images/maps and provenance. the interface says `recorded predictions`; sliders snap to the recorded three-point grid. unrecorded settings fail instead of being interpolated into fabricated inference. keep the recording and dataset-derived images outside git. the offline mode needs the cpu runtime but no gpu, network or model download.

## Recorded blade comparison

The separate blade gallery retains all 1,639 real test images, 12 methods and three seeds. It filters recorded decisions, including recovered and missed defects; it performs no live inference or uploads. Build it from the retained audited recording, native test packets and score files:

```bash
uv run --frozen --extra cpu python scripts/build_blade_gallery.py --evidence-dir results/blade-evidence --out results/blade-gallery
```

Open the generated index.html locally. The code ZIP contains the template and renderer; aggregate CSVs alone cannot reconstruct image cards. Generated thumbnails remain local under the AeBAD licence. The [complete result](routed-results.md) retains stronger controls, seed failures and the background regression.

## Checks

tests cover frozen calibration, fixed map scales, invalid requests, exact reference replay, cached/recorded labels and rejection of missing recorded settings. real released-head checks additionally cover the local gpu. isolated headless chromium checks cover controls, exports, all 27 sweep tiles, fixed thresholds/scales and desktop/mobile layouts. screenshots were inspected at 390, 760, 1280 and 1600 pixels. this verifies the tested browser, not every browser or operating system.

data attribution: [mvtec ad 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2); method: [simplenet](https://arxiv.org/abs/2303.15140). preserve the dataset licence and upstream attribution when sharing presentation artifacts.
