# blade execution source

the [blade source snapshot](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-blade-source-2026-11-18) contains the recorded training workers for all twelve arms, fixed real-image scoring, numerical audit helpers and required source dependencies/licences. it contains input filenames/hashes, not images, weights, checkpoints, private study plans or the ssh operator.

this is a source handoff, not another trained result. it does not reverse the failed source screen/confirmation or the original AD 2 recall failure. the positive blade finding remains exploratory and limited to the recorded three-seed means and conditions.

## requirements

- use the repository or complete submission ZIP; the execution archive is included under `execution-sources/`.
- use the locked Python 3.12 / torch 2.6.0+cu124 / torchvision 0.21.0+cu124 / numpy 1.26.4 / Pillow 11.3.0 environment (`uv sync --frozen --extra cuda`).
- training and scoring use the recorded workers' strict **NVIDIA L40S** guard and Linux `/dev/shm` workspace. their feature cache needs substantially more than a 6 GB laptop GPU. allocate one GPU and enough host/shared memory; the recorded training allocation used 30 GB host memory.
- acquire AeBAD and DTD separately. the manifests identify the exact native PNGs and 256 texture files. `--textures-root` is the folder containing their manifest-relative `images/...` paths.
- provide ImageNet `wide_resnet50_2-95faca4d.pth` separately, sha256 `95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8`.

verify the included archive before executing it:

```bash
python scripts/reproduce_routed_blade.py --source-archive execution-sources/routed-blade-execution-source-20261118.zip verify
```

the launcher verifies the pinned archive and every member before extraction. the worker files are byte-identical to the recorded numerical workers. the dependency export retains the original synthesis, prediction and decoding functions; operator/selection scripts and historical prose policies are omitted. the replacement manifest guard checks the exported runtime itself.

**fresh executions have new protocol identities.** the old study hashes are retained only as provenance; this wrapper does not relabel historical checkpoints as newly executed results. use checkpoints created by this handoff when resuming it.

## stage inputs

```bash
python scripts/reproduce_routed_blade.py --source-archive execution-sources/routed-blade-execution-source-20261118.zip prepare --aebad-archive data/AeBAD.zip --textures-root data/dtd --include-tests --out results/blade-inputs
```

preparation hashes and copies native bytes without decoding target PNGs or masks. without `--include-tests`, it stages source/replay inputs only. training receives only the 128 fit, 64 calibration and 64 procedural-development normal images plus the texture donors. test data never enters a training packet.

## train and resume

run each family (`base`, `routed`, `pure`) for each seed (`0`, `1`, `2`):

```bash
python scripts/reproduce_routed_blade.py --source-archive execution-sources/routed-blade-execution-source-20261118.zip train --family routed --seed 0 --inputs results/blade-inputs --weights weights/wide_resnet50_2-95faca4d.pth --out results/blade-training/routed/seed-0/attempt-1
```

the family controls are fixed:

| family | arms |
|---|---|
| base | B0, B1, B2, L0, L1, L2, L3 |
| routed | R0, R1, R2 |
| pure | P0, P1 |

every full case uses 128 fit normals, batch 4 and 5,120 updates (160 source passes). the recorded workers checkpoint and may return early near their per-packet walltime. a successful packet can therefore be a partial training run: inspect `case.json` and `receipt.json`.

resume a partial case into a new attempt directory, retaining its seed/family:

```bash
python scripts/reproduce_routed_blade.py --source-archive execution-sources/routed-blade-execution-source-20261118.zip train --family routed --seed 0 --inputs results/blade-inputs --weights weights/wide_resnet50_2-95faca4d.pth --resume results/blade-training/routed/seed-0/attempt-1/checkpoint-01024.pt --out results/blade-training/routed/seed-0/attempt-2
```

replace the example checkpoint with the actual last saved file. `--end-step` permits a bounded, multiple-of-32 partial execution; it cannot become a final scoring pack. keep only one completed final case per family/seed under the `--runs` tree; duplicate complete cases are rejected.

## pack, replay and score

```bash
python scripts/reproduce_routed_blade.py --source-archive execution-sources/routed-blade-execution-source-20261118.zip pack --runs results/blade-training --out results/blade-heads
python scripts/reproduce_routed_blade.py --source-archive execution-sources/routed-blade-execution-source-20261118.zip score --pack results/blade-heads --inputs results/blade-inputs --weights weights/wide_resnet50_2-95faca4d.pth --out results/blade-scoring
```

packing requires all nine full-budget cases and validates artifact hashes, encoder/feature identity and normal-only higher-q95 thresholds. it exports 36 heads and their procedural-development reference patches, without optimizers. no method or checkpoint is chosen from target performance; R0 stays fixed.

scoring first replays all 36 development outputs bit for bit. a mismatch stops execution before the four target domains. it then scores every illumination/same/background/view image, exports raw patches/image scores and all 144 domain/method/seed rows, and separately recounts the fixed mean gates and paired exact-RGB-cluster intervals in `audit.json`. source overlap and conflicting labels are recorded without dropping examples. the three trained seeds do not establish new-seed or physical-specimen generalization.

each packet writes an immutable launch, event stream, checksummed artifacts and completion receipt. an interruption does not create a successful receipt. the wrapper currently requires a new scoring output directory for a retry; completed recorded experiments and their earlier transport receipts remain unchanged.

## verification boundary

the release records source/worker identity, input-staging parity and subprocess/failure-guard checks. **a fresh nine-case full-budget training and real-image scoring run through this new local wrapper has not been completed.** the historical experiments ran through the original framed workers and private transport, whose results are reported separately. passing source/mechanics checks is not a new accuracy result or universal hardware certification.

Datasets and pretrained/trained weights remain external. the [results release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-results-2026-11-18) contains the completed original results and limits.
