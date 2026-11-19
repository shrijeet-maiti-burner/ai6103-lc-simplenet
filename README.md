# LC-SimpleNet

Paired relighting and patch-score consistency for industrial anomaly detection, based on [SimpleNet](https://arxiv.org/abs/2303.15140) (CVPR 2023).

We compare the original Gaussian/hinge objective, lighting augmentation, score consistency and a follow-up that routes the consistency gradient through the adapter. Training, calibration and evaluation remain separate.

The [completed results](docs/routed-results.md) include all controls and seeds. On the eight-category MVTec AD 2 public subset, R0 reduces lighting false alarms but loses too much defect recall. On the separate AeBAD blade experiment, the three-seed averages improve against our B0 under illumination and same conditions, with seed and background failures. The incremental gain over the stronger synthesis controls is uncertain.

## Run

Install Python 3.12.15 and [uv 0.13.0](https://github.com/astral-sh/uv/releases/tag/0.13.0). From a clone:

```bash
git clone --recurse-submodules https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet.git
cd ai6103-lc-simplenet
uv sync --frozen --extra cpu
uv run --frozen --extra cpu python -m lc_simplenet doctor
uv run --frozen --extra cpu pytest -q
uv run --frozen --extra cpu python scripts/smoke_paired.py
```

For the submission ZIP, extract it and run the same commands from its root, starting at `uv sync`. It includes the pinned upstream source; Git is not needed to run it. Use `--extra cuda` instead of `--extra cpu` for the CUDA 12.4 environment. See [setup](docs/setup.md).

Check the included aggregate results without installing dependencies:

```bash
python scripts/verify_routed_results.py evidence
```

## Reproduce

| Task | Instructions |
|---|---|
| Original method and paired objective | [method](docs/project.md), [paired implementation](docs/paired-lighting.md) |
| Normal data and image geometry | [data contract](docs/data-contract.md), [acquisition and manifests](docs/data.md) |
| Original single-view baseline | [training](docs/baseline-training.md), [AD reproduction](docs/ad-reproduction-results.md) |
| Paired B0/B1/B2 training | [runner](docs/pilot-training.md), [AD 2 study](docs/ad2-study-results.md) |
| Routed follow-up | [objective](docs/routed-consistency.md), [AD 2 launcher](docs/routed-reproduction.md), [blade launcher](docs/routed-blade-reproduction.md) |
| Image and native localization metrics | [protocol](docs/protocol.md), [localization](docs/localization.md) |
| Interactive inspection and recorded gallery | [demo](docs/demo.md) |
| Results for the report | [evidence index](docs/report-evidence.md), [completed follow-up](docs/routed-results.md) |
| Build the submission ZIP | [source package](docs/source-package.md) |

Datasets and pretrained/trained weights are external. The two `execution-sources/` archives preserve the numerical workers used by the follow-up experiments. Full routed training requires the documented Linux/L40S environment. One complete AD 2 category/seed passed the portable launcher, including resume, exact calibration replay and public scoring; see the [verification record](docs/routed-reproduction.md). The full AD 2 matrix and blade workflow have not been rerun through these launchers.

## Source

`src/lc_simplenet/` contains our integration and objectives; `scripts/` contains runnable commands; `tests/` checks numerical and data contracts. The original SimpleNet submodule remains at `351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b`. See [upstream integration](docs/upstream.md) and [attribution](THIRD_PARTY.md).

Repository and commit history: [github.com/shrijeet-maiti-burner/ai6103-lc-simplenet](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet). The ZIP's `SOURCE_BUNDLE.json` identifies its exact commit and file hashes. The report and presentation video are separate submissions.
