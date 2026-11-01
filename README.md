# LC-SimpleNet

AI6103 group project: test whether paired relighting and patch-score consistency reduce lighting-induced false alarms without reducing defect recall.

**Status: development training.** Normal manifest loading, paired-view pilot training, checkpoint continuation and normal-only calibration are implemented. Benchmark validation remains pending. No dataset or trained results are included. The selected source is [the revised proposal](references/proposal.pdf); older chat suggestions do not override it.

## Start

```powershell
git clone --recurse-submodules https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet.git
cd ai6103-lc-simplenet
uv sync --frozen
uv run python -m lc_simplenet doctor
uv run python -m lc_simplenet plan
uv run pytest
```

Python 3.12.15 and uv 0.13.0 are the setup baseline. If `uv` is installed only as a Python module, use `py -3.12 -m uv` in place of `uv`. Open [the workspace](ai6103-lc-simplenet.code-workspace) in VS Code. See [setup](docs/setup.md) for CUDA and CPU environments.

## Read

- [Project and method](docs/project.md): scope, objective, controls, first feasibility gate.
- [Evaluation protocol](docs/protocol.md): splits, thresholds, seeds, metrics, evidence.
- [Course requirements](docs/requirements.md): rubric and deliverables.
- [Team allocation and notes](docs/research/README.md): current workstreams and research templates.
- [Normal data and map contract](docs/data-contract.md): split enforcement, batch layout and native-coordinate scores.
- [Pilot training](docs/pilot-training.md): matched controls, local run command, checkpoints and evidence limits.
- [Manifest commands](manifests/README.md): normal-data inventory and holdouts.
- [Source review](docs/source-review.md): supplied materials and discrepancies.
- [Decisions](docs/decisions.md): what is fixed and what remains open.
- [Verification](docs/verification.md): actual setup checks and limits.

## Structure

```text
configs/             draft study configuration
docs/research/       individual research notes
references/          selected proposal and source checksums
report/              course-supplied AAAI style and report outline
src/lc_simplenet/    normal data, geometry, relighting, loss, metrics, diagnostics
manifests/           manifest commands; generated data remains ignored
tests/               checks for the implemented scientific contracts
third_party/SimpleNet/  original code, pinned as a Git submodule
data/                local dataset instructions; images ignored
results/             evidence convention; run outputs ignored
```

The full baseline reproduction, AD 2 evaluation loader and official metrics remain pending. The pilot reuses upstream components and replaces its test-driven training loop. See [the audit](docs/source-review.md#upstream-code).
