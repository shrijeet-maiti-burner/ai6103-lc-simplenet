# LC-SimpleNet

AI6103 group project: test whether paired relighting and patch-score consistency reduce lighting-induced false alarms without reducing defect recall.

**Status: training and public-subset evaluation complete; official private-server evaluation remains open.** The [45-run AD reproduction](docs/ad-reproduction-results.md) and [120-run paired AD 2 study](docs/ad2-study-results.md) are recorded. AD 2 evidence covers normal-validation diagnostics and separately labelled public image/native-localization results. Relighting improves localization but reduces nonregular-light defect recall; the study establishes no overall detection improvement. A [local inspection demo](docs/demo.md) and [report draft](report/README.md) use the released evidence. Datasets and model weights are excluded from Git. The selected source is [the revised proposal](references/proposal.pdf); older chat suggestions do not override it.

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

- [coding freeze](docs/code-freeze.md): 16 november source checks, evidence boundaries and final handoff.
- [simplenet recovery plan](docs/simplenet-recovery-plan.md): the proposed teacher-anchored follow-up, matched controls and acceptance gates.
- [source zip](docs/source-package.md): build and run a complete package without git history.
- [AD 2 launcher](docs/routed-reproduction.md) and [blade execution source](docs/routed-blade-reproduction.md): input preparation, fixed-budget training/resume, scoring and audit for both routed experiments.

- [Project and method](docs/project.md): scope, objective, controls, first feasibility gate.
- [Evaluation protocol](docs/protocol.md): splits, thresholds, seeds, metrics, evidence.
- [Course requirements](docs/requirements.md): rubric and deliverables.
- [Team allocation and notes](docs/research/README.md): current workstreams and research templates.
- [Normal data and map contract](docs/data-contract.md): split enforcement, batch layout and native-coordinate scores.
- [Pilot training](docs/pilot-training.md): matched controls, local run command, checkpoints and evidence limits.
- [Manifest commands](manifests/README.md): normal-data inventory and holdouts.
- [Single-view baseline](docs/baseline-training.md): fixed-final AD training, outputs and reload limits.
- [Baseline results](docs/baseline-results.md): measured training cost, artifact checks and evaluation handoff.
- [Source review](docs/source-review.md): supplied materials and discrepancies.
- [Decisions](docs/decisions.md): what is fixed and what remains open.
- [Verification](docs/verification.md): actual setup checks and limits.
- [AD 2 data audit](docs/ad2-data-audit.md): archive extract, Table IV expected counts, and the formal study's 256×256 full-frame resize, separate from the fabric smoke check.
- [Score-field validation](docs/score-field-validation-results.md): low-frequency residual on frozen B0/B1 anomaly maps. Validation-only; k=4 is the only eligible grid and the relight result is negative.
- [local inspection demo](docs/demo.md): released-head comparison, frozen calibration and recorded fallback.
- [evaluation hand-off](docs/evaluation-handoff.md): staged public data, lighting groups and complete macro/recall reporting.

## Structure

```text
configs/             draft study configuration
docs/research/       individual research notes
references/          selected proposal and source checksums
report/              course-supplied AAAI style and evidence-backed draft
demo/dist/           local inspection interface; generated records remain ignored
src/lc_simplenet/    normal data, geometry, relighting, loss, metrics, diagnostics
manifests/           manifest commands; generated data remains ignored
tests/               checks for the implemented scientific contracts
third_party/SimpleNet/  original code, pinned as a Git submodule
data/                local dataset instructions; images ignored
results/             evidence convention; run outputs ignored
```

The complete, separately labelled RTX public [image](docs/ad2-public-rtx-results.md) and [native localization](docs/ad2-native-public-results.md) matrices are available. The [primary A40 image-output/source audit](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/18#issuecomment-6605059690) also passed; official private-server evaluation remains pending. The pinned upstream training entry point selects checkpoints using test labels and must not be used unchanged for this study. The single-view runner calls only its discriminator-training loop; the paired runner reuses its components with a fixed-step loop. See [the audit](docs/source-review.md#upstream-code).

The working project due date is **23 November 2026**. The [submission checklist](docs/submission.md) records the current report, video and code requirements.
