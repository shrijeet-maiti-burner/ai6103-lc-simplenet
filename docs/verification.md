# Verification

Verified locally on 16 October 2026 (Singapore). No dataset-backed training or benchmark result is claimed.

| Check | Observed result |
|---|---|
| Supplied-source inventory | 27 files including chat export; 15 lecture PDFs / 1,070 pages; ZIP members individually hashed |
| Preserved proposal and AAAI style/bibliography files | Three byte-for-byte source hashes checked |
| Environment | Isolated `.venv`, Python 3.12.15, uv 0.13.0; frozen lock resolves 86 packages across optional environments |
| NVIDIA runtime | RTX 3060 Laptop GPU, 6 GB; torch 2.6.0+cu124, torchvision 0.21.0+cu124 |
| CUDA diagnostic | Matrix forward/backward and synchronization passed |
| Original SimpleNet architecture smoke | Random weights, four 64px views, 256 patch scores; adapter/discriminator gradients nonzero; backbone gradients absent |
| Scientific helper tests | Six passed: spatial relighting, input validation, fixed operating point/counts, ties/unknown labels, analytic two-branch MSE gradients |
| Ruff lint and format | Passed for all project Python; original submodule excluded |
| Markdown / source checks | 19 documents and three preserved-source hashes passed |
| Study configuration | Parses; explicitly draft, zero selected-category runs, unresolved values visible |
| Upstream | Exact pinned commit; original checkout unchanged |

The architecture smoke downloads no pretrained weights and uses no images from either dataset. It proves basic compatibility and the intended gradient path, not baseline accuracy, full-resolution feasibility, defect preservation or convergence. The complete upstream training/PRO path is not certified by these checks.

GitHub CI runs the lightweight suite on Windows/Linux and CPU model contracts on Linux. Its status is visible in the repository Actions tab; it is separate from the local checks above.

## Shared data contract, 26 October 2026

Verified in a fresh worktree and CPU environment using the frozen lock: Python 3.12.15, uv 0.13.0, torch 2.6.0+cpu and torchvision 0.21.0+cpu.

- 27 tests passed, covering the original scientific helpers, normal manifest generation/loading and continuous score-map restoration.
- Ruff lint and formatting passed for all 15 project Python files.
- Markdown links and three preserved-source hashes passed; `git diff --check` passed.
- The CPU upstream smoke again produced nonzero adapter/discriminator gradients with the backbone frozen. It used random weights and 64px synthetic inputs.
- The upstream checkout remains at `351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b`, with no source changes.

These publication checks use synthetic fixtures and no dataset images or pretrained downloads. They do not establish trained-detector accuracy, padding benefits, official metric parity or private-server performance. The AD 2 evaluation loader remains pending; pilot integration is checked below.

review follow-up, 26 october: the builder now rejects unsupported modes before manifest creation, matching the reader. four regression cases cover `RGBA`, `P`, `LA` and `I;16`; the data branch's full cpu suite passes 31 tests. lint, formatting, markdown/source checks and `git diff --check` also pass. crop coordinates remain outside the full-frame restoration helper's contract.

## Pilot integration, 26 October 2026

- The full CPU suite passes 33 tests. Six new tests cover exact B1/B2 equivalence at zero consistency weight, cold continuation across an epoch boundary, nonzero consistency updates, aligned photometric pairs, rejected validation/incompatible checkpoint sources, and an update/prediction/cold continuation through the pinned upstream components.
- The upstream test uses random weights and synthetic 64px images, without dataset or pretrained downloads. It checks exact next-update metrics, trainable-state hash and raw predictions after reconstruction; the backbone state remains unchanged.
- Ruff lint/format, local Markdown links, preserved-source hashes and `git diff --check` pass.

The runner and its evidence limits are described in [pilot training](pilot-training.md). These automated checks do not establish pretrained convergence, real-defect recall or benchmark performance. Private dataset runs and personal research artifacts remain outside Git.

review follow-up, 26 october: the combined cpu suite passes 47 tests. optional ad cropping matches upstream input tensors exactly on train and validation fixtures, including odd/even resize differences. cropped controls retain exact zero-weight equivalence and cold continuation. the real upstream random-weight checks cover both geometries and verify that the paired runner leaves backbone batch-norm counters unchanged. cropped maps carry explicit crop coordinates and are not restored as full native maps. ad 2 crop requests and rectangular sources under the square ad recipe are rejected.
