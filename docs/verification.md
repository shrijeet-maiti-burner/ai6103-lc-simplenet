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

These publication checks use synthetic fixtures and no dataset images or pretrained downloads. They do not establish trained-detector accuracy, padding benefits, official metric parity or private-server performance. Trainer integration and the AD 2 evaluation loader remain pending.

review follow-up, 26 october: the builder now rejects unsupported modes before manifest creation, matching the reader. four regression cases cover `RGBA`, `P`, `LA` and `I;16`; the full cpu suite passes 31 tests. lint, formatting, markdown/source checks and `git diff --check` also pass. crop coordinates remain outside the full-frame restoration helper's contract.

## Single-view baseline, 27 October 2026

The rebased B branch passes 34 CPU tests locally; that environment uses SciPy 1.16.3 instead of the locked 1.15.3. Ruff/format, Markdown/source checks, doctor and `git diff --check` pass. The original submodule remains unchanged. CUDA training used the committed direct dependency pins, including SciPy 1.15.3.

All three fixed-final bottle runs completed. Their [development results](baseline-results.md) record measured training cost, exact image-score reloads and the three-seed artifact audit. These checks do not establish test accuracy or full main-study provenance.
