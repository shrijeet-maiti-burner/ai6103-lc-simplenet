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
