# Setup

Use Python 3.12.15 and uv 0.13.0. The exact uv version is required by `pyproject.toml`; `.python-version` selects the Python version. Install uv from its [versioned release](https://github.com/astral-sh/uv/releases/tag/0.13.0), or use `py -3.12 -m pip install --user uv==0.13.0` on Windows. If another project needs a different uv version, keep a separate executable rather than replacing it.

From a Git clone, initialize the pinned submodule:

```bash
git submodule update --init --recursive
```

The complete submission ZIP already includes that source. Extract into an empty folder and run commands from the root.

## CPU

```bash
uv sync --frozen --extra cpu
uv run --frozen --extra cpu python -m lc_simplenet doctor
uv run --frozen --extra cpu pytest -q
uv run --frozen --extra cpu python scripts/smoke_upstream.py --device cpu
uv run --frozen --extra cpu python scripts/smoke_paired.py
uv run --frozen --extra cpu python scripts/check_repository.py
```

The smoke checks use random weights and synthetic inputs. Installing packages needs network access or a populated cache; these checks download no dataset or pretrained model.

On Windows, `py -3.12 -m uv` can replace `uv`. After syncing, the environment's interpreter is `.venv/Scripts/python.exe` on Windows and `.venv/bin/python` on Linux/macOS.

## CUDA

On Windows/Linux with a compatible NVIDIA driver:

```bash
uv sync --frozen --extra cuda
uv run --frozen --extra cuda python -m lc_simplenet doctor --require-cuda
uv run --frozen --extra cuda python scripts/smoke_upstream.py --device cuda
```

The lock fixes torch 2.6.0 and torchvision 0.21.0 with CUDA 12.4 wheels. CPU and CUDA extras are mutually exclusive. Keep the selected extra in subsequent `uv run` commands; omitting it can remove optional packages.

Full routed runs have stricter requirements: [AD 2](routed-reproduction.md) records L40S execution and exact calibration replay; [blade](routed-blade-reproduction.md) requires Linux, an NVIDIA L40S and sufficient host/shared memory. CPU checks do not certify those full GPU runs.

## Data and editor

Acquire datasets and weights separately using the [data instructions](data.md). Setup does not download them. Local datasets and outputs belong under ignored `data/` and `results/`.

The optional `ai6103-lc-simplenet.code-workspace` configures pytest and Ruff. On Linux/macOS select `.venv/bin/python`. CI runs the CPU suite on Windows and Linux.

The pinned upstream README describes its historical Python 3.8 / torch 1.12.1 environment. Our integration uses the committed modern lock; its [protocol changes](upstream.md) remain explicit.
