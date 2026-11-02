# Setup

Use Git, Python 3.12 and uv 0.13.0. `.python-version` selects Python 3.12.15. The committed `uv.lock` fixes transitive dependencies and wheel sources. Python 3.10's global PyTorch installation was not used or changed.

## uv 0.13.0 pin

`[tool.uv] required-version = "==0.13.0"` in `pyproject.toml` is exact. A newer PATH binary (for example 0.13.5) refuses `uv sync`. Do not loosen the pin and do not run `uv self update 0.13.0` if other projects need the newer binary.

Keep a 0.13.0 executable beside PATH and call it explicitly:

```bash
# one-time install of the 0.13.0 binary into a local directory, then:
export UV="$HOME/.local/uv-0.13.0/uv"
"$UV" --version    # must print 0.13.0
"$UV" sync --frozen --extra cpu
```

Leave the global `uv` on PATH unchanged.

```powershell
git submodule update --init --recursive
uv sync --frozen
uv run python -m lc_simplenet doctor
uv run python -m lc_simplenet plan
```

On this Windows machine uv is available as `py -3.12 -m uv`; substitute that for `uv` if needed. For example: `py -3.12 -m uv sync --frozen`.

## Model runtime

For an NVIDIA GPU on Windows or Linux, the pinned option is PyTorch 2.6.0 / torchvision 0.21.0, CUDA 12.4 wheels. These versions are selected for reproducibility, not as a claim about the latest release. The driver must support that runtime. macOS uses the CPU option.

```powershell
uv sync --frozen --extra cuda
uv run --frozen --extra cuda python -m lc_simplenet doctor --require-cuda
uv run --frozen --extra cuda python scripts/smoke_upstream.py --device cuda
uv run --frozen --extra cuda pytest
```

CPU alternative (mutually exclusive with `cuda`):

```powershell
uv sync --frozen --extra cpu
uv run --frozen --extra cpu python scripts/smoke_upstream.py --device cpu
uv run --frozen --extra cpu pytest
```

Use the selected extra in subsequent `uv run` commands. Running without it can remove optional runtime packages. Alternatively, after syncing, call `.venv\Scripts\python.exe` directly on Windows or `.venv/bin/python` on Linux/macOS. No activation or global package edits are required.

The upstream README describes Python 3.8 / torch 1.12.1 / torchvision 0.13.1. Our modern environment is a porting baseline, not an exact reproduction of that historical environment. Imports and the random-weight smoke check do not validate original benchmark performance. The legacy PRO path also needs documented changes for removed NumPy/pandas APIs; do not enable compatibility by silently monkeypatching them.

Official version pairing: [PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/). Exact pins and package indexes are in `pyproject.toml` and `uv.lock`.

## Editor and checks

Open `ai6103-lc-simplenet.code-workspace`. It selects `.venv/Scripts/python.exe` on Windows, pytest, Ruff and excludes datasets/results from search. On Linux/macOS select `.venv/bin/python` once.

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/check_repository.py
git diff --check
```

CI runs lightweight checks on Windows and Linux and the optional CPU scientific checks on Linux. No CI job downloads a dataset or pretrained model. The [single-view baseline](baseline-training.md) and [paired pilot](pilot-training.md) are available. The full experiment matrix, AD 2 evaluation loader and official regional metric remain pending. See the [first gate](project.md#first-feasibility-gate).
