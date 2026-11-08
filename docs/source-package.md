# source package

the submission zip contains the committed project and pinned simplenet source, including licences and the environment lock. it excludes git history and untracked files: datasets, weights, personal research and generated results stay outside it. this is a source package; report figures and trained-model evidence still require the documented artifacts.

## build

from a checkout with committed tracked changes and the initialized submodule:

```powershell
uv run --frozen python scripts/build_source_bundle.py --out results/submission/source.zip
```

the builder refuses an existing output, tracked changes or a different simplenet pin. it reads committed git objects, fixes zip timestamps, preserves committed executable flags and writes `SOURCE_BUNDLE.json` with both commits and every included file's sha-256. repeating the build from the same commits produces identical bytes with the same python/zlib runtime. fingerprints detect changed files; they do not authenticate the publisher. unix permissions also depend on the unzip tool; python's `ZipFile.extractall` does not restore executable bits.

## run after extraction

extract into an empty folder. git is unnecessary for execution; do not initialize another repository to make the code work. install the pinned [environment](setup.md), selecting one runtime extra:

```powershell
uv sync --frozen --extra cpu
uv run --frozen --extra cpu python -m lc_simplenet doctor
uv run --frozen --extra cpu python scripts/check_repository.py
uv run --frozen --extra cpu python scripts/smoke_upstream.py --device cpu
uv run --frozen --extra cpu pytest
```

the smoke run uses random weights and generated inputs; it downloads neither a dataset nor pretrained weights and establishes no benchmark result. environment installation needs network access or a populated package cache. choose `cuda` instead of `cpu` for gpu runs. obtain datasets and pretrained weights separately as described in the training/data instructions.

training and scoring records label this source as `source_archive`, retain both commits and record its manifest hash and changed/missing files. an archive never inherits an enclosing repository's identity. modifications are recorded as dirty; keep the original zip for comparison. bundle-construction tests need git; running the extracted package does not.
