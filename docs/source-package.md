# source package

the submission zip contains the committed project and pinned simplenet source, including licences and the environment lock. it excludes git history and untracked files: datasets, weights, personal research and generated results stay outside it. this is a source package; report figures and trained-model evidence still require the documented artifacts.

## build

from a checkout with committed tracked changes and the initialized submodule:

```powershell
uv run --frozen python scripts/build_source_bundle.py --out results/submission/source.zip
```

the builder refuses an existing output, tracked changes or a different simplenet pin. it reads committed git objects, fixes zip timestamps, preserves committed executable flags and writes `SOURCE_BUNDLE.json` with both commits and every included file's sha-256. repeating the build from the same commits produces identical bytes with the same python/zlib runtime. fingerprints detect changed files; they do not authenticate the publisher. unix permissions also depend on the unzip tool; python's `ZipFile.extractall` does not restore executable bits.

## final routed handoff

the three changes below were reviewed and merged into `main` on 18 november. checking out one of their original branches does not combine them:

| review | contents |
|---|---|
| [#37](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/37) | integrated feature losses, gradient routing and attribution |
| [#39](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/39) | AD 2 and blade input preparation, training/resume, scoring/audit launchers and source instructions |
| [#40](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/40) | completed results, aggregate verifier, recorded blade-gallery source and final handoff status |

build from a clean combined commit containing all three changes. the overlapping README edits have been resolved and the completed-results wording retained. the [code delivery release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/code-submission-2026-11-18) identifies the packaged commit, both execution snapshots, checksums and extraction checks; a later `main` may contain different files.

the generic builder includes committed Git objects, not release assets. the routed launchers also require these code-only execution snapshots:

| archive | source release | sha256 |
|---|---|---|
| `routed-ad2-execution-source-20261116.zip` | [AD 2 source](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-reproduction-source-2026-11-16) | `74f2c9adb729e432237a6879ae82373a3dcee032e5517e7daabb515b5096cb8a` |
| `routed-blade-execution-source-20261118.zip` | [blade source](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-blade-source-2026-11-18) | `023a644655348d1a0b610e2e7406cced3cae4fbe997ff0e5055573722a047436` |

include both archives and their checksums in the final delivery, without adding images or weights. one layout is a `code/` directory extracted from the builder's source zip, a sibling `execution-sources/` directory containing both snapshots, and a short top-level README with commands using `--source-archive ../execution-sources/<archive>`. preserve the inner source manifest and upstream files; do not silently rewrite a published snapshot or its checksum.

from a fresh extraction, verify both archive hashes and each launcher's `verify` command, then run the documented environment, smoke and guard checks. GPU training/scoring requires the documented locked runtime and L40S hardware. a fresh full-budget run through the new portable wrappers remains unverified; report that separately from the completed original experiments and passing mechanics checks.

report/video authors should use [the completed results](routed-results.md) and [aggregate evidence release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-results-2026-11-18). the older PDF/video are review drafts, not updated final deliverables. the exact portal cutoff, sole report submitter, member identifiers, instructor access and final upload receipts remain handoff tasks.

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
