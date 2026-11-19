# Code package

The submission ZIP contains runnable source, the pinned SimpleNet implementation, both frozen execution archives, dependency lock, tests, demo source and aggregate evidence. Report and presentation files are submitted separately. Git history is available at the [repository](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet); `SOURCE_BUNDLE.json` identifies the ZIP's exact commit.

## Build

From a clean committed checkout with its initialized submodule:

```bash
uv run --frozen --extra cpu python scripts/build_source_bundle.py --out results/submission/ai6103-code.zip
```

The builder reads committed Git objects, rejects tracked changes, a wrong submodule pin or an existing output, and excludes untracked files. It fixes timestamps and preserves committed executable flags. Only author-tracked bytecode caches are omitted from upstream; their paths are recorded in the manifest. Repeated builds are byte-identical with the same Python/zlib runtime.

The archive includes:

| Path | Purpose |
|---|---|
| `src/`, `scripts/`, `tests/`, `configs/` | Implementation, commands, checks and frozen settings |
| `third_party/` | Original SimpleNet source and retained licences |
| `execution-sources/` | Exact source snapshots required by both routed launchers |
| `evidence/` | Complete aggregate matrices, summary and checksums |
| `demo/`, `docs/`, `references/` | UI source, reproduction/results guides, proposal and input manifests |

Datasets, weights, generated galleries, run outputs, report/video files and personal notes are excluded. ZIP permission restoration depends on the extractor; Python's `extractall` does not restore executable bits. The documented Python entry points do not need those bits.

## Verify and run

Extract into an empty directory. From its root:

```bash
python scripts/check_repository.py
python scripts/verify_routed_results.py evidence
python scripts/reproduce_routed_ad2.py --source-archive execution-sources/routed-ad2-execution-source-20261116.zip verify
python scripts/reproduce_routed_blade.py --source-archive execution-sources/routed-blade-execution-source-20261118.zip verify
uv sync --frozen --extra cpu
uv run --frozen --extra cpu python -m lc_simplenet doctor
uv run --frozen --extra cpu pytest -q
uv run --frozen --extra cpu python scripts/smoke_upstream.py --device cpu
uv run --frozen --extra cpu python scripts/smoke_paired.py
```

The first four commands use only Python's standard library. The source check hashes every listed bundle file; missing or changed bytes fail. Fingerprints detect changes but do not authenticate the publisher. Compare the outer ZIP checksum with the release as well.

Running needs no Git repository. Bundle-construction tests use Git when available and skip when it is absent. Package installation needs network access or a populated cache. Use the CUDA environment and separately acquired data/weights for real training; see [setup](setup.md).

## Execution snapshots

| Archive | SHA-256 |
|---|---|
| `routed-ad2-execution-source-20261116.zip` | `74f2c9adb729e432237a6879ae82373a3dcee032e5517e7daabb515b5096cb8a` |
| `routed-blade-execution-source-20261118.zip` | `023a644655348d1a0b610e2e7406cced3cae4fbe997ff0e5055573722a047436` |

Their numerical workers and licences retain their original bytes. They are distinct from the newer portable launchers. A full seven-method AD 2 category/seed now passed the portable launcher on L40S, including checkpoint resume, exact normal-calibration replay and native public scoring; see the [verification record](routed-reproduction.md). The complete AD 2 matrix and blade workflow have not been rerun through these launchers.

The [results](routed-results.md) and included aggregate matrices are the evidence for report/demo authors. The aggregate verifier checks counts, means and fixed gates; it does not rerun inference or recompute bootstrap intervals.
