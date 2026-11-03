# report draft

`main.tex` uses the supplied, unchanged `aaai24.sty` and `aaai24.bst`. the current draft reports measured ad reproduction, ad 2 normal-validation evidence and complete, separately labelled rtx3060 public image/native-localization matrices. primary-artifact verification remains pending. role d has reported the a40 image matrix in #18; all 150 published means/sds match the local run at their displayed precision. it is not a submission-ready report.

the five official names are confirmed. matriculation numbers will be added by shrijeet. the course limit is six body pages, excluding references; retain the original layout and readable figures.

## rebuild

download the `dev-ad2-study-2026-10-31` release zips and checksum manifests into `results/release-ad2-20261031/`. download `ad-reproduction-2026-10-29.zip` and its `.sha256` file from the `dev-ad-reproduction-2026-10-29` release into the same folder. [stage the paired runs](../docs/demo.md), then run from the repository root:

```powershell
uv run --frozen --extra cpu python scripts/build_report_evidence.py --runs results/study-artifacts/paired --ad-release results/release-ad2-20261031/ad-reproduction-2026-10-29.zip
uv run --frozen --extra cpu python scripts/build_public_report.py --evaluation results/public-evaluation-rtx3060-20261103/evaluation --protocol results/public-evaluation-rtx3060-20261103/frozen-protocol.json
uv run --frozen python scripts/build_native_report.py --native results/native-public-external-20261103 --scoring results/public-evaluation-rtx3060-20261103
uv run --frozen --extra cpu python scripts/build_public_examples.py --runs results/public-evaluation-rtx3060-20261103/runs --root data/evaluation-ad2 --protocol results/public-evaluation-rtx3060-20261103/frozen-protocol.json
Push-Location report
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
bibtex build/main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
Pop-Location
```

the public commands require the retained frozen cohort after [scoring and aggregation](../docs/evaluation-handoff.md); the normal-only training release does not contain public outputs. use your explicitly labelled cohort paths for a40. the public builder rejects missing heads, incomplete seeds and mismatched scopes. examples use the first sorted regular-light defective image per category, seed 0 and every method, without consulting prediction outcomes.

the evidence builder checks all 120 paired identities, verifies the ad archive checksum and saves hashes beside generated tables/figures. output is `report/build/main.pdf`; all generated files stay ignored. see the [claim-to-evidence ledger](../docs/report-evidence.md).

## remaining gates

- compare the primary a40 image matrix with the complete local rtx matrix; retain any numerical/hardware differences.
- audit the [complete native matrix](../docs/ad2-native-public-results.md) and its retained input/source hashes. keep private-server outputs separate.
- inspect the generated all-eight-category gallery and retain its fixed selection rule, calibration and display scales.
- incorporate the team's verified contributions and the missing matriculation numbers.
- remove the draft notice only after auditing every claim against saved evidence. inspect every rendered page again after results change pagination; reject overfull boxes, unresolved citations and illegible plot labels.

the five-minute recording should explain the problem, original method, controlled change, measured result and limitation as one argument. demonstration timing is not a formal inference-throughput benchmark.

## compact image review archive

[download the private rtx review artifact](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad2-public-rtx-2026-11-03). verify its zip checksum and per-file manifest, then extract into an ignored folder. it contains all scores needed to reproduce image tables; all six included evaluation outputs replay byte-for-byte. run/result identity files are labelled projections with personal paths omitted and original hashes retained. the archive does not contain maps or image pixels: rebuilding the qualitative gallery additionally needs retained maps and the public dataset. primary a40 provenance remains a separate review.

the [native review artifact](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/dev-ad2-native-2026-11-03) contains all 120 native outputs, their source/input bindings and the exact evaluator snapshots. run `build_native_report.py` against its `native` and `scoring` directories to reproduce both category scopes. metadata projections explicitly omit historical absolute source paths and retain their original hashes. restoring maps from dataset images requires the released heads and original scoring cohort; aggregation alone requires neither images nor weights.
