# report draft

`main.tex` uses the supplied, unchanged `aaai24.sty` and `aaai24.bst`. the current draft reports measured ad reproduction and ad 2 normal-validation evidence. it explicitly leaves real-lighting defect results pending. it is not a submission-ready report.

the five official names are confirmed. matriculation numbers will be added by shrijeet. the course limit is six body pages, excluding references; retain the original layout and readable figures.

## rebuild

download the `dev-ad2-study-2026-10-31` release zips and checksum manifests into `results/release-ad2-20261031/`. download `ad-reproduction-2026-10-29.zip` and its `.sha256` file from the `dev-ad-reproduction-2026-10-29` release into the same folder. [stage the paired runs](../docs/demo.md), then run from the repository root:

```powershell
uv run --frozen --extra cpu python scripts/build_report_evidence.py --runs results/study-artifacts/paired --ad-release results/release-ad2-20261031/ad-reproduction-2026-10-29.zip
Push-Location report
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
bibtex build/main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
Pop-Location
```

the evidence builder checks all 120 paired identities, verifies the ad archive checksum and saves hashes beside generated tables/figures. output is `report/build/main.pdf`; all generated files stay ignored. see the [claim-to-evidence ledger](../docs/report-evidence.md).

## remaining gates

- finish the frozen ad 2 fpr/recall matrix, with b1 versus b0 and every b2 weight versus both.
- verify native-coordinate pixel metrics and au-pro-0.05 against the official utility; keep private-server outputs separate.
- add success/failure examples selected by a fixed rule, with unchanged calibration and display scales.
- incorporate the team's verified contributions and the missing matriculation numbers.
- remove the draft notice only after auditing every claim against saved evidence. inspect every rendered page again after results change pagination; reject overfull boxes, unresolved citations and illegible plot labels.

the five-minute recording should explain the problem, original method, controlled change, measured result and limitation as one argument. demonstration timing is not a formal inference-throughput benchmark.
