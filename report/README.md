# Report

Build from this directory with TeX Live or another pdfLaTeX/BibTeX installation:

```text
pdflatex -halt-on-error main.tex
bibtex main
pdflatex -halt-on-error main.tex
pdflatex -halt-on-error main.tex
```

The original AAAI style and bibliography style are included unchanged. The figure is an embedded-font vector PDF; building needs no Python, model weights or datasets.

The [report release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/report-final-2026-11-19-r4) contains the PDF, source ZIP, checksums and validation record. The separate [code release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/code-final-2026-11-19) remains the code submission. Full category/domain/seed counts are in the [frozen result evidence](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/tree/4c2b7c8331d7e7a5f7af694f69269cb6fbef0b4c/evidence) and [results release](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-results-2026-11-18).
