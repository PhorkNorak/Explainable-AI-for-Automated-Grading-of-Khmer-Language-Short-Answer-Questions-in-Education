# IEEE manuscript — build instructions

A complete `IEEEtran` (IEEE Access–style) journal manuscript for the Khmer ASAG benchmark.

## Files
- `main.tex` — the manuscript (10 sections + abstract + ethics + references).
- `refs.bib` — bibliography (verified citations).
- `make_figures.py` — regenerates the result figures from the project's CSVs.
- `figures/` — generated charts + copied heatmaps/curves.
- `numbers.md` — provenance: every reported number → its source file.

## Build

### Option A — Overleaf (recommended; no local LaTeX needed)
1. Zip the `paper/` folder (including `figures/`).
2. Overleaf → New Project → Upload Project → select the zip.
3. Set the main document to `main.tex`, compiler **pdfLaTeX**. Compile.

### Option B — local TeX (if installed)
```bash
python make_figures.py        # (re)generate figures from results
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Before submission — author checklist
- Fill in **author/affiliation/email** and the **title** placeholders.
- Complete the **Compliance with Ethical Standards** section with the real IRB/consent facts
  (currently marked *[to be confirmed]*) — IEEE requires this for human-subjects research.
- Decide venue framing: **IEEE Access** (soundness-weighted) or **IEEE TLT** (lead with the
  explainability/leakage insights, since TLT wants insight beyond benchmarking).
- Optional but recommended: deposit the dataset on **IEEE DataPort** for a citable DOI.
- For the strongest version, complete the **encoder/LLM unseen-question + faithfulness** runs on
  GPU (currently labelled *ongoing*) and update Table I/II and the relevant sentences.

## Khmer figure (important)
`figures/heatmap_classical.png` is a **placeholder** rendered by matplotlib, which **cannot shape
Khmer** (subscripts/vowels are misplaced). The correct, browser-shaped version is
`figures/heatmap_examples.html` (and `results_xai/.../heatmaps/*_gallery.html`). **Before final
build:** open `figures/heatmap_examples.html` in a browser and **export/screenshot it to
`figures/heatmap_classical.png`** (overwriting the placeholder). The figure caption already notes
it is browser-shaped. All numeric figures (`fig_cis`, `fig_leakage`, `fig_faithfulness`) contain
no Khmer and are correct as generated.

## Note
This draft reports only computed numbers. Items that depend on GPU runs not done locally are
labelled *ongoing*; the classical confidence interval is on uncalibrated scores (the calibrated
point estimate is 0.864). See `numbers.md`.
