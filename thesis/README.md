# Bachelor Thesis - LaTeX source (Overleaf-ready)

**Title:** Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education
**Format:** Royal University of Phnom Penh (RUPP) bachelor thesis.

## How to compile (Overleaf)

1. Upload this whole `thesis/` folder to a new Overleaf project (or zip and import).
2. **Set the compiler to XeLaTeX**: Overleaf → *Menu → Compiler → XeLaTeX*. (XeLaTeX is required - the Khmer abstract and Khmer examples will not render under pdfLaTeX.)
3. **Khmer font:** the document uses **Noto Sans Khmer**. Overleaf includes the Noto family; if the
   Khmer text shows as boxes, upload `NotoSansKhmer-Regular.ttf` to the project root (free from
   Google Fonts) - `fontspec` will pick it up.
4. Build order is handled by Overleaf automatically (xelatex → bibtex → xelatex → xelatex). Locally:
   ```
   xelatex main && bibtex main && xelatex main && xelatex main
   ```

## Local build note
This project was authored on a machine without XeLaTeX, so it has **not** been compiled locally - it
is structured to compile cleanly on Overleaf. If a font error appears, it is almost always the Khmer
font (step 3 above) or the Latin main font; you can replace `TeX Gyre Termes` in `main.tex` with any
installed serif.

## Placeholders to fill in (search for `[` brackets)
- Author (Phork Norak) and supervisor (Dr. Khim Chamroeun) are already filled in.
- `frontmatter/titlepage.tex`, `committee.tex`, `statements.tex`: **other examination-committee
  names, department/programme name, month/year**.
- `frontmatter/abstract_kh.tex`: **proofread the drafted Khmer abstract** (machine-assisted draft).
- `chapters/ch4_experiments.tex` (§4.5): **the live demo URL**.
- `appendices/appendix.tex` (§D): **1–2 prototype screenshots**; (§E) **IRB/consent details**.

## Structure
- `main.tex` - document setup (XeLaTeX, polyglossia EN+Khmer, natbib/apalike) and includes.
- `frontmatter/` - title, committee, Khmer + English abstracts, supervisor/candidate statements,
  acknowledgements.
- `chapters/` - Ch1 Introduction, Ch2 Literature Review, Ch3 Methodology, Ch4 Experiments & System,
  Ch5 Results & Discussion, Ch6 Limitations & Future Work, Ch7 Conclusion.
- `appendices/appendix.tex` - hyperparameters, compute, attribution comparison, heatmaps, ethics.
- `figures/` - figures reused from the paper (faithfulness, leakage, heatmaps, curves).
- `refs.bib` - bibliography (author-year; same entries as the paper).

## Consistency
All numbers match the audited results in `../results_stats/` and the paper/slides: uncalibrated
headline QWK 0.795 / 0.845 / 0.820 / 0.842 (comparable, 0.05 band); LLM deployment 67% exact, 79% within
±1; occlusion faithful, attention configuration-dependent; question leakage 0.76 → 0.35.
