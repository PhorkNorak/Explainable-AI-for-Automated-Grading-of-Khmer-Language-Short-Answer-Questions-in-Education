# Bachelor Thesis - LaTeX source (Overleaf-ready)

**Title:** Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education
**Format:** Royal University of Phnom Penh (RUPP) bachelor thesis.

## How to compile (Overleaf)

1. Upload this whole `thesis/` folder to a new Overleaf project (or zip and import).
2. **Set the compiler to XeLaTeX**: Overleaf → *Menu → Compiler → XeLaTeX*. (XeLaTeX is required - the Khmer abstract and Khmer examples will not render under pdfLaTeX.)
3. **Khmer font:** the document uses **Noto Sans Khmer** by default. Overleaf includes the Noto
   family; if the Khmer text shows as boxes, upload `NotoSansKhmer-Regular.ttf` to the project root
   (free from Google Fonts) - `fontspec` will pick it up.

## Matching the official RUPP format exactly

The preamble is already configured to the official RUPP report identity: **A4**, margins
**left 1.25in, top/bottom/right 1in**, **1.5 line spacing**, **6pt** paragraph spacing,
**12pt body**, headings in **bold** (chapter 14pt, sections 12pt, numbered `N.M`), and **12pt
captions**. It compiles out of the box with the always-available substitute fonts
(**TeX Gyre Termes** for Latin, **Noto Sans Khmer** for Khmer).

To reproduce the *exact* official fonts (Times New Roman + the Khmer OS family), upload these
TTFs to the Overleaf project and switch the commented lines in `main.tex` (the alternatives sit
directly under the active ones):
- `Times New Roman` -> uncomment `\setmainfont{Times New Roman}`;
- `Khmer OS Siemreap` (body Khmer) -> uncomment the matching `\newfontfamily\khmerfont` line;
- `Khmer OS Muol Light` (title-page Khmer) -> uncomment the matching `\khmertitlefont` line.
The Khmer OS family is free (Cambodian government / KhmerOS project). If a font is not uploaded,
keep the default line or XeLaTeX will stop with a "font not found" error.

Tables use clean horizontal rules (`booktabs`). If your committee requires full-grid
("Table Grid") tables, that is a one-package change - ask and it can be swapped.
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
headline QWK 0.795 / 0.845 / 0.820 / 0.843 (comparable, 0.05 band); LLM deployment 66% exact, 83% within
±1; LOO occlusion faithful for all four pillars (gap +0.096 / +0.257 / +0.135 / +0.047); question leakage 0.76 → 0.35.
