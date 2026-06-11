# CLAUDE.md — project brief for a new Claude session

This file orients a fresh Claude session on what this repo is and how to work in it.
Read it fully before acting. **The instructions in "Hard rules" override default behaviour.**

---

## What this project is

A **bachelor thesis** (RUPP — Royal University of Phnom Penh, Data Science & Engineering) titled
**"Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education."**

Author: **Phork Norak** · Supervisor: **Dr. Khim Chamroeun**.

It is an **Automated Short-Answer Grading (ASAG)** study on a real Khmer secondary-school corpus,
across four model families ("pillars"), evaluated on three axes that must be judged **together**:
1. **accuracy** (agreement with the human grader),
2. **deployment quality** (exact / within-one-point integer scores), and
3. **explainability** (faithful word-attribution highlighting).

The central message: *no single pillar dominates all three axes.* The four pillars are **comparable
on QWK**, a fine-tuned LLM wins the deployment metrics, word-attribution explanations are faithful
while attention is not automatically faithful, and scores collapse on unseen questions (leakage).

This is a finished/auditing-stage project: the code is run, the thesis/paper/slides are drafted, and a
live prototype is built. Most tasks now are **consistency, verification, and writing**, not new ML.

---

## Task formulation (how grading works)

- Predict a normalised score **ŷ ∈ [0,1]** with **MSE regression** (ordinal task treated as regression).
- `round(4·ŷ)` → a **5-class grade** (0–4) for class metrics.
- Raw points for a question = `round(ŷ · max_score)`.
- **Primary metric: Quadratic Weighted Kappa (QWK).** Also: Cohen κ (unweighted, for the Alaoui
  comparison), accuracy, macro-F1, exact / adjacent(±1) agreement.
- **No confidence intervals** anywhere (removed by request — "comparable" / "narrow band", never
  "statistically tied"). Do not reintroduce CIs or bootstrap/paired tests.

## The four model pillars

| Pillar | Model | Notes |
|---|---|---|
| **Classical** | TF-IDF (char_wb 2–4 gram) + **RBF-SVR** | CPU, ~30 s, strong baseline |
| **RNN** | char **BiLSTM + additive attention** | CPU; attention is a native explanation |
| **Transformer** | mBERT / XLM-R / **GTE** dual & cross encoders | GPU/HPC |
| **LLM** | **Qwen 3.5 4B** QLoRA (unsloth), id `Qwen/Qwen3.5-4B` | GPU/HPC; best deployment metrics |

## Dataset

- **1,184 answers · 41 questions · 203 students · 4 subjects**; max-scores ∈ {5,6,7,8,10,12,15,20}.
- Three cleaning/filter variants used as datasets: **`no10c_no0`** (default), `no10c`, `full`;
  split sizes referenced as **895 / 909 / 1184**. Only Classical and BiLSTM share a test set, so
  cross-pillar QWK is "indicative", not strict head-to-head.
- Preprocessing pipeline: NFC → strip invisibles → **KCC reorder** → strip punctuation →
  **khmernltk** word segmentation. ⚠️ KCC reorder produces model-internal text that is *unreadable*
  to humans (e.g. "ស៊ី"→"សី៊"); any **display / heatmap / feedback must use readable original tokens**
  (strip-invisibles + strip-punct + khmernltk, **no KCC reorder**).

---

## Canonical numbers (the consistency spine — must match across thesis ↔ paper ↔ slides ↔ code)

- **Headline QWK (uncalibrated):** Classical **0.795**, RNN **0.845**, encoder **0.820**, LLM **0.842**
  (a narrow ~0.05 band → "comparable", no significant QWK winner).
- **LLM deployment:** exact **0.672 (67%)**, within ±1 **0.788 (79%)**, MAE ≈ 0.98 pt (909 split).
- **Calibration** is a *fragile, model-dependent ablation* (not in the headline).
- **Cohen κ:** 0.47 / 0.62 / 0.62 / 0.77 vs Alaoui 0.48 (BERT) / 0.60 (transformer).
- **Leakage:** random split QWK **0.759** → question-held-out **0.354** (≈ −0.40, 5 seeds).
- **Explainability / faithfulness (n=135, seed 42):** see next section.

When asked for a number, source it from `results_stats/*.csv` and
`results_xai/no10c_no0/faithfulness_leaderboard.csv`. **Never invent or "round to a nicer" number.**

---

## Explainability = "word attribution / text highlighting" (the framing to use)

The XAI feature is **LOO (Leave-One-Out) word attribution (text highlighting)**: highlight exactly
which words drove the grade. Use this vocabulary in all docs. LOO is the **sole, unified XAI method**
across all four model families — no attention, saliency, rationale, SHAP, or LIME in the headline.

- **Engine = LOO occlusion:** drop a word, measure the score change. Model-agnostic, deterministic,
  works for *every* pillar including the non-differentiable classical SVR.
- **Faithfulness numbers** (ERASER comprehensiveness/sufficiency + AOPC vs a random-removal baseline;
  plausibility = reference-overlap proxy):
  - Classical · occlusion: **AOPC-comp +0.150**, gap +0.096, plaus 0.75 → **faithful**.
  - BiLSTM · occlusion: **AOPC-comp +0.317**, gap +0.274, plaus 0.68 → **faithful**.
  - Encoder / LLM: HPC-pending.
- **Do not introduce SHAP, LIME, attention, saliency, or rationale as XAI methods** in any
  deliverable. They were evaluated and demoted: SHAP is not reliably faithful for char-BiLSTM
  (gap −0.003); attention is configuration-dependent; LIME/IG were cross-check only (exp13,
  deleted). LOO is the final, verified choice.

---

## Repo layout (`final_kxs/`)

```
config.py, data.py, preprocess.py, evaluate.py   core pipeline (metrics, splits, Khmer cleaning)
models/        dual_encoder.py, bilstm, svr, llm heads
xai/
  explainers.py      occlusion (LOO) — the sole unified attribution method
  attributions.py    word_importance() — thin wrapper over occlusion_importance
  faithfulness.py    ERASER comprehensiveness/sufficiency + AOPC + random baseline
  plausibility.py    reference-overlap proxy
  render*.py         Khmer heatmap rendering (browser shaping; matplotlib can't shape Khmer)
experiments/
  exp01..exp08       the model grid (classical / rnn / transformer / ensemble / llm)
  exp09_xai.py       explainability study -> results_xai/<dataset>/faithfulness_leaderboard.csv
  exp10_significance.py   champion point metrics + leakage (no CIs)
  exp11_cleaning_ablation.py / exp12_hparam_tuning.py
results/champions/   best checkpoint per experiment (e.g. rnn_clean_ra_bilstm_895/best.pt)
results_stats/       champion_metrics.csv, split_compare.csv, cleaning_ablation.csv, hparam_tuning.csv
results_xai/         faithfulness_leaderboard.csv + heatmaps
prototype/           Gradio teacher-facing web app (app.py) + README + requirements + venv/
thesis/              XeLaTeX Overleaf project (main.tex, frontmatter/, chapters/ ch1–ch7, appendices/)
paper/               conference-style paper (main.tex, refs.bib, make_figures.py)
docs/                slides.md + script.md (deck + speaker notes), references.md
README.md            full run instructions incl. HPC tmux steps
```

The deliverables (**thesis / paper / slides+script / prototype**) must stay **consistent with each
other and the code** — terminology, naming, and every number.

---

## Hard rules (do not violate)

1. **Never `git commit` or `git push`.** The user manages all commits. (Branch is `experiments`.)
2. **No invented numbers / no hallucination.** Every figure must trace to a verified result file.
   If a value isn't computed yet, mark it `[pending]` / `[HPC-pending]`, don't fabricate it.
3. **AI feedback must use an OPEN-SOURCE LLM** (Ollama/vLLM/llama.cpp/LM Studio/OpenRouter over an
   OpenAI-compatible endpoint), **never the Anthropic/Claude API**. It must fall back to rule-based
   feedback when no LLM server is reachable.
4. **Local machine is CPU-only with limited internet.** Classical + BiLSTM run locally; all
   GPU/network work (encoder, LLM, attribution cross-check) is **HPC-scripted**, not run locally.
5. **No "AI slop"**: no em-dashes (`—` / `---`), no CIs, no over-hedged "statistically tied" language.
6. **Khmer display correctness:** show readable original tokens (no KCC reorder) in heatmaps/feedback.
7. **LaTeX builds with XeLaTeX** (polyglossia + fontspec Noto Sans Khmer); never tell the user to
   comment out fontspec. `thesis/main.tex` carries `% !TEX program = xelatex`.

## Environment quirks worth knowing

- **GTE-multilingual-base ships a NaN-corrupted rotary cache** — must rebuild `cos_cached` in fp32
  (see `_patch_rope` in `models/dual_encoder.py`).
- Prototype lives in its own **venv** (`prototype/venv/`) to avoid numpy 2.x conflicts.
- Gradio 6: pass `theme=` to `.launch()`, not `gr.Blocks()`.
- SHAP is far slower than occlusion on a dense model (~16 s vs ~4 s per answer); the prototype uses
  occlusion for interactivity.

## Common commands

```bash
# Explainability study (CPU): classical + BiLSTM word-attribution faithfulness
python experiments/exp09_xai.py --families classical bilstm --dataset no10c_no0

# Champion metrics + leakage (CPU)
python experiments/exp10_significance.py

# Live prototype (CPU; Classical + RNN active, Transformer/LLM need GPU)
pip install -r prototype/requirements.txt
python prototype/app.py        # http://127.0.0.1:7860
```

## Still-open placeholders for the user

Programme name confirmation, month/date, committee member names, the Khmer abstract proofread, and
the deployed prototype demo URL (after pushing to a Hugging Face Space).
