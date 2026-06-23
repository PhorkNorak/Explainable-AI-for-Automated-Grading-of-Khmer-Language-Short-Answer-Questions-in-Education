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
3. **explainability** (plausible word-attribution highlighting).

The central message: *no single pillar dominates all three axes.* The four pillars are **comparable
on QWK**, a fine-tuned LLM (the KhmerGrader family) wins the deployment metrics, and **SHAP**
word-attribution explanations are plausible for every pillar (most plausible for the BiLSTM). Results
are reported as agreement with a single human grader (the honesty anchor).

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
| **RNN** | char **BiLSTM + additive attention** | CPU; attention is architecture only, NOT used as explanation (SHAP is) |
| **Transformer** | mBERT / XLM-R / **GTE** dual & cross encoders | GPU/HPC |
| **LLM** | **KhmerGrader family** (QLoRA/unsloth): Qwen 3.5 4B (champion, id `Qwen/Qwen3.5-4B`), Gemma 4 E4B, SEA-LION v4.5 E2B | GPU/HPC; best deployment metrics; each compared vs its zero-shot base |

## Dataset

- **1,184 answers · 41 questions · 203 students · 4 subjects**; max-scores ∈ {5,6,7,8,10,12,15,20}.
- Two cleaning/filter variants used as datasets: **`no10c`** (default, 909), `full` (1184). Grade-0 is
  **kept** (full 5-class task; the old `no10c_no0`/895 drop-grade-0 variant was removed). Classical,
  BiLSTM, and LLM share the no10c test set; the encoder is on full, so cross-pillar QWK is
  "indicative", not strict head-to-head.
- Preprocessing pipeline (**no KCC**): `clean` = strip invisibles → NFC → strip punctuation;
  `segment` = clean → **khmernltk** word segmentation. The custom KCC reorder was removed (had an
  independent-vowel bug; matches the senior's recipe) — do not reintroduce it. Heatmaps / feedback use
  the readable segmented tokens.

---

## Canonical numbers (the consistency spine — must match across thesis ↔ paper ↔ slides ↔ code)

- **Headline QWK (uncalibrated):** Classical **0.795**, RNN **0.845**, encoder **0.820**, LLM **0.843**
  (a narrow ~0.05 band → "comparable", no significant QWK winner).
- **Per-pillar P / R / macro-F1** (uncalibrated, from `champion_metrics.csv`): Classical 0.478 / 0.421 / 0.418,
  RNN 0.540 / 0.523 / 0.529, encoder 0.548 / 0.550 / 0.541, LLM 0.790 / 0.784 / 0.780.
- **LLM deployment (Qwen-KhmerGrader-4B, 909 split):** exact **0.657 (66%)**, within ±1 **0.832 (83%)**,
  MAE ≈ 0.93 pt. Still wins every deployment metric vs the other three pillars.
- **KhmerGrader family (fine-tuned, published):** Qwen-KhmerGrader-4B QWK **0.843** (champion),
  SEA-LION-KhmerGrader-E2B **0.802** (best exact **0.693**, at 2.3B), Gemma-KhmerGrader-4B **0.763**.
  Base **zero-shot** QWK 0.500 / 0.541 / [pending] (Gemma base switched to the instruct `-it` variant,
  so all three bases are now instruction-tuned; its old −0.082/+0.85 was the pretrained base). Licences
  Apache-2.0 / MIT; lineage in `docs/model_cards.md`.
- **Calibration** is a *fragile, model-dependent ablation* (not in the headline).
- **Cohen κ:** 0.47 / 0.62 / 0.62 / **0.71** vs Alaoui 0.48 (BERT) / 0.60 (transformer).
- **SHAP plausibility:** Classical **0.66**, BiLSTM **0.59** (encoder + LLM HPC-pending). See next section.

⚠️ These headline numbers are **pre-re-run placeholders** (produced with the old KCC cleaning). After
the no-KCC grid re-run (`bash run_pipeline.sh`), refresh every number from the regenerated CSVs.
When asked for a number, source it from `results_stats/*.csv` and
`results_xai/no10c/`. **Never invent or "round to a nicer" number.**

---

## Explainability = SHAP word attribution / text highlighting (the framing to use)

The XAI feature is **SHAP word attribution (text highlighting)**: highlight which words drove the
grade, following the explainable-scoring approach of Kumar & Boulanger (2020). Use this vocabulary in
all docs. SHAP is the **sole, unified XAI method** across all four model families — no attention,
saliency, rationale, LOO/ERASER faithfulness, or LIME in the headline.

- **Engine = SHAP:** distribute the predicted score among the answer words by their Shapley values
  (occlusion / LOO is kept only as a fast special case in the code). Model-agnostic, works for *every*
  pillar including the non-differentiable classical SVR.
- **Plausibility** is the only XAI metric = fraction of the top-20% SHAP words that appear in the
  reference answer. All pillars anchored on the same `no10c` split:
  - Classical: plausibility **0.66**.
  - BiLSTM: plausibility **0.59** (most plausible of the small models).
  - Encoder, LLM: **HPC-pending** (the LLM runs SHAP under a **capped** eval budget — each eval is a
    full generation; `exp09 --families llm --shap-max-evals 50`).
- **Do not reintroduce** ERASER faithfulness (comprehensiveness/sufficiency/AOPC), LOO as the headline
  method, attention, saliency, LIME, or a leakage / question-held-out analysis in any deliverable.
  SHAP + plausibility is the final, verified choice; the leakage analysis was **dropped entirely**
  (RQ4 removed → 3 RQs; see the project memory).

---

## Repo layout (`final_kxs/`)

```
config.py, data.py, preprocess.py, evaluate.py   core pipeline (metrics, splits, Khmer cleaning)
models/        dual_encoder.py, bilstm, svr, llm heads
xai/
  explainers.py      SHAP word attribution (+ occlusion as a fast special case)
  attributions.py    word_importance() — thin wrapper over occlusion_importance
  plausibility.py    reference-overlap proxy (the only XAI metric)
  render*.py         Khmer heatmap rendering (browser shaping; matplotlib can't shape Khmer)
experiments/
  exp01..exp08       the model grid (classical / rnn / transformer / ensemble / llm)
  exp09_xai.py       SHAP explainability study -> results_xai/<dataset>/ (plausibility + global words)
  exp10_significance.py   champion point metrics (no leakage, no CIs)
  exp11_cleaning_ablation.py / exp12_hparam_tuning.py
results/champions/   best checkpoint per experiment (e.g. rnn_clean_ra_bilstm_909/best.pt)
results_stats/       champion_metrics.csv, cleaning_ablation.csv, hparam_tuning.csv
results_xai/         plausibility leaderboard + SHAP heatmaps
prototype/           Gradio teacher-facing web app (app.py) + README + requirements + venv/
thesis/              XeLaTeX Overleaf project (main.tex, frontmatter/, chapters/ ch1–ch7, appendices/)
paper/               conference-style paper (main.tex, refs.bib, make_figures.py)
docs/                slides.md + slide_final.md + script.md (decks + speaker notes), references.md
run_pipeline.sh      one-shot end-to-end run (tmux); see README
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
6. **No KCC + Khmer display correctness:** preprocessing has no KCC reorder; show readable segmented
   tokens in heatmaps/feedback. Do not reintroduce KCC, ERASER faithfulness, or a leakage analysis.
7. **LaTeX builds with XeLaTeX** (polyglossia + fontspec Noto Sans Khmer); never tell the user to
   comment out fontspec. `thesis/main.tex` carries `% !TEX program = xelatex`.

## Environment quirks worth knowing

- **GTE-multilingual-base ships a NaN-corrupted rotary cache** — must rebuild `cos_cached` in fp32
  (see `_patch_rope` in `models/dual_encoder.py`).
- Prototype lives in its own **venv** (`prototype/venv/`) to avoid numpy 2.x conflicts.
- Gradio 6: pass `theme=` to `.launch()`, not `gr.Blocks()`.
- SHAP is slower than occlusion (~16 s vs ~4 s per answer on a dense model). The prototype uses SHAP
  to match the thesis; the LLM pillar runs SHAP under a capped eval budget (or on-request in the
  prototype) because each SHAP eval is a full generation. The prototype's written feedback uses an
  open-source LLM (`FEEDBACK_LLM_*`) with a
  rule-based fallback.

## Common commands

```bash
# Full pipeline end-to-end in one go (HPC tmux): trains the grid, runs SHAP + ablations, makes figures
bash run_pipeline.sh

# SHAP explainability study (CPU): classical + BiLSTM plausibility
python experiments/exp09_xai.py --families classical bilstm --dataset no10c

# Champion point metrics (CPU)
python experiments/exp10_significance.py

# Live prototype (CPU; Classical + RNN active, Transformer/LLM need GPU)
pip install -r prototype/requirements.txt
python prototype/app.py        # http://127.0.0.1:7860
```

## Still-open placeholders for the user

Programme name confirmation, month/date, committee member names, the Khmer abstract proofread, and
the deployed prototype demo URL (after pushing to a Hugging Face Space).

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
