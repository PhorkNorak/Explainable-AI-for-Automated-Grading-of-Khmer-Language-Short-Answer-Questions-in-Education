# Project brief for documentation and research work

## Final-report reference

This is Phork Norak's undergraduate report, "Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education", supervised by Dr. Khim Chamroeun at RUPP. The designated final PDF is `docs/reports/Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education.pdf`, dated June 2026. The cover names the Bachelor of Engineering in Data Science and Engineering; the supervision statement names the Honor Program of Data Science and Engineering.

The public layout is src/khmer_asag, experiments, prototype, scripts, tests and docs. Private files live under ignored local/. The approved final PDF is the 67-page easier-language version in `docs/reports/`; see `docs/reports/README.md` for recovery and editing history. All editable manuscript sources and their supporting assets remain local and must be excluded from Git, including `local/manuscripts/thesis/`, `local/manuscripts/final-report/` and all other folders under `local/manuscripts/`. Do not force-add LaTeX or bibliography files. The sources were recovered from backups and the original report, not the exact original Overleaf export. Preserve earlier PDFs and local working projects.

Read `docs/guides/report_evidence.md` for report-to-result differences. The LaTeX under `local/manuscripts/thesis/` is a separate historical audit revision. Do not overwrite earlier versions or rewrite audit history to make them agree.

## Findings and terminology

Use the exact research questions from slide 6 of the final presentation: RQ1: Which model family gives the best agreement with the human teacher? RQ2: Which model is most useful for classroom deployment based on exact mark and within-one-point agreement? RQ3: Can SHAP provide understandable word-level explanations across all model families? Results measure agreement with a single grader; deployment uses raw-mark metrics and explanation assessment uses SHAP reference-overlap plausibility. The four pillars are TF-IDF/RBF-SVR, character BiLSTM with additive attention, multilingual Transformer encoders, and QLoRA LLMs.

Use **KhmerGrader** and the model names in the designated final report. Later experiments are outside the final-report documentation scope and must not be presented as aliases or continuations of the report's models. Attention is architecture, not the explanation method.

Report Table 10 and `local/results/results_stats/champion_metrics.csv` give QWK **0.802 / 0.789 / 0.777 / 0.850**, in pillar order. The LLM is the highest reported QWK performer. Report Tables 10 and 13 give Qwen exact **0.67** and within-one-point **0.80** (report precision). These are report findings, not a newly reproduced champion selection: the historical audit could not reconcile them with a coherent saved champion set. Never replace evidence files to match prose.

Report Table 12 gives fine-tuned QWK Qwen **0.850**, Gemma **0.844**, SEA-LION **0.713**. Qwen zero-shot QWK is **0.529**. Do not carry over the older PDF's family exact-match table or pending baseline entries: the latest Table 12 contains fine-tuned QWK only. Frontier results and parsed sample counts are recorded in `docs/guides/report_evidence.md`; the report dates that snapshot to 2026-06-24, with no fresh API verification implied.

SHAP word attribution (text highlighting) is the sole headline XAI method. Plausibility is the fraction of the top 20% SHAP words found in the reference answer. Report Table 14 states **0.63 / 0.60 / 0.68 / 0.61**, matching the saved summary after rounding, on **66 / 66 / 18 / 18** answers. Encoder is highest in this summary, but unequal samples limit ranking. The latest PDF has no pending-result labels. Historical estimator/checkpoint provenance concerns remain separate from this report-to-summary agreement. Do not reintroduce attention, saliency, LIME, ERASER faithfulness or leakage/question-held-out analysis as deliverables.

## Data and formulation

The corpus has 1,184 answers, 41 questions, 203 students, four subjects and two schools. `full` has 1,184 answers; `no10c` has 909. Both retain grade 0. Classical, RNN and LLM headline rows use the 137-answer no10c test set; encoder uses the 178-answer full test set. Comparisons involving the encoder are indicative, not strict head-to-head.

Predict normalised scores in [0,1], map to five-class labels with `round(4*y)`, and to raw marks with `round(y*max_score)`, bounded to valid ranges. SVR uses its regression objective, neural regression heads use MSE, and LLMs supervise integer-score completions. Preprocessing strips invisibles, applies NFC and strips punctuation; segmentation uses khmernltk. No custom KCC reorder.

Public, pseudonymous corpus copies are hosted at https://huggingface.co/datasets/phorknorak/khmer-asag-benchmark and https://www.kaggle.com/datasets/phorknorak/khmer-asag-benchmark under CC BY-NC 4.0. Local original records, predictions and weights remain excluded from Git. The author's current local CSVs correct one score from 19/15 to 9/15; dataset version 1.0.1 documents this correction. Report metrics have not been rerun on corrected data and a label-stratified split can change. See `docs/guides/dataset.md` and preserve the historical audit in `local/archive/audit/`. Public availability does not independently establish consent or ethics approval.

## Repository navigation

Use the root README and `experiments/README.md` for the pipeline; `prototype/README.md` for the app; `docs/reports/README.md` for final PDF documents; `docs/guides/slide_final.md`, `docs/guides/slides.md`, `docs/guides/script.md` and `docs/guides/defense_qa.md` for presentations. No active conference-paper source has been confirmed. Preserve dated audit records and archived writing.

## Hard rules (do not violate)

1. **Never `git commit` or `git push`.** The user manages all commits. (Branch is `experiments`.)
2. **No invented numbers / no hallucination.** Every figure must trace to a verified result file.
   If a value isn't computed yet, mark it `[pending]` / `[HPC-pending]`, don't fabricate it.
3. **AI feedback must use an OPEN-SOURCE LLM** (Ollama/vLLM/llama.cpp/LM Studio/OpenRouter over an
   OpenAI-compatible endpoint), **never the Anthropic/Codex API**. It must fall back to rule-based
   feedback when no LLM server is reachable.
4. **Local machine is CPU-only with limited internet.** Classical + BiLSTM run locally; all
   GPU/network work (encoder, LLM, attribution cross-check) is **HPC-scripted**, not run locally.
5. **No "AI slop"**: no em-dash prose or decorative dash separators, no CIs, no over-hedged "statistically tied" language.
6. **No KCC + Khmer display correctness:** preprocessing has no KCC reorder; show readable segmented
   tokens in heatmaps/feedback. Do not reintroduce KCC, ERASER faithfulness, or a leakage analysis.
7. **LaTeX builds with XeLaTeX** (polyglossia + fontspec Noto Sans Khmer); never tell the user to
   comment out fontspec. `local/manuscripts/final-report/main.tex` carries `% !TEX program = xelatex`.

## Environment quirks worth knowing

- **GTE-multilingual-base ships a NaN-corrupted rotary cache**; must rebuild `cos_cached` in fp32
  (see `_patch_rope` in `src/khmer_asag/models/dual.py`).
- Prototype lives in its own **venv** (`local/environments/prototype/`) to avoid numpy 2.x conflicts.
- Gradio 6: pass `theme=` to `.launch()`, not `gr.Blocks()`.
- SHAP adds model-evaluation cost; latency depends on the model and evaluation budget. The prototype uses SHAP
  to match the undergraduate report; the LLM pillar runs SHAP under a capped eval budget (or on-request in the
  prototype) because each SHAP eval is a full generation. The prototype's written feedback uses an
  open-source LLM (`FEEDBACK_LLM_*`) with a
  rule-based fallback.


## Commands and open items

Run the prototype from the root with `python prototype/app.py` after installing its requirements in an isolated environment. For new research runs follow `docs/guides/hpc.md`; preserve original results. The point-metric script requires `--output-dir`; consult `experiments/README.md` rather than old invocation examples.

The final PDF supplies programme, date and committee names: Dr. Khim Chamroeun (Chairperson), Ms. Chea Daly, Mr. Ky Soklay, Mr. Chap Chanpiseth and Mr. Mao Makara. Signatures, consent records, Khmer proofreading, compatible result provenance and a verified public demo URL remain unverified.

## graphify

When `graphify-out/graph.json` exists, query it first for codebase questions with `graphify query "<question>"`; use `path` and `explain` for focused relationships. Use `graphify-out/wiki/index.md` for broad navigation if present. Read the full graph report only when scoped queries are insufficient. After code modifications, run `graphify update .` to keep the graph current. No graph is currently present in this checkout.
