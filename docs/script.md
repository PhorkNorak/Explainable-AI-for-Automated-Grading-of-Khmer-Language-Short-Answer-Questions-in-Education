# Speaker notes, Khmer ASAG thesis defense

Concise talking points, one block per slide of `slides.md`. Aim ~45–60 s per content slide;
the core deck runs ~15 min (deep tables and appendices are for Q&A). Numbers in **bold** are the ones worth
pausing on.

---

### Title
Greet, state the title. One sentence: "I built the first reproducible benchmark for grading
Khmer short answers, comparing four families of models, and the honest finding is that there
is *no free lunch*: the best model depends on what you measure."

### Outline
Walk the seven sections in one breath. Tell them the punchline up front (the one-line thesis) so
they listen for the trade-off, not a horse race.

---

## Background & Motivation

### 1.1 What is ASAG
Define ASAG plainly: score a free-text answer against a reference, on a number scale.
Contrast with essay scoring (content vs style). Motivate: hand-grading free text is slow and
inconsistent; good ASAG scales feedback. The catch: progress has been on English/Arabic, not
low-resource languages.

### 1.2 Khmer is under-served
**17M+ speakers, still low-resource.** No whitespace between words → tokenization is itself a
research problem. Multilingual encoders barely saw Khmer. Crucially: **no prior published
Khmer ASAG benchmark**, that's the opening. State the opportunity sentence.

### 1.3 Why explainability matters
This is the slide that earns the title. A grade is high-stakes, students/teachers need to know
**why**. Teachers only adopt a grader they can **audit**. And the key technical point to plant
early: the explanation should point at the **content** that justified the grade, so a teacher can
check it against the reference answer. Say: "so I treat XAI as a first-class axis, with one unified
SHAP explanation across all four model families."

## Problem & Research Questions

### 2.1 Problem statement
Three hard properties: (1) no benchmark to compare to; (2) **heterogeneous max scores**, "8"
means different things per question; (3) **small, single-grader** corpus. And the conceptual
point: separate *research quality* (QWK) from *deployment quality* (exact points), most
papers conflate them.

### 2.2 Research questions
Read the RQs. Emphasize RQ2 (data vs algorithm), RQ3 (research vs deployment), and the
**explainability RQ (the central one for the title)**: across families, does one unified **SHAP**
method give *plausible* explanations (do the highlighted words overlap the teacher's reference),
and is there an accuracy↔explainability trade-off? Tell them each result slide maps back to an RQ.

## Literature Review

### 3.1 Classical → transformers
Three eras in 30 seconds: Mohler & Mihalcea (similarity features) → SemEval-2013 Task 7
(shared datasets) → transformer/BERT survey (Sung et al. 2022). The throughline:
answer-vs-reference comparison, increasingly powerful encoders.

### 3.2 LLM era + metric
LLMs (GPT-4 prompting, QLoRA fine-tuning, hybrids) are competitive but **fine-tuned
specialists still lead**, that motivates our v08 fine-tune rather than zero-shot. Then justify
**QWK** as the field-standard ordinal metric, and say we *also* report deployment metrics.

### 3.3 Anchor + gap
Alaoui (Arabic) is the closest study, **1,276 answers, 3 classes, transformer 95.67/77.22**.
Flag now that their kappa is **unweighted**, you'll use that honestly later. Then the Khmer
landscape (tokenization, POS, pretrained models, scene text), **but no ASAG.** State the gap:
no benchmark, no multi-family comparison, **no explainability study**, little honest gap
reporting. "This thesis fills all of them."

### 3.4 XAI & explainability
Two key ideas: (1) **SHAP** word attribution (distribute the score among the answer words by their
Shapley values) is the model-agnostic method — it works for SVR, RNN, Transformer, and LLM alike,
with no gradient access required; (2) we judge an explanation by its **plausibility** — do the
highlighted words overlap the reference content a teacher would check? Following Kumar & Boulanger
(2020). End with the Khmer gap: none of this has been done for Khmer grading, we do it across all four families.

## Methodology

### 4.1 Dataset
Hit the real stats: **1,184 answers, 2 schools, 8 classes, 203 students, 41 questions**, four
subjects, max scores 5–20. Ordinal label = round(4·score/max). Point at the **42% class-4
skew**, a majority predictor already gets 0.42 accuracy, so accuracy alone is misleading;
that's *why* QWK.

### 4.2 Curation (RQ2)
One noise source found by error inspection: the **10C-Biology** subset (inconsistent grading)
→ 909 (`no10c`). Key design choice: run **both variants** (`full` 1184, `no10c` 909, full 5-class
kept) so we can attribute gains to data vs algorithm. This is what makes RQ2 answerable.

### 4.3 Shared pipeline
Walk the diagram top-to-bottom: split (seed 42) → 3 preprocess modes → 2 input formats →
model → normalized-score MSE → round → 8 metrics. The discipline point: **one recipe**, so
differences come from the axis under test, not from per-model tuning.

### 4.4 Four pillars
Name them: classical (TF-IDF+SVR), RNN (BiLSTM), encoder transformers (Dual/Cross ×
mBERT/XLM-R/GTE), LLM (Qwen 3.5 4B QLoRA). Mention GTE's RoPE-NaN fp32 fix only if asked, 
it's an engineering footnote. Restate metrics: QWK primary + deployment metrics.

### 4.5 v01–v08 grid
Don't read every row. Frame it as: classical baseline (v01) + cheap levers (v02 calibration,
v03/v03b max-score) + the model families (v05 RNN, v06 encoders) + ensemble (v07) + LLM (v08),
all × 3 datasets, all reported train+test. "Hundreds of cells, one leaderboard, fully
reproducible."

### 4.6 Explainability method (RQ5)
The XAI methodology slide. Three points: (1) **SHAP is the sole method** — same operation for all
four pillars, no per-family adaptation needed; (2) every family's champion is explained on the
**same 909 answers** at **Khmer word** level → directly comparable; (3) judged by **plausibility**
(fraction of the top SHAP words that appear in the reference answer), plus a **global** ranking of
the words each model relies on. Classical+RNN run here; encoder+LLM on HPC.

## Results

### 5.1 Per-pillar champions
The money slide. **The four pillars are comparable on QWK**, classical 0.795, RNN 0.845,
encoder 0.820, LLM 0.843, all **uncalibrated** and within a narrow 0.05 band. Say it plainly:
no single research winner. The **LLM is the clear winner, and only on
deployment**, Qwen wins all four deployment metrics (exact 0.657, within±1 0.832, MAE 0.93 pt).
Land the line: **no single pillar dominates.** (If asked why uncalibrated: see 5.3, calibration is
model-dependent, so we report one consistent uncalibrated number per pillar.)

### 5.2 Research vs deployment (RQ3)
Show the goal→winner table. The QWK lead is small (the pillars are comparable), so choose by the
*other* axes. Then the 909 head-to-head: encoder and LLM **tie on QWK**, but Qwen is far better on
what a teacher cares about (+22.7 pp exact match). Guidance: **cheapest + self-explaining →
classical; classroom point-accuracy → LLM.**

### 5.3 Data > algorithm (RQ2)
Same architecture (GTE) across 1184→909: **+0.022 QWK from removing 10C alone**, bigger than
architecture swaps. Then the honest calibration point: it's a **separate, model-dependent
ablation**, it helps the classical model on test (0.795→0.847) but *hurts* the BiLSTM on test
even though validation rose, so we keep headline numbers uncalibrated. Punchline: on small data,
**data quality moves the needle as much as architecture, and cheap post-hoc levers are fragile.**

### 5.4 Max-score feature (RQ4)
Explain the intuition: "0.5" is ambiguous across questions with different max scores; give the
model the scale. Consistent deployment-metric gains (e.g. +4.5 pp exact on 909). Cheapest
neural win in the grid.

### 5.5 QWK in context, caveat slide
Be explicitly careful here. We compare on the **same unweighted Cohen κ** Alaoui uses (like-for-like
metric): ours **0.47 (classical) → 0.71 (LLM)** vs their **0.48 (BERT) / 0.60 (transformer)**. Our
encoder/RNN (~0.62) match their transformer; our LLM is higher. Then the caveat out loud: **still
different datasets and class counts (5-class Khmer vs 3-class Arabic), so this is context, not a
leaderboard.** Our primary QWK (quadratic-weighted) is reported separately. The fully fair
comparison is the train/test gap (next slide). This honesty earns credibility.

### 5.6 Generalization gap, apples-to-apples
Both report accuracy, so compare the **train→test gap**. Our deep cells: **+1.4 pp (GTE), +2.4 pp
(Qwen)**; even the simple classical (+15.6 pp) stays under Alaoui's best
transformer: **+18.5 pp**, and their own paper says it overfits faster. Conclusion: **no
hidden overfitting** under our headline numbers.

### 5.7 Per-question difficulty
Low-max questions (5–7) are near-solved; **high-max (15–20)** are the bottleneck (more
partial-credit granularity). This scopes future work precisely.

### 5.8 Explainability results, the star finding (RQ5)
Slow down here; this is the title's payoff. **SHAP gives a plausible explanation for every pillar**:
the highlighted words overlap the reference content (plausibility **0.66** classical, **0.59** BiLSTM;
encoder and LLM are HPC-pending). The **global** top SHAP words are subject-content terms, confirming
the models grade on content rather than surface form. SHAP works identically on the non-differentiable
SVR, the BiLSTM, the GTE encoder, and the fine-tuned LLM — no gradient access required. So one unified,
model-agnostic method explains all four families; the LLM runs SHAP under a capped evaluation budget
(each evaluation is a full generation).

### 5.9 The third axis
Tie it together: three dimensions, accuracy, deployment, explainability, but they behave
differently under scrutiny. On QWK the pillars are *tied*; the LLM wins deployment; **SHAP**
is the unified, plausible explanation for every family. No accuracy-vs-explainability
trade-off: the cheapest model and the most expensive one use the same explanation method.

### 5.10 The full standard metric set
The honesty slide. On **QWK all four pillars sit in a 0.05 band** → no single research winner
(classical and BiLSTM, on the same 909 test set, differ by only 0.05). On the **accuracy /
deployment metrics the LLM leads by clear margins.** Say plainly: "the LLM's deployment advantage is
the solid ranking; on QWK the pillars are comparable." Note all headline QWKs are **uncalibrated**
for cross-pillar consistency; calibration is the model-dependent ablation from 5.3, not in the headline.

### 6.3b Ethics
One breath: real student data (minors) → consent + IRB to confirm; only pseudonymous codes, no
names; teacher-assist not autonomous grader; single-grader/narrow-coverage fairness caveat. Full
statement in docs/ethics.md.

## Conclusion

### 6.1 Findings
Recap by RQ: pillars comparable on QWK; LLM clearly best on deployment; **SHAP** gives a plausible
explanation for every family. Practical guidance: cheap classical matches on QWK and explains via
SHAP; Qwen for exact points.

### 6.2 Contributions
(1) the **first *explainable* Khmer ASAG benchmark**; (2) a **model-agnostic SHAP word-attribution
+ plausibility protocol** across all four families (the XAI core); (3) **honest multi-variant
reporting** (train+test across 3 dataset variants); (4) max-score feature + a calibration ablation.
Lead with (1) and (2), they are what the title promises.

### 6.3 Limitations
Say these *unprompted*, it pre-empts the examiners. Single grader (no ceiling), small corpus and
only 41 questions, single seed, narrow subject coverage, no field study. Being first to raise them
is a strength.

### 6.4 Future work
2nd grader (a measured human ceiling), larger multi-school corpus, Khmer pretraining,
more LLMs (incl. re-tuning Gemma), diverse ensemble, field study.

### 6.5 Summary
Headline results: **QWK 0.80–0.85 across four tied pillars** / 66% exact / 83% within ±1 /
**SHAP plausible across all four pillars**, plus the one-paragraph
"first *explainable* Khmer ASAG benchmark."

### 7 System, live prototype (demo / closer)
End on the tool. One line: "the research isn't just a benchmark, it runs." Walk the flow in a
sentence: **teacher pastes Question + Reference + Student answer, picks any of the four models, and
gets a score + the word-attribution highlighting + written feedback.** Stress the loop: **the same
SHAP word attribution from the study is what the teacher sees**, XAI is the feature.
Frame as **teacher-assist, human-in-the-loop**, not autonomous. If live: open [demo URL] and grade
one real answer (~30 s); if not, show the 1–2 screenshots. Then stop and invite questions.

> ⏱️ **15-min budget:** Motivation+RQs ~2 · Related Work ~1 · Dataset+Methodology ~3 ·
> Results (5.1–5.6) ~4 · Explainability (5.8–5.9) ~2 · System demo ~1 ·
> Conclusion ~1. Leave deep tables (5.7, 5.10) and Appendices A–E for Q&A.

---

## Likely Q&A, prepared answers

- **"Why does a 30-second classical model beat neural nets?"** Small data (≈600 train rows),
  char n-grams capture Khmer lexical overlap well, and TF-IDF+SVR can't overfit much. Neural
  models need more data or in-language pretraining to pull ahead, see the +0.02–0.04 QWK
  expected from Khmer continued pretraining.
- **"Is QWK ≈ 0.80–0.85 good?"** It's high *for this dataset and seen questions*. I report results
  across two dataset variants and contextualize against the literature human ceiling (~0.80–0.88). A 2nd grader for true IAA
  wasn't obtainable, so I'm explicit that this is agreement with one teacher, with that as partial
  mitigation, not a claim of beating a human ceiling.
- **"How different are the pillars on QWK?"** Barely, they sit in a 0.05 band; the classical and
  BiLSTM champions on the same 909 test set differ by only 0.05. The clear gap is the LLM's
  deployment / accuracy lead.
- **"Why Qwen and not GPT-4?"** I needed an open model I could **fine-tune** (QLoRA) and run
  locally/reproducibly; literature shows fine-tuned specialists beat prompted general LLMs.
- **"Why two dataset variants, not more cleaning?"** I keep the full 5-class range (grade 0 included)
  and study only the one defensible cleaning step, removing the inconsistently-graded 10C subset. I
  report **both** the full set and `no10c`, so the effect of that choice is visible, not hidden.

### XAI-specific Q&A
- **"Why SHAP and not attention or LIME?"** SHAP is model-agnostic (works on the non-differentiable
  SVR), is the field standard for explainable scoring (Kumar & Boulanger 2020), and one method covers
  all four pillars. Attention is configuration-dependent and not a reliable explanation.
- **"Why is the classical model both accurate and explainable?"** TF-IDF features are lexical
  and the SVR score moves directly with word presence, so SHAP attributions are
  meaningful and overlap the reference. It's a genuine advantage for accountability.
- **"What about the LLM's explanations?"** I run the same SHAP attribution on the LLM; its
  plausibility cell is HPC-pending; it runs SHAP under a capped budget, since each SHAP evaluation is
  a full generation.
- **"Why only word-level, not character-level?"** Khmer has no word spaces; word units (via
  khmernltk) are what a teacher reads and what aligns to the rubric.
