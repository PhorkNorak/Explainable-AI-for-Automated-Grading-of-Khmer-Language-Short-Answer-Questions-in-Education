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
early: an explanation is only useful if it is **faithful** (reflects what the model actually
used); a *plausible but unfaithful* highlight gives false confidence and is arguably worse than
nothing. Say: "so I treat XAI as a first-class axis, I measure faithfulness, not just show
heatmaps."

## Problem & Research Questions

### 2.1 Problem statement
Three hard properties: (1) no benchmark to compare to; (2) **heterogeneous max scores**, "8"
means different things per question; (3) **small, single-grader** corpus. And the conceptual
point: separate *research quality* (QWK) from *deployment quality* (exact points), most
papers conflate them.

### 2.2 Research questions
Read the **six** RQs. Emphasize RQ2 (data vs algorithm), RQ3 (research vs deployment), and the
new **RQ5 (explainability, the central one for the title)**: across families, which gives
*faithful* explanations, are they *plausible* (do they overlap the teacher's reference), and is there an
accuracy↔explainability trade-off? Tell them each result slide maps back to an RQ.

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

### 3.4 XAI & faithfulness
Two key ideas: (1) **LOO occlusion** (Leave-One-Out, drop a word, measure the score change) is the
model-agnostic, perturbation-based attribution method — it works for SVR, RNN, Transformer, and LLM
alike, with no gradient access required; (2) **ERASER** comprehensiveness/sufficiency is how we
*measure* faithfulness — a positive gap vs random removal means the explanation is doing real work.
End with the Khmer gap: none of this has been done for Khmer grading, we do it across all four families.

## Methodology

### 4.1 Dataset
Hit the real stats: **1,184 answers, 2 schools, 8 classes, 203 students, 41 questions**, four
subjects, max scores 5–20. Ordinal label = round(4·score/max). Point at the **42% class-4
skew**, a majority predictor already gets 0.42 accuracy, so accuracy alone is misleading;
that's *why* QWK.

### 4.2 Curation (RQ2)
Two noise sources found by error inspection: the **10C-Biology** subset (inconsistent grading)
→ 909, and **14 score=0 outliers** → 895. Key design choice: run **all three variants** so we
can attribute gains to data vs algorithm. This is what makes RQ2 answerable.

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

### 4.6 Explainability methods (RQ5)
The XAI methodology slide. Three points: (1) **LOO occlusion is the sole method** — same operation
for all four pillars, no per-family adaptation needed; (2) every family's champion is explained on
the **same 895 answers** at **Khmer word** level → directly comparable; (3) evaluated on two axes:
**faithfulness** (ERASER comprehensiveness ↑ / sufficiency ↓, vs random-removal baseline) and
**plausibility** (reference-overlap proxy). Jacovi & Goldberg's faithfulness-vs-plausibility
distinction. Classical+RNN run here; encoder+LLM on HPC.

## Results

### 5.1 Per-pillar champions
The money slide. **The four pillars are comparable on QWK**, classical 0.795, RNN 0.845,
encoder 0.820, LLM 0.842, all **uncalibrated** and within a narrow 0.05 band. Say it plainly:
no single research winner. The **LLM is the clear winner, and only on
deployment**, Qwen wins all four deployment metrics (exact 0.672, within±1 0.788, MAE 0.98 pt).
Land the line: **no single pillar dominates.** (If asked why uncalibrated: see 5.3, calibration is
model-dependent, so we report one consistent uncalibrated number per pillar.)

### 5.2 Research vs deployment (RQ3)
Show the goal→winner table. The QWK lead is small (the pillars are comparable), so choose by the
*other* axes. Then the 909 head-to-head: encoder and LLM **tie on QWK**, but Qwen is far better on
what a teacher cares about (+22.7 pp exact match). Guidance: **cheapest + self-explaining →
classical; classroom point-accuracy → LLM.**

### 5.3 Data > algorithm (RQ2)
Same architecture (GTE) across 1184→909→895: **+0.027 QWK from cleaning alone**, bigger than
architecture swaps. Then the honest calibration point: it's a **separate, model-dependent
ablation**, it helps the classical model on test (0.795→0.847) but *hurts* the BiLSTM on test
even though validation rose, so we keep headline numbers uncalibrated. Punchline: on small data,
**data quality moves the needle as much as architecture, and cheap post-hoc levers are fragile.**

### 5.4 Max-score feature (RQ4)
Explain the intuition: "0.5" is ambiguous across questions with different max scores; give the
model the scale. Consistent deployment-metric gains (e.g. +4.5 pp exact on 895). Cheapest
neural win in the grid.

### 5.5 QWK in context, caveat slide
Be explicitly careful here. We compare on the **same unweighted Cohen κ** Alaoui uses (like-for-like
metric): ours **0.47 (classical) → 0.77 (LLM)** vs their **0.48 (BERT) / 0.60 (transformer)**. Our
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
Slow down here; this is the title's payoff. **LOO word attribution is reliably faithful for every
model**: removing its top words drops the score far more than random (classical AOPC-comp **+0.150**,
BiLSTM **+0.317**), and keeping only those words preserves the score (low sufficiency). LOO works
identically on the non-differentiable SVR and the BiLSTM — no gradient access required. Explanations
are also **plausible** (≈68–75% reference overlap). Be honest about the consistency caveat (LOO
judged by an occlusion-based metric has a lower bar, but the positive gap vs random confirms it's
doing real work). Encoder/LLM rows are HPC-pending; say so plainly.

### 5.9 The third axis
Tie it together: three dimensions, accuracy, deployment, explainability, but they behave
differently under scrutiny. On QWK the pillars are *tied*; the LLM wins deployment; LOO occlusion
is the unified, reliably faithful explanation for every family. No accuracy-vs-explainability
trade-off: the cheapest model and the most expensive one use the same explanation method.

### 5.10 The full standard metric set
The honesty slide. On **QWK all four pillars sit in a 0.05 band** → no single research winner
(classical and BiLSTM, on the same 895 test set, differ by only 0.05). On the **accuracy /
deployment metrics the LLM leads by clear margins.** Say plainly: "the LLM's deployment advantage is
the solid ranking; on QWK the pillars are comparable." Note all headline QWKs are **uncalibrated**
for cross-pillar consistency; calibration is the model-dependent ablation from 5.3, not in the headline.

### 5.11 Robustness to unseen questions, the big honest result
Slow down. Our split shares questions between train/test (41 questions only). Under a
**question-held-out** split, classical QWK **collapses 0.76 → 0.35** (Δ≈0.40, 5 seeds). So most of the
seen-question score was question-memorization. Frame it as a *contribution*: we **quantify** the
leakage that inflates small-pool ASAG benchmarks (ours and the literature's). Be honest about
variance (only ~6 test questions). Bottom line: "grading genuinely new questions is the open
problem; I report it rather than hide it." This slide is what moves the work toward journal grade.

### 6.3b Ethics
One breath: real student data (minors) → consent + IRB to confirm; only pseudonymous codes, no
names; teacher-assist not autonomous grader; single-grader/narrow-coverage fairness caveat. Full
statement in docs/ethics.md.

## Conclusion

### 6.1 Findings
Recap by RQ, **six** answers now: pillars comparable on QWK; LLM clearly best on deployment;
LOO occlusion reliably faithful for every family; and the leakage collapse on unseen questions.
Practical guidance: cheap classical matches on QWK and explains via LOO; Qwen for exact points;
unseen-question grading still open.

### 6.2 Contributions
Six: (1) the **first *explainable* Khmer ASAG benchmark**; (2) a **model-agnostic faithfulness
+ plausibility protocol** across all families (the XAI core); (3) a **quantified question-leakage
analysis**; (4) **honest multi-variant reporting** (train+test across 3 dataset variants); (5) max-score feature + a
calibration ablation; (6) honest train+test reporting. Lead with (1) and (2), they are what the
title promises.

### 6.3 Limitations
Say these *unprompted*, it pre-empts the examiners. Single grader (no ceiling), **question-
level leakage** (41 questions, per-row split), single seed, small corpus, no field study. Being
first to raise them is a strength.

### 6.4 Future work
2nd grader + question-held-out split (removes the two biggest caveats), Khmer pretraining,
more LLMs (incl. re-tuning Gemma), diverse ensemble, field study.

### 6.5 Summary
Headline results: **QWK 0.80–0.85 across four tied pillars** / 67% exact / 79% within ±1 /
**faithful occlusion vs unreliable attention** / **leakage 0.76→0.35**, plus the one-paragraph
"first *explainable* Khmer ASAG benchmark."

### 7 System, live prototype (demo / closer)
End on the tool. One line: "the research isn't just a benchmark, it runs." Walk the flow in a
sentence: **teacher pastes Question + Reference + Student answer, picks any of the four models, and
gets a score + the word-attribution highlighting + written feedback.** Stress the loop: **the same
faithfulness-checked word attribution from the study is what the teacher sees**, XAI is the feature.
Frame as **teacher-assist, human-in-the-loop**, not autonomous. If live: open [demo URL] and grade
one real answer (~30 s); if not, show the 1–2 screenshots. Then stop and invite questions.

> ⏱️ **15-min budget:** Motivation+RQs ~2 · Related Work ~1 · Dataset+Methodology ~3 ·
> Results (5.1–5.6) ~4 · Explainability (5.8–5.9) ~2 · Leakage (5.11) ~1 · System demo ~1 ·
> Conclusion ~1. Leave deep tables (5.7, 5.10) and Appendices A–E for Q&A.

---

## Likely Q&A, prepared answers

- **"Why does a 30-second classical model beat neural nets?"** Small data (≈600 train rows),
  char n-grams capture Khmer lexical overlap well, and TF-IDF+SVR can't overfit much. Neural
  models need more data or in-language pretraining to pull ahead, see the +0.02–0.04 QWK
  expected from Khmer continued pretraining.
- **"Is QWK ≈ 0.80–0.85 good?"** It's high *for this dataset and seen questions*. I report results
  across three dataset variants and contextualize against the literature human ceiling (~0.80–0.88). A 2nd grader for true IAA
  wasn't obtainable, so I'm explicit that this is agreement with one teacher, with that as partial
  mitigation, not a claim of beating a human ceiling.
- **"Isn't the question leakage a problem?"** Yes, and I **measured** it (slide 5.11): under a
  question-held-out split classical QWK drops 0.76 → 0.35 (5 seeds). So I report seen-question
  numbers *and* the unseen-question collapse. Quantifying that inflation is a contribution.
- **"How different are the pillars on QWK?"** Barely, they sit in a 0.05 band; the classical and
  BiLSTM champions on the same 895 test set differ by only 0.05. The clear gap is the LLM's
  deployment / accuracy lead.
- **"Why Qwen and not GPT-4?"** I needed an open model I could **fine-tune** (QLoRA) and run
  locally/reproducibly; literature shows fine-tuned specialists beat prompted general LLMs.
- **"Why drop the score=0 rows, isn't that cherry-picking?"** Only 14 rows (1.2%),
  effectively untrainable label noise; I report **all three** variants including the full set,
  so the effect of the choice is visible, not hidden.

### XAI-specific Q&A
- **"Isn't occlusion judged by an occlusion metric circular?"** LOO judged by an occlusion-based
  metric is self-consistent (a lower bar), but the positive gap vs random removal on both models
  (+0.096 classical, +0.274 BiLSTM) confirms it's doing real work. I report a random baseline and
  AOPC across five k-values, not a single cutoff.
- **"Why is the classical model both accurate and explainable?"** TF-IDF features are lexical
  and the SVR score moves directly with word presence, so occlusion attributions are
  meaningful and overlap the reference. It's a genuine advantage for accountability.
- **"What about the LLM's explanations?"** I evaluate LOO word attribution on the LLM with the
  same ERASER faithfulness on HPC, so a convincing output can't pass unchecked.
  (Result currently HPC-pending.)
- **"Why only word-level, not character-level?"** Khmer has no word spaces; word units (via
  khmernltk) are what a teacher reads and what aligns to the rubric.
