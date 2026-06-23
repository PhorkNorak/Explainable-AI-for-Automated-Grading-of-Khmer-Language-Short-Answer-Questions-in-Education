---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Royal University of Phnom Penh · Faculty of Engineering'
footer: 'Phork Norak · Explainable AI for Khmer ASAG'
style: |
  section { font-size: 20px; padding: 40px 56px 60px; }
  h1 { color: #1a365d; border-bottom: 2px solid #2c5282; padding-bottom: 5px; margin-bottom: 12px; font-size: 30px; }
  h2 { color: #2c5282; font-size: 24px; margin-top: 8px; margin-bottom: 8px; }
  h3 { font-size: 20px; }
  table { font-size: 15px; margin: 4px auto; border-collapse: collapse; }
  code { font-size: 13px; }
  pre { font-size: 13px; line-height: 1.25; }
  th { background: #2c5282; color: white; padding: 4px 7px; }
  td { padding: 3px 7px; }
  p { margin: 4px 0; }
  ul, ol { margin: 4px 0; padding-left: 22px; }
  li { margin: 2px 0; }
  .small { font-size: 14px; }
  .red { color: #c53030; font-weight: bold; }
  .green { color: #2f855a; font-weight: bold; }
  .blue { color: #2c5282; font-weight: bold; }
  blockquote { border-left: 4px solid #2c5282; padding-left: 12px; color: #2d3748; margin: 6px 0; }
  header, footer { font-size: 11px; }
---

<!-- _paginate: false -->

# Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education

<br>

**Student Name:** Phork Norak
**Royal University of Phnom Penh**
**Faculty of Engineering**
**Advisor:** Dr. Khim Chamroeun
**Date:** [Date]
**Degree:** Bachelor of Data Science & Engineering

---

## CONTENT

1. **Introduction**
2. **Problem**
3. **Research Objectives**
4. **Literature Review**
5. **Methodology**
6. **Experiment Setup**
7. **Result**
8. **Limitations & Future Works**
9. **Conclusion**

- References
- Appendix

---

# I. Introduction

- **Automatic Short Answer Grading (ASAG)** scores a student's free-text answer against a
  reference answer on a numeric scale. Unlike essay scoring, short answers are judged on
  **content correctness**, not length or style.
- **Why it matters:** grading free-text by hand is slow and inconsistent. Reliable ASAG gives
  teachers faster, more consistent feedback and scales formative assessment.
- **Khmer** (ភាសាខ្មែរ), the official language of Cambodia (17M+ speakers), is still treated as
  **low-resource** in NLP. Modern ASAG progress (transformers, LLMs) has **not reached Khmer**.
- A grade is a **high-stakes decision**: students and teachers need to know **why** a score was
  given. A bare number is not enough for trust, audit, or appeal, so **explainability** is essential.

> This thesis builds the **first reproducible, explainable Khmer ASAG benchmark** and judges
> every model on three axes together: **accuracy, deployment quality, and explainability**.

---

# II. Problem

Grading Khmer short answers automatically is hard for several reasons:

- **No benchmark, no baselines.** Nothing published for Khmer ASAG to compare against.
- **Khmer is complex and low-resource.** No whitespace word boundaries, stacked syllables, and
  little Khmer in multilingual pretraining, so even tokenization is non-trivial.
- **Heterogeneous scoring scales.** Each question has its own max score ∈ {5,6,7,8,10,12,15,20},
  so a raw score of "8" means different things on different questions.
- **Small, single-grader corpus.** 1,184 answers graded by **one** teacher, limited data and no
  inter-rater ceiling.
- **A number is not trust.** A grade must come with an explanation that **points at the content**
  that justified it, so a teacher can check it against the reference.

> We must also separate two questions usually conflated in ASAG: **research quality** (ordinal
> agreement, QWK) versus **deployment quality** (does it predict the *exact* teacher score?).

---

# III. Research Objectives

In this study, we aim to:

- **Build** the first reproducible, explainable Khmer ASAG benchmark and **release** the corpus
  and fine-tuned models for future research.
- **Compare** four model families on a 1,184-sample Khmer corpus: **Classical ML, RNN, encoder
  Transformer, and a fine-tuned LLM** (RQ1).
- **Separate** research quality (QWK) from **deployment quality** (exact match, within ±1 point),
  and test whether the same model wins both (RQ3).
- **Measure** two cheap levers: encoding each question's **max score** as a feature and **post-hoc
  threshold calibration** (RQ4), and whether **data cleaning** beats algorithm choice (RQ2).
- **Explain** every grade across all four families with one unified, model-agnostic **SHAP**
  word-attribution method, and check its **plausibility** against the reference (RQ5, central).

---

# IV. Literature Review

## 1. ASAG: from similarity measures to the LLM era

- **Classical era.** Mohler, Bunescu & Mihalcea (2011) grade short answers with
  **semantic-similarity** and dependency features, establishing reference-vs-answer comparison as
  the core idea. **SemEval-2013 Task 7** (Dzikovska et al.) standardized datasets and evaluation.
- **Deep-learning era.** A 2022 survey (Sung et al.) traces the shift word embeddings → sequential
  models → **attention/transformers**; BERT-family encoders become the workhorses for low-data ASAG.
- **LLM era.** Recent work fine-tunes or prompts LLMs (GPT-4, QLoRA cross-prompt). Finding across
  studies: general LLMs are *competitive*, but **specialized fine-tuned models** still lead.
- **The standard metric.** **Quadratic Weighted Kappa (QWK)** is the de-facto standard for ordinal
  scoring: it rewards near-misses and penalizes large disagreements, unlike plain accuracy.

> We adopt **QWK as the primary metric** and additionally report classroom-oriented **deployment**
> metrics (exact-score match, within ±1 point).

---

# IV. Literature Review

## 2. Explainable AI and attribution

- **Attribution methods.** Model-agnostic perturbation (**LIME**, Ribeiro et al. 2016; occlusion)
  and **SHAP** (Lundberg & Lee 2017) attribute a prediction to input tokens.
- **Attention ≠ explanation.** Raw attention weights are *not* guaranteed reliable
  (Jain & Wallace 2019); the rebuttal (Wiegreffe & Pinter 2019) argues attention must be
  **evaluated**, not trusted or dismissed.
- **SHAP for scoring.** Kumar & Boulanger (2020) use **SHAP** to expose which features drive an
  essay score and show it has pedagogical value; SHAP is the standard ASAG attribution (Pinto 2025).
  We judge an explanation by its **plausibility** (Jacovi & Goldberg 2020).

> A **plausible** explanation highlights the rubric-relevant content a teacher would check against
> the reference, which is what turns an attribution into a basis for feedback.

---

# IV. Literature Review

## 3. Closest anchor (Arabic) and the Khmer gap

- **Arabic ASAG, Alaoui et al. (2024, IJECE).** Closest comparable study: self-collected Arabic
  dataset (**1,276 answers, 18 questions, 3 classes 0–2**, 6th-grade). Best model = a transformer at
  **95.67% train / 77.22% test accuracy**; the paper notes the transformer **overfits faster** with
  epochs. They report **unweighted** Cohen's κ = 0.60 (not QWK).
- **Khmer NLP landscape.** Pretrained Khmer models + POS/news data (Buoy et al. 2022), the
  Khmer-NLTK word segmenter (Hoang 2020), and scene text, **but no ASAG**.

> **Gap:** no Khmer ASAG benchmark, no multi-family comparison, **no explainability study**, and
> little honest train/test-gap reporting in low-resource ASAG. This thesis addresses all four.

---

# V. Methodology

## 1. Research Design

```
data.csv (1184 / 909 variants)
   ↓  stratified 70/15/15 split (seed = 42)  →  train / val / test
   ↓  Khmer-aware preprocessing  (NFC → strip invisibles → strip punct → khmernltk)
   ↓  four model pillars  (Classical · RNN · Transformer · LLM)
   ↓  predict normalized score  ŷ ∈ [0,1]   →   round(4·ŷ) → grade {0..4}
   ↓  evaluate accuracy + deployment metrics  (train + test)
   ↓  explain with SHAP word attribution  →  plausibility vs reference
```

- One **consistent recipe** for every cell, so differences come from the **axis under test**, not
  tuning noise. Trainable models use AdamW with **early-stop on validation QWK**, single seed (42).
- **Target:** `score / max_score ∈ [0,1]`, trained as **bounded MSE regression**.

> The pipeline is identical across all four pillars, which is what makes the comparison fair and
> the benchmark reproducible.

---

# V. Methodology

## 2. Data Collection & Preprocessing

| Property | Value |
|---|---|
| Graded answers | **1,184** (after dropping 1 incomplete row) |
| Schools / classes / students | 2 (AS, PT) · 8 classes · 203 students |
| Subjects | Biology (435), History (434), Geography (307), Earth Science (9) |
| Unique questions | **41** · Max-score per question ∈ {5,6,7,8,10,12,15,20} |
| Grader | One trained teacher (single-grader corpus) |

**Ordinal label** = `round(4 · StudentScore / MaxScore)`: classes 0–4 with counts
14 / 153 / 327 / 191 / **499** (a 42% full-credit majority class). **Two cleaning variants:**
`full` (1184) and `no10c` (909, drop the noisy **Class 10C Biology** subset); both keep the full
5-class range. **Khmer cleaning:** NFC → strip invisibles →
strip punctuation → optional khmernltk segmentation; digits kept as content.

> Running every experiment on **both variants** lets us attribute gains to **data** vs **algorithm**.

---

# V. Methodology

## 3. Models: Classical (TF-IDF + SVR)

- **Backbone:** TF-IDF over **character `char_wb` 2–4-grams** (robust to Khmer's lack of word
  boundaries), max 15,000 features.
- **Head:** **RBF Support Vector Regression** on the pair feature
  `[a; b; |a−b|; a⊙b; cos]` built from the answer and reference vectors.
- **Cost:** **CPU only, ~30 seconds.** No GPU, no pretraining.
- **Role:** a strong, cheap, **self-contained baseline**, the floor every deep model must beat.

> Despite its simplicity, the classical pillar matches the deep models on QWK, the surprise of the study.

---

# V. Methodology

## 3. Models: RNN (char BiLSTM + Attention)

- **Backbone:** a **character-level Bidirectional LSTM** with **additive attention**
  (hidden 128, 2 layers, embed 128, dropout 0.3).
- **Head:** a 4-way MLP → σ producing ŷ ∈ [0,1].
- **Trained from scratch** on CPU; no external pretraining.
- **Note on attention:** the attention layer is **architecture only**. It is **not** used as the
  explanation, because attention is not guaranteed faithful. Explanations use **SHAP** instead.

> The BiLSTM is the strongest pillar on QWK and, as we will see, gives the **most plausible** SHAP explanations.

---

# V. Methodology

## 3. Models: Transformer (encoder, BERT-family)

- **Backbone:** **encoder transformers** used as **dual** and **cross** encoders, over three
  multilingual backbones: **mBERT, XLM-R, and GTE-multilingual-base**.
- **Head:** a 4-way / [CLS] MLP → σ; bottom 6 layers frozen, fine-tuned at lr 2e-5.
- **Compute:** GPU / HPC.
- **Engineering note:** GTE ships a **NaN-corrupted RoPE cache** under fp16; we rebuild `cos_cached`
  in fp32 (`models/dual_encoder.py`). The **GTE dual encoder + max-score feature** is the champion cell.

> mBERT, XLM-R, GTE (and the LLM) are all Transformers; this pillar uses **encoders**, the next uses a **decoder**.

---

# V. Methodology

## 3. Models: LLM (Qwen-KhmerGrader, QLoRA)

- **The KhmerGrader family:** three open base models fine-tuned identically with **QLoRA/unsloth** to
  emit the integer score as text, each compared against its zero-shot base.
- **Champion: Qwen-KhmerGrader-4B** (QLoRA r=16, lr 2e-4, 10 epochs, completion-only loss; greedy
  decode → first integer → clip). Also Gemma-KhmerGrader-4B and SEA-LION-KhmerGrader-E2B (2.3B).
- **Compute:** GPU / HPC. Licences Apache-2.0 / MIT (lineage in `docs/model_cards.md`).
- **QLoRA lift over zero-shot:** **+0.34 / +0.26 / [pending]** QWK across the three bases.

> The fine-tuned LLM wins **every deployment metric**, but its explanations are the **least sharp** (§7).

---

# V. Methodology

## 4. Grading task & Explainability method (SHAP)

**Grading task.** Predict ŷ ∈ [0,1] by **bounded MSE regression**; `round(4·ŷ)` gives a 5-class
grade (0–4); raw points = `round(ŷ · max_score)`. Two cheap levers tested: **max-score as a feature**
and **post-hoc threshold calibration**.

**Explainability = SHAP word attribution (text highlighting).** Distribute the predicted score among
the answer words by their Shapley values. This **SHAP** attribution is the **single, unified method**
applied identically to all four pillars:

- Needs **no gradients or internal access**, so it works on the non-differentiable classical SVR.
- Model-agnostic; explains the *answer* at the **Khmer word** level.
- Heatmaps and feedback display **readable original tokens**.

> Following Kumar & Boulanger (2020): **SHAP is the unified, verified XAI choice** across all four pillars.

---

# V. Methodology

## 5. Evaluation Metrics

| Axis | Metrics |
|---|---|
| **Research (accuracy)** | **QWK** (primary, ordinal standard) · Accuracy · **macro-F1** (handles 42% imbalance) · **Cohen's κ** (unweighted, = Alaoui anchor) |
| **Deployment** | **exact integer match** · **within ±1 point** |
| **Explainability** | **SHAP plausibility** (overlap of the top SHAP words with the reference answer) |

- All numbers reported on **train + test** to expose overfitting.
- **No confidence intervals**: we use "comparable" / "narrow band", never "statistically tied".

> QWK is the primary research metric; deployment and explainability are reported alongside it, since
> the central claim is that **all three axes must be judged together**.

---

# VI. Experiment Setup

## 1. The model grid

- A **v01–v08 grid** run across **2 dataset variants** (1184 / 909) → one reproducible
  leaderboard, every cell reported on **train + test**.

| Ver | Pillar | What it adds |
|---|---|---|
| v01–v04 | Classical | TF-IDF + SVR, **threshold calibration**, **max-score feature**, per-bucket routing |
| v05 | RNN | BiLSTM + Attention grid |
| v06 | Transformer | Dual + Cross encoders × {mBERT, XLM-R, GTE} |
| v07 | Ensemble | weighted top-3 by val QWK |
| **v08** | **LLM** | **Qwen-KhmerGrader-4B** QLoRA fine-tune (10 epochs) |

- **Compute split:** Classical + BiLSTM run **locally on CPU**; encoders and the LLM run on **HPC/GPU**
  (~40 GPU-hours total). Seed 42, stratified 70/15/15 split throughout.

---

# VI. Experiment Setup

## 2. Explainability evaluation

- **Explainability.** Each pillar's champion is explained on the **same 909 test answers**
  (**n = 135, seed 42**) at the **Khmer word** level. **SHAP** → rank words → **plausibility** =
  fraction of the top SHAP words that appear in the reference answer.
- **Global view.** Aggregating SHAP values across answers gives a **global ranking** of the words
  each model relies on, which (for the classical model) are subject-content terms.

> Plausibility checks whether the highlighted words are the **rubric-relevant content** a teacher
> would look for in the reference answer.

---

# VII. Result

## Per-pillar champions (test set) and ranked readout

| Pillar | Best cell | QWK | Acc | Cohen κ | macro-F1 | Exact | Within ±1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Classical | TF-IDF+SVR (909) | 0.795 | 0.63 | 0.47 | 0.42 | 0.20 | 0.711 |
| RNN | BiLSTM+Attn (909) | **0.845** | 0.748 | 0.62 | 0.53 | 0.541 | 0.770 |
| Transformer | GTE dual + max-score (1184) | 0.820 | 0.730 | 0.62 | 0.54 | **0.573** | 0.770 |
| **LLM** | **Qwen-KhmerGrader-4B (909)** | 0.843 | **0.803** | **0.71** | **0.78** | **0.657** | **0.832** |
| _ref_ | _Qwen base, zero-shot_ | 0.500 | – | – | – | – | – |

- The four **trained** pillars tie on QWK; the Qwen base scores only **0.500 zero-shot**, so the LLM's
  standing is earned by QLoRA fine-tuning (**+0.34**), not the base.
- **LLM (Qwen-KhmerGrader-4B)** wins **deployment**: **66% exact, 83% within ±1**, MAE ≈ 0.93 pt, and
  leads Cohen κ (0.71) and macro-F1 (0.78).
- **RNN (BiLSTM)** tops **QWK** at **0.845**, effectively level with the LLM (0.843).
- **Transformer (GTE)** follows (QWK 0.820) with the best encoder exact match (0.573).
- **Classical (TF-IDF+SVR)** matches on QWK (0.795) at **~30 s on CPU**, but is weak on exact (0.20).

> **No single pillar dominates.** The four are **comparable on QWK** (a narrow ~0.05 band); the
> fine-tuned **LLM clearly wins deployment**. Data cleaning lifts the same model **+0.027 QWK**, more
> than any architecture swap (RQ2); the **max-score feature** gives consistent near-free gains (RQ4).
> <span class="small">Headline numbers are **uncalibrated**; calibration helps the classical model but hurts the BiLSTM (fragile, model-dependent).</span>

---

# VII. Result

## Explainability (RQ5)

**SHAP plausibility** (n = 135, seed 42): fraction of the top-20% SHAP words that appear in the reference.

| Pillar | SHAP plausibility |
|---|---:|
| Classical SVR | **0.66** |
| RNN BiLSTM | 0.59 |
| Transformer GTE | [HPC-pending] |
| LLM Qwen (capped) | [HPC-pending] |

- **SHAP gives a plausible explanation for every pillar:** about two of every three highlighted
  words are reference-relevant (classical 0.66, BiLSTM 0.59). The **global** top SHAP words are
  subject-content terms, confirming the models grade on content. **One unified, model-agnostic method.**
- The encoder and LLM plausibility cells are **HPC-pending**; the LLM runs SHAP under a **capped**
  evaluation budget (each evaluation is a full generation).

> SHAP is the same method in the study and in the live prototype, so the explanation the teacher
> sees is exactly what the thesis evaluated.

---

# VIII. Limitations & Future Works

**Limitations**

- **Single grader (no inter-rater ceiling).** We report agreement *with this teacher* and
  contextualize against the literature ceiling (~0.80–0.88).
- **Small corpus** (≈909–1,184) and only **41 questions**, with narrow four-subject coverage.
- **No classroom field study**: the "83% within ±1" claim is offline.
- **Plausibility** is an automatic reference-overlap proxy, not a human-rationale study.

**Future Works**

- Add a **second grader** to measure a human agreement ceiling.
- Complete **encoder + LLM SHAP plausibility** on HPC; evaluate **more open LLMs**.
- Collect **more questions** and subjects; **Khmer-Wikipedia pretraining**.
- Run a **classroom field study** and a **human rubric** study of explanations.

---

# IX. Conclusion

This study delivers the **first explainable, multi-pillar Khmer ASAG benchmark**: a 1,184-answer
corpus with two curated variants, a unified pipeline across **classical → RNN → encoder → LLM**,
and a cross-family SHAP explainability study.

- The **four pillars are comparable on QWK** (0.795–0.845, a narrow 0.05 band): no single research winner.
- The **fine-tuned LLM (Qwen-KhmerGrader-4B) clearly wins deployment** (66% exact, 83% within ±1, MAE ≈ 0.93 pt).
- **SHAP word attribution is plausible for every pillar**, one unified, model-agnostic explanation.
- The contribution is **realized as a live teacher-facing prototype** (score + SHAP explanation + feedback).

> **The core message:** accuracy, deployment, and **explainability** must be weighed **together**.
> No single model family dominates all three axes.

---

# References

<span class="small">

[1] Sung, Dhamecha, Mukhi (2022). *A survey on automated short answer grading with deep learning.* arXiv:2204.03503.
[2] Tan, Hu, Yeo, Cheong (2025). *A comprehensive review on automated grading systems in STEM using AI.* Mathematics 13(17):2828.
[3] Chang, Ginter (2024). *Performance of GPT-4 on automated short answer grading.* arXiv:2309.09338.
[4] Soulimani (Alaoui), El Achaak, Bouhorma (2024). *Deep learning based Arabic short answer grading in serious games.* IJECE 14(1).
[5] Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023). *QLoRA: Efficient finetuning of quantized LLMs.* NeurIPS.
[6] Xu et al. (2025). *Explainable AI for education: rubric-aligned chain-of-thought prompting.* Preprints.org.
[7] Pinto Jr., Shin (2025). *Evaluating attribution methods in ASAG systems (incl. Leave-One-Out).* J. Educational Measurement 62(2).
[8] Lyu, Apidianaki, Callison-Burch (2024). *Towards faithful model explanation in NLP: a survey.* Computational Linguistics 50(2).
[9] DeYoung et al. (2020). *ERASER: a benchmark to evaluate rationalized NLP models.* ACL.
[10] Devlin, Chang, Lee, Toutanova (2019). *BERT.* NAACL-HLT.
[11] Kumar, Boulanger (2020). *Explainable automated essay scoring.* Frontiers in Education 5:572367.
[12] Lachana (2025). *Automated scoring systems for open-ended questions in Dutch education.* M.Sc. thesis, Utrecht University.
[13] Buoy et al. (2022). *Pretrained models and evaluation data for the Khmer language.* Tsinghua Science and Technology.
[14] Hoang (2020). *khmer-nltk: Khmer NLP toolkit (word segmentation).* GitHub.
[15] Huot et al. (2025). *Educating in the age of AI: preparing Cambodian teachers and students.* JETELI 1(2).

</span>

---

# Appendix

 Royal University of Phnom Penh · Faculty of Engineering

**Appendix 1 · Dataset & fine-tuned models (Hugging Face):** [insert HF link]

**Appendix 2 · Source code:** github.com/PhorkNorak/kxs

**Appendix 3 · Live teacher-facing prototype:** [demo URL to insert]
<span class="small">teacher types Question + Reference + Student answer → choose any of the four pillars →
returns a **score**, a **Khmer word-attribution heatmap** (the same **SHAP** explanation), and
**written feedback** (open-source LLM, with a rule-based fallback). Positioned as **teacher-assist**
(human-in-the-loop), not an autonomous grader.</span>

<span class="small">Backup slides (compute, hyperparameters, XAI heatmaps, ensemble) follow.</span>

---

# THANK YOU

<br><br>

**Royal University of Phnom Penh · Faculty of Engineering**

**Presented by:** Phork Norak
**Advisor:** Dr. Khim Chamroeun
**Date:** [Date]

<br>

*Full leaderboards, champion predictions & training curves are in the repository.*

---

<!-- Backup -->

# Backup · Hyperparameter inventory

```
SEED            = 42
SPLIT           = 0.70 / 0.15 / 0.15  (stratified by score label)
TF-IDF          = char_wb 2–4-gram, max_features = 15000
SVR             = RBF, C = 1.0
BiLSTM          = hidden 128, 2 layers, embed 128, dropout 0.3, lr 1e-3, batch 64
TRANSFORMER     = lr 2e-5, batch 16, max_len 256, freeze bottom 6 layers,
                  max 20 epochs, early-stop patience 4 on val QWK
LLM (KhmerGrader) = QLoRA r=16 α=16, lr 2e-4, batch 4 × grad_accum 4, 10 epochs,
                  completion-only loss; greedy decode → first integer → clip
MAX_SCORE_NORM  = 20.0   (denominator for the max-score feature)
PREPROCESS      = raw | clean | segment        INPUT = ra | qar
```

---

# Backup · Compute budget & KhmerGrader family

| Pillar | Cells | Wall time (A40-equivalent) |
|---|---:|---:|
| Classical (v01–v04) + post-hoc | many | ~1.5 hr |
| RNN (v05) | 18 | ~75 min |
| Encoder transformers (v06) | 108 | ~10 hr |
| Max-score neural (v03b) | 126 | ~12 hr |
| LLM (v08, 10 epochs) | 1 | ~4 hr / cell |

**KhmerGrader family (909 test):** Qwen-KhmerGrader-4B QWK **0.843** (champion),
SEA-LION-KhmerGrader-E2B **0.802** (best exact **0.693** at 2.3B), Gemma-KhmerGrader-4B **0.763**.
Zero-shot bases 0.500 / 0.541 / [pending] → QLoRA lift **+0.34 / +0.26 / [pending]** (all bases instruction-tuned).

> The cheapest pillar (classical, < 2 hr CPU) is also a QWK champion; nearly all compute went to the neural + LLM pillars.

---

# Backup · XAI protocol & example heatmaps

**Protocol (one rule for all families).** Explain the *answer* at the **Khmer word** level on the
same 909 test items. **SHAP** → rank words → **plausibility** = fraction of the top SHAP words that
appear in the reference (`xai/` + `experiments/exp09_xai.py`; results in `results_xai/`).

> **What you see.** Heatmaps display the **original** answer (for readability), so stripped
> punctuation (។ / ?) appears as near-zero-importance tiles. That is correct behaviour, not a bug.

**Correct-Khmer heatmaps** are the **HTML** galleries (browsers shape Khmer; matplotlib/PNG cannot):

```
results_xai/no10c/heatmaps/{classical,bilstm,encoder,llm}/*_gallery.html
```

| Model | SHAP plausibility |
|---|---:|
| Classical | **0.66** |
| BiLSTM | 0.59 |
| Encoder | [HPC-pending] |
| LLM (capped) | [HPC-pending] |

---

# Backup · Ensemble (v07), a negative result

We ensembled the **top-3 cells by validation QWK** (softmax-weighted average + recalibration). It
**did not beat the best single cell on any dataset**: the top candidates were **too similar**
(same family), so weighted averaging had no leverage.

> **Diversity, not weighting, is the bottleneck.** A forced-diverse pool {classical + encoder + LLM}
> is the promising direction (future work).

> Render the deck with the VS Code **Marp** extension (or `marp docs/slides.md`). For Khmer example
> figures, screenshot the **HTML galleries** (browsers shape Khmer; matplotlib does not).
