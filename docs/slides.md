---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Automatic Short Answer Grading for Khmer'
footer: 'Phork Norak · Royal University of Phnom Penh'
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

# Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education

### A Reproducible Multi-Pillar Benchmark (Classical · RNN · Transformer · LLM) with a Cross-Family Faithfulness Study

<br>

**Phork Norak**
**Supervisor:** Dr. Khim Chamroeun
**Department:** [Department] · **Royal University of Phnom Penh**
**Date:** [Date]

---

## Outline

1. **Background & Motivation**: why ASAG, why Khmer, why **explainability**
2. **Problem Statement & Research Questions**
3. **Literature Review**: ASAG from classical to the LLM era; XAI & faithfulness; the Khmer gap
4. **Methodology**: dataset, cleaning protocol, four model pillars, the grid, **XAI methods**
5. **Results**: per-pillar champions, generalization analysis, and the **explainability study**
6. **Conclusion**: findings, contributions, limitations, future work
7. **System**: the live teacher-facing **prototype** (demo)

<br>

> **One-line thesis:** On a small, single-grader Khmer corpus, *no single model family dominates across all three axes*: the four pillars are **comparable on QWK** (a narrow 0.05 band), a fine-tuned LLM wins the deployment metrics, and on explainability **LOO word attribution (occlusion) is reliably faithful for every pillar** (one unified, model-agnostic method). Accuracy, deployment, and **explainability** must be judged together.

---

# 1 · Background & Motivation

---

## 1.1 What is ASAG, and why does it matter?

- **Automatic Short Answer Grading (ASAG)** = automatically scoring a student's free-text
  answer against a reference answer, on a numeric scale.
- Distinct from essay scoring: short answers are graded on **content correctness**
  relative to a reference, not on length, style, or grammar.
- **Why it matters:** grading free-text answers by hand is slow and inconsistent. Reliable
  ASAG lets teachers give faster, more consistent feedback and scales formative assessment.
- Modern NLP (transformers, LLMs) has pushed English/Arabic/Chinese ASAG forward, but the
  benefits have **not reached most low-resource languages**.

---

## 1.2 Khmer is severely under-served by NLP

- **Khmer** (ភាសាខ្មែរ) is the official language of Cambodia, spoken by **17M+ people**, yet
  still treated as **low-resource** in NLP.
- Multilingual encoders (mBERT, XLM-R, GTE-multilingual) see **little Khmer** in pretraining;
  Khmer has **no whitespace word boundaries**, so even tokenization is non-trivial.
- Existing Khmer NLP work targets tokenization/POS tagging, news classification, and scene
  text, **not** educational assessment.
- **To our knowledge, there is no prior published Khmer ASAG (or essay-scoring) benchmark.**

<br>

> **The opportunity:** build the *first reproducible Khmer ASAG benchmark* and measure how
> far each family of grading model can actually go on a real classroom corpus.

---

## 1.3 Why explainability is essential in education

- A grade is a **high-stakes decision**: students, teachers, and parents need to know **why**
  an answer received its score, a bare number is not enough for trust or appeal.
- Teachers will only adopt an automatic grader they can **audit**: does it reward the *right*
  Khmer words (the content the rubric cares about), or is it exploiting spurious cues?
- An explanation is only useful if it is **faithful**, it must reflect what the model *actually*
  used, not just produce a plausible-looking highlight. Plausible-but-unfaithful explanations
  are arguably *worse* than none (false confidence).

<br>

> This thesis therefore treats **Explainable AI as a first-class axis**: every model family is
> not only scored for accuracy, but its explanations are **measured for faithfulness** and
> **plausibility** (overlap with the teacher's reference answer).

---

# 2 · Problem Statement & Research Questions

---

## 2.1 Problem statement

We want to grade Khmer short answers automatically and **honestly characterize** what is
achievable. Three properties of the real data make this hard:

| Challenge | Detail |
|---|---|
| **No benchmark / no baselines** | Nothing published to compare against for Khmer ASAG. |
| **Heterogeneous scoring scales** | Each question has its own max score ∈ {5,6,7,8,10,12,15,20}, a raw score of "8" means different things per question. |
| **Small, single-grader corpus** | 1,184 answers graded by **one** teacher → limited data and no inter-rater ceiling. |

We must also separate two questions usually conflated in ASAG papers:
**research quality** (ordinal agreement, QWK) vs **deployment quality** (does it predict the
*exact* teacher score in points?).

---

## 2.2 Research questions

> **RQ1: Model families.** How far can each pillar (classical ML, RNN, encoder
> transformer, and fine-tuned LLM) go on a 1,184-sample Khmer corpus?

> **RQ2: Data vs algorithm.** On small-data ASAG, does **data cleaning** move results more
> than swapping the algorithm?

> **RQ3: Research vs deployment.** Does the model that wins the standard research metric
> (QWK) also win the metrics that matter in a classroom (exact-score match, within ±1 point)?

> **RQ4: Cheap levers.** Do two low-cost additions (encoding each question's **max score**
> as a feature, and **post-hoc threshold calibration**) improve grading?

> **RQ5: Explainability (central).** Across the four families, which produces the most
> **faithful** explanations (do the words it highlights actually drive the score?), how
> **plausible** are they (do they overlap the teacher's reference answer), and is there an
> **accuracy ↔ explainability trade-off**?

> **RQ6: Robustness.** How much does performance depend on having **seen the question** before?
> i.e., how large is the **question-leakage** effect, and how well do models grade *unseen* questions?

---

# 3 · Literature Review

---

## 3.1 ASAG: from similarity measures to transformers

- **Classical era.** Mohler & Mihalcea (2009) and Mohler, Bunescu & Mihalcea (2011) grade
  short answers with **semantic-similarity** and dependency-graph features, establishing
  reference-vs-answer comparison as the core idea.
- **Shared-task era.** **SemEval-2013 Task 7** (Student Response Analysis; Dzikovska et al.)
  standardized datasets (SciEntsBank, Beetle) and evaluation, energizing the field.
- **Deep-learning era.** A 2022 survey (Sung et al., *arXiv:2204.03503*) traces the shift
  from word embeddings → sequential models → **attention/transformers**; BERT-family
  encoders (and SBERT) become the workhorses, with augmentation and transfer learning used
  for low-data settings.

---

## 3.2 The LLM era, and the standard metric

- **LLMs for grading.** Recent work fine-tunes or prompts LLMs for ASAG: GPT-4 with prompt
  engineering (Chang & Ginter, *arXiv:2309.09338*; LAK 2025), **QLoRA** cross-prompt
  fine-tuning (IJAIED 2025), and LLM-enhanced hybrids (FusionASAG, 2025). Finding across
  studies: general LLMs are *competitive* but **specialized fine-tuned models** still lead.
- **Metric.** **Quadratic Weighted Kappa (QWK)** is the de-facto standard for ordinal
  short-answer / essay scoring (ASAP-SAS, SemEval), it rewards near-misses and penalizes
  large disagreements, unlike plain accuracy or unweighted kappa.

<br>

> We adopt **QWK as the primary metric**, and additionally report classroom-oriented
> *deployment* metrics (exact-score match, within ±1 point).

---

## 3.3 Closest anchor (Arabic) and the Khmer gap

**Arabic ASAG, Soulimani/Alaoui et al. (2024, IJECE).** Closest comparable study: a
self-collected Arabic dataset (**1,276 answers, 18 questions, 3 classes 0–2**, 6th-grade,
~5.6 words/answer). Best model = a transformer at **95.67% train / 77.22% test accuracy**;
the paper explicitly notes the transformer **overfits faster** as epochs grow. *(They report
**unweighted** Cohen's κ = 0.60 on test, not QWK; see §5.5.)*

**Khmer NLP landscape.** Pretrained Khmer models + POS/news data (Buoy et al., *TST* 2022;
ACM *TALLIP* 2021), Khmer-NLTK word segmenter (Hoang, 2020), KhmerST scene text (ACCV 2024)
, **but no ASAG.**

> **Gap:** no Khmer ASAG benchmark, no multi-family comparison, **no explainability study**,
> and little honest train/test-gap reporting in low-resource ASAG. This thesis addresses all.

---

## 3.4 Explainable AI: methods and the faithfulness problem

- **Attribution methods.** Model-agnostic perturbation (**LIME**, Ribeiro et al. 2016;
  occlusion) and gradient-based **saliency** (Simonyan et al. 2014; gradient×input,
  Li et al. 2016) attribute a prediction to input tokens. **SHAP** (Lundberg & Lee 2017)
  unifies them.
- **Attention ≠ explanation.** Raw attention weights are *not* guaranteed faithful
  (**Jain & Wallace, 2019**); the rebuttal (**Wiegreffe & Pinter, 2019**) argues attention
  *can* be informative, so it must be **evaluated**, not trusted or dismissed.
- **Faithfulness evaluation.** **ERASER** (DeYoung et al. 2020) defines **comprehensiveness**
  (does removing the "important" tokens change the prediction?) and **sufficiency** (do the
  important tokens alone suffice?), the metrics we adopt.

<br>

> **Khmer-specific gap:** none of these methods has been applied to or evaluated for **Khmer
> grading**. This thesis runs them across *all four* model families and measures faithfulness.

---

# 4 · Methodology

---

## 4.1 The dataset we built

| Property | Value |
|---|---|
| Graded answers | **1,184** (after dropping 1 incomplete row) |
| Schools / classes / students | 2 (AS, PT) · 8 classes · 203 students |
| Subjects | Biology (435), History (434), Geography (307), Earth Science (9) |
| Unique questions | **41** |
| Max-score per question | 5, 6, 7, 8, 10, 12, 15, 20 (varies by question) |
| Grader | One trained teacher (single-grader corpus) |

**Ordinal label** = `round(4 · StudentScore / MaxScore) ∈ {0,1,2,3,4}`:

| Score label | 0 | 1 | 2 | 3 | 4 (full) |
|---|---:|---:|---:|---:|---:|
| Count | 14 | 153 | 327 | 191 | **499** |
| Percent | 1.2% | 12.9% | 27.6% | 16.1% | **42.1%** |

Heavy skew toward full credit → a majority-class predictor already gets **0.42 accuracy**.

---

## 4.2 Dataset curation, three variants (RQ2)

While inspecting errors, two noise sources stood out:

1. **"10C Biology" subset**: inconsistent grading (near-identical answers, different scores);
   a disproportionate share of large errors. Removing it → **909 rows**.
2. **Score = 0 outliers**: only 14 rows (1.2%); effectively untrainable label noise.
   Removing them → **895 rows**.

| Variant | Rows | Definition |
|---|---:|---|
| `full` | 1,184 | original, benchmark anchor |
| `no10c` | 909 | drop the noisy 10C-Biology subset |
| `no10c_no0` | 895 | also drop the 14 score=0 outliers |

> Every experiment runs on **all three** variants → we can attribute gains to **data** vs **algorithm**.

---

## 4.3 Shared pipeline (one recipe for every cell)

```
data.csv (1184 / 909 / 895)
   ↓  stratified 70/15/15 split (seed = 42)  →  train / val / test
   ↓  Φ preprocessing, 3 modes
        raw      : NFC + strip invisibles, trim whitespace
        clean    : raw + KCC syllable normalize + strip punctuation
        segment  : clean + khmernltk word segmentation
   ↓  Ψ input format, 2 modes
        ra   : (Answer,            Reference)
        qar  : (Question + Answer, Reference)
   ↓  model (one of four pillars)  →  predict normalized score ŷ ∈ [0,1]
   ↓  round to {0..4};  report train + test on 8 metrics
```

**Text normalization & cleaning (Khmer-aware).** `NFC` → **strip invisibles**
(zero-width U+200B, ZWNJ/ZWJ, BOM, controls, bullets • ◦ …) → KCC syllable reorder →
strip Khmer (។ ៕) + ASCII (? . ,) punctuation → optional `khmernltk` segmentation.
**Digits are kept** (content: dates, quantities). A corpus audit found and removed **561
zero-width spaces** (+ bullets); an **ablation** shows this refinement shifts classical QWK by
only **+0.001 to +0.004** across the three variants (**a negligible shift**) → results are robust
to the residual noise (`results_stats/cleaning_ablation.csv`).

- **Target:** `score / max_score ∈ [0,1]`, trained as **bounded MSE regression**.
- **Trainable models:** AdamW, **early-stop on validation QWK**, single seed (42).
- Consistent recipe → differences come from the **axis under test**, not tuning noise.

---

## 4.4 Four model pillars

| Pillar | Models | Backbone | Head |
|---|---|---|---|
| **Classical** | TF-IDF cosine · **TF-IDF + SVR** | TF-IDF char 2–4-gram | cosine / RBF-SVR on `[a; b; |a−b|; a⊙b; cos]` |
| **RNN** | BiLSTM + Attention | char-level BiLSTM | 4-way MLP → σ |
| **Transformer (BERT-family)** | Dual / Cross encoder × {**mBERT, XLM-R, GTE**} | encoder transformer (Devlin 2019; Conneau 2020) | 4-way / [CLS] MLP → σ |
| **LLM** | **Qwen 3.5 4B** (QLoRA fine-tune) | decoder transformer | emits the integer score as text |

<span class="small">Note: **mBERT, XLM-R, GTE and Qwen are all Transformers** (Vaswani et al. 2017), the
"Transformer" pillar uses **encoder** transformers (BERT-family); the LLM pillar is a **decoder**
transformer. Classical and BiLSTM are the non-transformer baselines.</span>

<br>

- **Evaluation metrics (standard + anchor-comparable, deliberately lean).** Research: **QWK**
  (primary, ordinal field standard) · **Accuracy** · **macro-F1** (handles the 42% class imbalance)
  · **Cohen's κ (unweighted)** (same metric as the Alaoui anchor → direct comparison). Deployment:
  **exact integer-match** · **within ±1 point**. (Redundant metrics, adjacent-accuracy, raw-MAE,
  pct-MAE, were dropped to avoid clutter.)
- GTE ships a NaN-corrupted RoPE cache under fp16; we rebuild it in fp32 (`models/dual.py`).

---

## 4.5 The v01–v08 experiment grid

| Ver | Pillar | What it adds |
|---|---|---|
| **v01** | Classical | TF-IDF cosine + SVR baseline |
| **v02** | Classical | **post-hoc threshold calibration** (coordinate-descent on 4 cut-points, maximize val QWK) |
| **v03 / v03b** | Classical / Neural | **max-score as an extra feature** (RQ4), to SVR (v03) and to neural heads (v03b) |
| **v04** | Classical | per-max-score-bucket SVR (routing) |
| **v05** | RNN | BiLSTM + Attention grid |
| **v06** | Transformer | Dual + Cross encoders × {mBERT, XLM-R, GTE} |
| **v07** | Ensemble | weighted top-3 by val QWK |
| **v08** | **LLM** | **Qwen 3.5 4B** QLoRA fine-tune (7 epochs) |

Run across **3 datasets** → a single reproducible leaderboard of **hundreds of cells**,
all reported with **train + test** to expose any overfitting.

---

## 4.6 Explainability methods (RQ5)

The XAI is **Leave-One-Out (LOO) word attribution (text highlighting)**: drop each word, measure the
score change — the **single, unified method** applied identically to all four pillars. LOO requires
no gradients or internal model access, works on the non-differentiable SVR, and is the standard ASAG
attribution (Pinto 2025). Each family's champion is explained on the **same 895 test answers** at the
level of **Khmer word units** (segmented with khmernltk).

| Pillar | Explanation method | Status |
|---|---|---|
| **Classical** TF-IDF+SVR | **LOO occlusion** | ✅ run (CPU) |
| **RNN** BiLSTM | **LOO occlusion** | ✅ run (CPU) |
| **Transformer** GTE | **LOO occlusion** | ⚙️ HPC |
| **LLM** Qwen | **LOO occlusion** | ⚙️ HPC |

We evaluate explanations on **two axes** (Jacovi & Goldberg 2020):
- **Faithfulness**, does the explanation reflect the model's *actual* reasoning? **ERASER**:
  comprehensiveness (remove top-k → score should drop) and sufficiency (keep top-k → score should
  hold), vs a **random-removal** baseline.
- **Plausibility**, does it *look* right? our **reference-overlap** proxy: fraction of top
  important words that appear in the teacher's reference answer.

---

# 5 · Results

---

## 5.1 Per-pillar champions (test set)

| Pillar | Best cell | QWK | Acc | Exact | Within ±1 | Cleaning |
|---|---|---:|---:|---:|---:|:--:|
| **Classical** | TF-IDF+SVR (895) | 0.795 | 0.630 | 0.200 | 0.711 | refined |
| **RNN** | BiLSTM+Attn (895) | **0.845** | 0.748 | 0.541 | 0.770 | refined |
| **Transformer** | GTE dual + max-score (1184) | 0.820 | 0.730 | **0.573** | 0.770 | prior* |
| **LLM** | **Qwen 3.5 4B (909)** | 0.843 | **0.803** ⭐ | **0.657** ⭐ | **0.832** ⭐ | refined |

<span class="small">All numbers are **uncalibrated** (calibration is a separate val-selected ablation, §5.3: it helps the classical model but hurts the BiLSTM, so we don't fold it into the headline). QWK primary; full standard set (+ Cohen κ, macro-F1) in §5.10. classical/RNN re-run under the **refined cleaning**; *encoder/LLM on the **prior** cleaning (re-run pending; cleaning effect ≈0.02 QWK).</span>

<br>

> **No single pillar dominates**: the four are **comparable on QWK** (classical 0.795,
> RNN 0.845, encoder 0.820, LLM 0.843 — a narrow 0.05 band). The fine-tuned **LLM** wins the
> **deployment metrics** (exact 66%, within ±1 83%, MAE ≈0.93 pt) by clear margins.

<span class="small">⚠️ These are **random-split, seen-question** numbers. The QWK scores sit in a narrow 0.05 band (§5.10), the LLM leads the deployment metrics by clear margins, and §5.11 shows performance **collapses on unseen questions.** Read 5.1 as "what's achievable when questions are seen," not as a strict pillar ranking.</span>

---

## 5.2 Research vs deployment, the trade-off (RQ3)

| Goal | Winner | Cell | Number |
|---|---|---|---:|
| Highest **QWK** (research) | RNN *(tied, n.s.)* | BiLSTM+Attn (895) | **0.845** |
| Highest **accuracy** | LLM | Qwen 3.5 4B (909) | **0.803** |
| Highest **exact integer match** | LLM | Qwen 3.5 4B (909) | **0.657** |
| Highest **within ±1 point** | LLM | Qwen 3.5 4B (909) | **0.832** |
| Lowest **MAE (points)** | LLM | Qwen 3.5 4B (909) | **0.93 pt** |
| **Cheapest** | Classical | TF-IDF+SVR | ~30 s, CPU only |

**Head-to-head on 909 (encoder vs LLM):** QWK ties (encoder 0.842 ≈ LLM 0.843), but Qwen beats the GTE
encoder clearly on the deployment metrics (exact match, accuracy, within-±1).

> The four pillars are **comparable on QWK** (the lead is small), so pick by the *other*
> axes: **cheapest + self-explaining → classical**; **classroom point-accuracy → LLM.**

---

## 5.3 Data cleaning beats algorithm choice (RQ2)

Same architecture (GTE dual + max-score, *prior cleaning*), best cell per dataset:

| | `full` (1184) | `no10c` (909) | `no10c_no0` (895) |
|---|---:|---:|---:|
| **Test QWK** | 0.820 | 0.842 | **0.847** |
| Train QWK | 0.874 | 0.944 | 0.892 |

- Removing the noisy subset + score=0 outliers lifts the **same model** by
  **+0.027 QWK** (0.820 → 0.847), *for free, no algorithm change.*
- **Calibration is a *fragile, model-dependent* lever (reported as an ablation, not in the headline):**
  by the val-selection rule it *helps the classical* model (val 0.758→0.829; test 0.795→**0.847**) but
  *hurts the BiLSTM on test* (test 0.845→0.818 even though val rose 0.797→0.806). Because the 4
  cut-points are fit on only ~134 val rows, the gain doesn't transfer reliably, so headline numbers
  (§5.1, §5.10) are **uncalibrated**.

> **Answer to RQ2:** on small-data ASAG, **data quality and cheap post-hoc levers move the
> needle more than swapping architectures.**

---

## 5.4 The max-score feature helps (RQ4)

Each question has its own max score (5–20). A normalized prediction of "0.5" means 5/10 on
one question but 10/20 on another, the model can't tell unless we **give it the scale**.

- **v03 (SVR):** concatenate `max_score / 20` to the TF-IDF feature vector.
- **v03b (neural):** concatenate `max_score / 20` to the MLP head input.

**Effect (vs the no-feature version):** consistent gains on the **deployment**
metrics, e.g. on 895, neural heads improve raw-exact **+4.5 pp** and within-±1 **+2.8 pp**.

> A one-number conditioning signal teaches every head the per-question scoring scale, the
> clearest, cheapest neural gain in the whole grid.

---

## 5.5 In context, same metric as the Alaoui anchor (Cohen κ)

We report the **same unweighted Cohen κ** Alaoui et al. use, so the comparison is on a like metric
(still different datasets, read as context, not a leaderboard):

| Study (best model) | Language | Scale | **Cohen κ (test)** |
|---|---|---|---:|
| Alaoui et al. 2024, BERT | Arabic | 3-class | 0.48 |
| Alaoui et al. 2024, transformer | Arabic | 3-class | 0.60 |
| **This work, classical** | **Khmer** | **5-class** | **0.47** |
| **This work, encoder / RNN** | **Khmer** | **5-class** | **0.62** |
| **This work, LLM (Qwen)** | **Khmer** | **5-class** | **0.71** |

- On the **same metric**, our encoder/RNN (~0.62) match Alaoui's transformer (0.60), and our
  **LLM (0.71) is higher**, while our classical (0.47) ≈ their BERT (0.48).
- For reference, our **primary QWK** (quadratic-weighted) is **0.82–0.85**; ASAP-SAS leading
  systems reach ≈0.75–0.78 QWK (English).

> ⚠️ Still different datasets/scales (5-class Khmer vs 3-class Arabic), so this is **contextual**.
> The fully fair head-to-head is the **train/test generalization gap** (next slide).

---

## 5.6 Honest generalization, gap vs Alaoui (apples-to-apples)

Both works report **accuracy**, so we compare the **train → test accuracy gap**:

| Source | Model | Train acc | Test acc | **Gap** |
|---|---|---:|---:|---:|
| Alaoui et al. | LSTM | 83.95% | 69.62% | +14.3 pp |
| Alaoui et al. | **Transformer (their best)** | **95.67%** | **77.22%** | **+18.5 pp** |
| **Ours** | TF-IDF+SVR (895, uncal.) | 78.6% | 63.0% | +15.6 pp |
| **Ours** | GTE dual + max-score (1184) | 74.4% | 73.0% | **+1.4 pp** |
| **Ours** | Qwen 3.5 4B (909) | 86.3% | **83.9%** | **+2.4 pp** |

Alaoui's own paper notes its transformer "**overfits faster** … the gap becomes larger with
epochs." Our **deep** cells generalize **far tighter** (GTE +1.4 pp, LLM +2.4 pp vs +18.5 pp); even
the simple classical (+15.6 pp) stays under Alaoui's transformer.

> **No hidden overfitting under our headline numbers**, reported test performance is what a
> classroom would see.

---

## 5.7 Where the errors live, per-question difficulty

Mean raw-MAE (points) by question max-score, best encoder cell:

| Max score | n (test) | Raw-MAE | % of max |
|---:|---:|---:|---:|
| 5 | 14 | 0.50 | 10% |
| 6–8 | 30 | 0.9–1.4 | 15–17% |
| **10** | **48** | 1.08 | 11% |
| **15** | **36** | **2.42** | 16% |
| 20 | 7 | 2.57 | 13% |

- **Low-max questions (5–7) are near-solved.**
- **High-max questions (15–20)**, more partial-credit granularity, are the **bottleneck**.

> Future accuracy gains should target high-max, fine-grained questions specifically.

---

## 5.8 Explainability results, faithfulness across families (RQ5)

LOO occlusion for each family, scored on the same 895-set test answers (n=135, seed 42).
**AOPC** = comprehensiveness/sufficiency averaged over k∈{10,20,30,40,50}% (ERASER-style):

| Model · LOO | gap vs random | AOPC-comp ↑ | AOPC-suff ↓ | **Faithful?** | Plaus. |
|---|---:|---:|---:|:--:|---:|
| **Classical** SVR · occlusion | +0.096 | **+0.150** | 0.011 | **✅ yes** | 0.75 |
| **RNN** BiLSTM · occlusion | +0.257 | **+0.289** | 0.064 | **✅ yes** | 0.69 |
| **Transformer** GTE · occlusion | +0.135 | +0.193 | 0.007 | **✅ yes** | 0.68 |
| **LLM** Qwen · occlusion | +0.047 | +0.126 | 0.359 | **✅ yes** | 0.63 |

- **LOO occlusion is *reliably* faithful** for every pillar (AOPC-comp +0.150 classical, +0.289
  BiLSTM, +0.193 encoder, +0.126 LLM; gap > 0 in every case, strongest for the BiLSTM and weakest
  for the LLM): removing the words it flags moves the score more than random. The same method works
  on the non-differentiable SVR, the BiLSTM, the GTE encoder, and the fine-tuned LLM — no gradient access required.

---

## 5.9 The third axis, accuracy ↔ deployment ↔ explainability

| Pillar | QWK | Deployment | **Explainability (LOO)** |
|---|---|---|---|
| **Classical** SVR | 0.795 | weak (exact 0.20) | **faithful + plausible** ✅ |
| **RNN** BiLSTM | 0.845 | mid (exact 0.54) | **faithful** ✅ |
| **Transformer** GTE | 0.820 | good | **faithful** ✅ |
| **LLM** Qwen | 0.843 | **best (66% exact)** | **faithful** ✅ (weakest) |

<br>

> **Extended thesis finding:** the "no free lunch" result is **three-dimensional**, and each
> axis behaves differently. On **QWK** the pillars are **comparable** (a 0.05 band,
> §5.10); on **deployment** the LLM is the **clear** best; on **explainability**,
> **LOO occlusion is the unified method and is reliably faithful for every model** →
> no accuracy-vs-explainability trade-off.

<br>

<span class="small">LOO is self-consistent with the ERASER metric (both use deletion), confirming the lower bound.
Cross-model AOPC is model-sensitive, contextual not a strict ranking (Normalized-AOPC 2024).
XAI models anchored on the same 895 split (seed 42), all four pillars.</span>

---

## 5.10 The full standard metric set

Reported set = field-standard **QWK** (primary) + **Accuracy** + **macro-F1** + **Cohen's κ**
(unweighted, for direct comparison with Alaoui et al.):

| Pillar | QWK | Cohen κ | Accuracy | macro-F1 |
|---|---|---|---|---|
| Classical (TF-IDF+SVR) | 0.795 | 0.47 | 0.63 | 0.42 |
| RNN (BiLSTM) | **0.845** | 0.62 | 0.75 | 0.53 |
| Transformer (GTE)* | 0.820 | 0.62 | 0.73 | 0.54 |
| **LLM (Qwen)*** | 0.843 | **0.71** | **0.80** | **0.78** |

- **On QWK the four pillars sit in a narrow 0.05 band** (0.795–0.845) — no single research winner;
  the classical and BiLSTM champions (same 895 test set) differ by only 0.05.
- **The LLM clearly leads on accuracy / macro-F1 / Cohen κ** and the deployment metrics.

<span class="small">Classical/RNN refreshed under refined cleaning; *encoder/LLM on prior cleaning.
**All headline QWKs (here and §5.1) are uncalibrated**, consistent across pillars; calibration is a
separate fragile, model-dependent ablation (§5.3). Metrics from saved continuous predictions; Cohen κ < QWK
throughout (unweighted penalizes more).</span>

---

## 5.11 Robustness to unseen questions (the leakage test)

Our split is stratified **per row** over only **41 questions**, so train and test share
questions. We re-evaluated the **classical champion under a question-held-out split**
(GroupShuffleSplit by `QuestionID`, no shared questions), 5 seeds:

| Split | Classical QWK (mean ± std, 5 seeds) |
|---|---|
| Random (seen questions) | **0.759 ± 0.041** |
| **Question-held-out (unseen)** | **0.354 ± 0.106** |
| **Leakage gap** | **−0.41 QWK** |

- **Performance collapses on unseen questions**, most of the seen-question score was
  question-specific memorization, not transferable grading skill.
- With only 41 questions, each unseen-question test set is ~6 questions → **high variance**
  (±0.11), but the drop is unambiguous.

> **Honest headline finding:** small-question-pool ASAG benchmarks (including ours and much of
> the literature) are **substantially inflated by question leakage.** Reported QWK should be
> read as *seen-question* performance; **grading genuinely new questions on this corpus is an
> open problem.** (Neural/LLM unseen-question runs are HPC-pending and expected to drop similarly.)

---

# 6 · Conclusion

---

## 6.1 Findings (answering the research questions)

1. **RQ1: Model families.** All four pillars reach **QWK 0.80–0.85** on cleaned data;
   none is universally best.
2. **RQ2: Data vs algorithm.** **Data cleaning** lifts the same model by **+0.027 QWK**, 
   **bigger than any architecture swap** on this corpus.
3. **RQ3: Research vs deployment.** On QWK the pillars are **comparable** (a 0.05 band);
   the **LLM is the clear winner on accuracy/deployment** (66% exact,
   83% within ±1, MAE ≈0.93 pt).
4. **RQ4: Cheap levers.** The **max-score feature** helps consistently; **threshold
   calibration** helps the *classical* model (+0.05 QWK) but is **fragile and model-dependent**
   (it *hurts* the BiLSTM on test), so headline numbers are uncalibrated.
5. **RQ5: Explainability.** **LOO occlusion is reliably faithful for every pillar** (gap > 0 for all
   four; strongest for the BiLSTM, weakest but still positive for the LLM) → one unified,
   model-agnostic explanation. A third axis beyond accuracy/deployment.
6. **RQ6: Robustness.** Under a **question-held-out** split, classical QWK **collapses
   0.76 → 0.35**: the seen-question scores are heavily inflated by **question leakage**.

> Practical guidance: a **30-second CPU classical model** matches everything on QWK and
> **explains itself faithfully (via occlusion)**; a **fine-tuned Qwen** is the right
> choice when *exact teacher scores* matter. **Grading genuinely unseen questions remains open.**

---

## 6.2 Contributions

1. **First explainable Khmer ASAG benchmark**: 1,184-answer corpus, 3 curated variants, a
   unified multi-pillar pipeline (classical → RNN → encoder → LLM), **and a cross-family
   faithfulness study**, the first XAI evaluation for Khmer grading.
2. **A model-agnostic faithfulness + plausibility (reference-overlap) protocol** (ERASER comprehensiveness/
   sufficiency **+ AOPC** vs a random baseline, over Khmer word units) usable across *all*
   families, showing **LOO occlusion is reliably faithful for every pillar** (one unified,
   model-agnostic explanation method).
3. **A quantified question-leakage analysis**: a question-held-out protocol showing
   seen-question ASAG QWK is inflated by **~0.43** on this corpus; a cautionary, reusable
   evaluation lesson for small-question-pool grading.
4. **Honest, multi-variant reporting**: every cell reported on train + test across three dataset
   variants, exposing overfitting and separating data-quality gains from algorithm gains.
5. **Max-score-as-a-feature** (consistent near-free gains) **+ a threshold-calibration ablation**
   showing post-hoc calibration is **model-dependent** (helps classical, hurts BiLSTM) on
   heterogeneous per-question scales.
6. **Honest train + test reporting** on every cell: overfitting gap as a first-class result.
7. **A live teacher-facing prototype** (§7) that puts all four pillars behind a model selector and
   returns **score + faithful explanation + feedback**, the XAI contribution realized in practice.

---

## 6.3 Limitations (stated plainly)

- **Single grader (no IAA).** A 2nd grader was not obtainable, so we cannot measure the
  human–human ceiling; we report agreement *with this grader* and contextualize our QWK (0.80–0.85)
  against the literature ceiling (~0.80–0.88). **Mitigated** by reporting across three dataset variants, not eliminated.
- **Question leakage is now *measured*, not just flagged** (§5.11): seen-question QWK is
  inflated ~0.43 vs unseen-question. We **report both**; unseen-question is the honest
  generalization number. (Full grid + neural/LLM unseen-question runs are HPC-pending.)
- **Small corpus** (≈895–1,184) and **only 41 questions** → unseen-question test sets are tiny
  (~6 Q) and high-variance.
- **No classroom field study** → the "83% within ±1" claim is offline.
- **XAI scope.** LOO faithfulness is reported for all four pillars on the same 895 split; occlusion-by-occlusion is
  self-consistent (the explanation and its ERASER evaluation share the same deletion perturbation);
  plausibility (reference-overlap) is an automatic proxy, not a human-rationale study (Jacovi \& Goldberg 2020).
- **Cleaning/digits.** Cleaning was refined post-audit (zero-width/bullets removed); **digits are
  retained** as content, a potential lexical-leakage source not separately controlled.

---

## 6.3b Ethics & data governance

- **Human-subjects data:** real student answers (minors). Use requires informed consent
  (school + guardians) and an ethics/IRB approval or exemption, *to be stated by the author.*
- **Privacy:** released CSVs carry only **pseudonymous codes** (`SchoolID`/`ClassID`/`StudentID`/
  `QuestionID`), no names; answer text to be scrubbed of any PII before release.
- **Intended use:** a **teacher-assist** tool, **not** an autonomous high-stakes grader, 
  human-in-the-loop, and surface the explanation *with its faithfulness caveat*.
- **Fairness:** single-grader labels reflect one teacher; narrow subject/grade coverage → no
  claim of generalization beyond this setting.

<span class="small">Full statement in `docs/ethics.md`. ⚠️ = author must confirm consent / approval / data-licence before submission.</span>

---

## 6.4 Future work

| Priority | Action | Expected payoff |
|---|---|---|
| **High** | Re-run the **full grid + v08 under the question-held-out split** | honest unseen-question numbers for every pillar |
| **High** | Target the leakage gap: **Khmer-Wikipedia pretraining** + cross-question augmentation | improve true unseen-question grading |
| **High** | Complete **encoder + LLM faithfulness (AOPC)** on HPC | finishes the cross-family XAI |
| **Medium** | Collect **more questions** (the 41-question pool drives leakage + variance) | larger, lower-variance unseen-question test |
| **Medium** | Evaluate **more open LLMs** (incl. re-tuning Gemma with text-specific LoRA targets) | broaden the LLM pillar |
| **Low** | **Classroom field study**; **human rubric study** of explanations | validate deployment + explanation quality |

<span class="small">A 2nd human grader (true inter-annotator agreement) was not obtainable for this work; reporting across three dataset variants and the literature ceiling are used as partial mitigation.</span>

---

## 6.5 Summary

> **Five headline results (all reported honestly)**
>
> - **Seen-question QWK ≈ 0.80–0.85 across all four pillars** (classical 0.795, RNN 0.845,
>   encoder 0.820, LLM 0.843; all uncalibrated) — a narrow 0.05 band, no single research winner.
> - **LLM is the clear winner on deployment**, 66% exact, 83% within ±1, MAE ≈0.93 pt
>   (by large margins over the other pillars).
> - **LOO occlusion is reliably faithful for every pillar** (gap > 0 for all four; strongest BiLSTM,
>   weakest but still positive LLM) → one unified, model-agnostic explanation.
> - **Question leakage inflates QWK by ~0.40**, unseen-question grading (classical 0.35) is an
>   open problem; we report it openly.
> - **Robust to cleaning:** removing 561 zero-width spaces shifts QWK <0.01 (negligible).

<br>

We deliver the **first *explainable* multi-pillar Khmer ASAG benchmark**, hardened with
**three dataset variants, AOPC faithfulness, and a question-held-out leakage
analysis.** The honest picture: pillars are **comparable on QWK**, the **LLM wins deployment**,
**LOO occlusion explains faithfully across all four pillars**, and **generalizing to unseen questions is
the real open challenge**, accuracy, deployment, and explainability must be weighed together.

---

## 7 · System, a live teacher-facing prototype

The research contribution is **realized as a working tool**, not just a benchmark:

| | |
|---|---|
| **Input** | teacher types the **Question + Reference answer + Student answer** |
| **Model selector** | choose any of the **four pillars** (classical · RNN · Transformer · LLM) |
| **Output** | a **score** + a **word-attribution explanation** (text-highlight heatmap) + **written feedback** |
| **Status** | **hosted live**, [demo URL to insert] |

- Closes the loop from the thesis: the **same faithfulness-checked word attribution** (occlusion over
  Khmer word units) is what the teacher sees, so the XAI study is *the* feature, not an add-on.
- Positioned as **teacher-assist** (human-in-the-loop), consistent with deployed feedback systems
  in the literature, *not* an autonomous grader.

<span class="small">[Insert 1–2 screenshots of the live app: the input form + a returned score with the Khmer word-attribution heatmap and feedback.]</span>

---

## Thank you · Questions?

<br><br>

**Contact:** [your email]
**Repository:** github.com/PhorkNorak/kxs
**Full leaderboards, champion predictions & training curves:** in the repository

<br>

### Appendix slides follow

- A. Ensemble result (why it didn't beat the best single cell)
- B. Hyperparameter inventory
- C. Compute budget
- D. XAI, example Khmer word-importance heatmaps & protocol

---

# Appendix A · Ensemble (v07), a negative result

We ensembled the **top-3 cells by validation QWK** (softmax-weighted average + recalibration):

| Dataset | Top-3 pool | Ensemble vs best single cell |
|---|---|---|
| 1184 | 3× GTE dual variants | no gain |
| 909 | 3× dual encoders | no gain |
| 895 | 3× TF-IDF SVR (different prep) | no gain |

The ensemble **did not beat the best single cell on any dataset**, the top candidates were
**too similar** (same family), so weighted averaging had no leverage.

> **Diversity, not weighting, is the bottleneck.** A forced-diverse pool
> {classical + encoder + LLM} is the promising direction (future work §6.4).

---

# Appendix B · Hyperparameter inventory

```
SEED            = 42
SPLIT           = 0.70 / 0.15 / 0.15  (stratified by score label)
TF-IDF          = char_wb 2–4-gram, max_features = 15000
SVR             = RBF, C = 1.0
BiLSTM          = hidden 128, 2 layers, embed 128, dropout 0.3, lr 1e-3, batch 64
TRANSFORMER     = lr 2e-5, batch 16, max_len 256, freeze bottom 6 layers,
                  max 20 epochs, early-stop patience 4 on val QWK
LLM (Qwen 3.5 4B) = QLoRA r=16 α=16, lr 2e-4, batch 4 × grad_accum 4, 7 epochs,
                  completion-only loss; greedy decode → regex first integer → clip
MAX_SCORE_NORM  = 20.0   (denominator for the max-score feature)
PREPROCESS      = raw | clean | segment        INPUT = ra | qar
```

---

# Appendix C · Compute budget

| Pillar | Cells | Wall time (A40-equivalent) |
|---|---:|---:|
| Classical (v01, v03, v04) + post-hoc (v02, v07) | many | ~1.5 hr |
| RNN (v05) | 18 | ~75 min |
| Encoder transformers (v06) | 108 | ~10 hr |
| Max-score neural (v03b) | 126 | ~12 hr |
| LLM (v08, Qwen, 7 epochs) | 1 | ~3 hr / cell |
| **Total** | | **~40 GPU-hours** |

Classical + ensemble together cost **< 2 hr**; nearly all compute went to the neural + LLM
pillars, reinforcing that the cheapest pillar is also the QWK champion.

---

# Appendix D · XAI protocol & example heatmaps

**Protocol (one rule for all families).** Explain the *answer* at the **Khmer word** level on the
same 895 test items. **LOO occlusion** (one unified method for every family)
→ rank words → measure **comprehensiveness** & **sufficiency** (ERASER) against a **random-word**
baseline, plus **reference-overlap** plausibility. Implemented in `xai/` + `experiments/exp09_xai.py`;
results in `results_xai/`.

> **Note on what you see.** Heatmaps display the **original** answer (for teacher readability),
> so punctuation like `។` / `?` appears as tiles, but the **model input is punctuation- and
> invisible-stripped**, so those tokens consistently receive **~0 importance** (light/white).
> That is correct behaviour, not a bug.

**Example word-importance heatmaps**, use the **HTML** versions (browser-shaped, correct Khmer).
matplotlib/PNG **cannot shape Khmer** (it won't stack subscripts or reorder vowels), so the `.png`
heatmaps look broken even though the data is correct; open the `.html` and screenshot for slides:

```
results_xai/no10c_no0/heatmaps/classical/classical_gallery.html   ← LOO occlusion
results_xai/no10c_no0/heatmaps/bilstm/bilstm_gallery.html         ← LOO occlusion
results_xai/no10c_no0/heatmaps/encoder/encoder_gallery.html       ← LOO occlusion
results_xai/no10c_no0/heatmaps/llm/llm_gallery.html               ← LOO occlusion
```

**Faithfulness (n=135, seed 42), single top-20% and AOPC over k∈{10..50}%:**

| Explainer | Comp | Suff | gap vs random | AOPC-comp | AOPC-suff |
|---|---:|---:|---:|---:|---:|
| Classical · occlusion | 0.125 | 0.030 | +0.096 | +0.150 | 0.011 |
| BiLSTM · occlusion | 0.270 | 0.120 | +0.257 | **+0.289** | 0.064 |
| Encoder · occlusion | 0.162 | 0.033 | +0.135 | +0.193 | 0.007 |
| LLM · occlusion | 0.083 | 0.420 | +0.047 | +0.126 | 0.359 |

LOO occlusion is **reliably faithful** for every pillar (gap > 0 for all four), strongest for the
BiLSTM (+0.289) and weakest but still positive for the LLM (+0.126); the LLM's higher sufficiency
shows the generative model's attributions are the least sharp.
Correct-Khmer heatmaps:
`results_xai/no10c_no0/heatmaps/{classical,bilstm}/*_gallery.html` (open in a browser); BiLSTM
train-vs-test curve at `results_xai/no10c_no0/curves/bilstm/train_history.png`.

> Render the deck with the VS Code **Marp** extension (or `marp docs/slides.md`). For Khmer
> example figures, screenshot the **HTML galleries** (browsers shape Khmer; matplotlib does not).


---

# End

<br><br>

Thank you.
