---
marp: true
theme: default
paginate: true
size: 16:9
---

# Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education

**Phork Norak** | Supervisor: **Dr. Khim Chamroeun**

Royal University of Phnom Penh
Bachelor of Engineering in Data Science and Engineering | June 2026

[Final report](<../reports/Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education Report final.pdf>) | [Evidence notes](report_evidence.md)

---

## Three research questions

1. **RQ1: Which model family gives the best agreement with the human teacher?**
2. **RQ2: Which model is most useful for classroom deployment based on exact mark and within-one-point agreement?**
3. **RQ3: Can SHAP provide understandable word-level explanations across all model families?**

Results measure agreement with a single grader.

---

## Khmer corpus

**1,184 answers | 41 questions | 203 students | four subjects | two schools**

Biology, History, Geography and Earth Science.

- `full`: 1,184 answers.
- `no10c`: 909 answers, excluding class-10C Biology; grade 0 retained.
- Reported split: 70/15/15, seed 42.

Private student data; public release is not established.

---

## Four families, one grading task

| Pillar | Approach |
|---|---|
| Classical | Character TF-IDF + RBF-SVR |
| RNN | Character BiLSTM + additive attention |
| Transformer | Multilingual mBERT / XLM-R / GTE |
| LLM | KhmerGrader with QLoRA fine-tuning |

Normalised score -> five-class grade or question-specific raw mark.
SHAP supplies explanations for every family; attention is architecture.

---

## Reported agreement (RQ1)

| Pillar | Test n | QWK | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Classical | 137 | 0.802 | 0.650 | 0.469 |
| BiLSTM | 137 | 0.789 | 0.693 | 0.525 |
| GTE | 178 | 0.777 | 0.680 | 0.510 |
| Qwen-KhmerGrader-4B | 137 | **0.850** | 0.810 | 0.804 |

Report Table 10; local summary `local/results/results_stats/champion_metrics.csv`.
Encoder comparison is indicative because its test set differs.

---

## Deployment metrics (RQ2)

Qwen-KhmerGrader-4B, as reported:

- **0.67 exact raw-mark match** (67%).
- **0.80 within one point** (80%).

Report Tables 10 and 13. Exact agreement for Classical / BiLSTM / GTE is 0.28 / 0.34 / 0.49.

These metrics describe agreement, not demonstrated classroom readiness.

---

## KhmerGrader family

| Model | Fine-tuned QWK |
|---|---:|
| Qwen-KhmerGrader-4B | **0.850** |
| Gemma-KhmerGrader-4B | 0.844 |
| SEA-LION-KhmerGrader-E2B | 0.713 |

Report Table 12. Qwen zero-shot reference: 0.529 (Table 10).

---

## Reported frontier comparison

| Model, best reported setting | QWK | Parsed test n |
|---|---:|---:|
| Qwen-KhmerGrader-4B, fine-tuned | **0.850** | 137 |
| Claude Opus 4.8, reasoning | 0.734 | 137 |
| GPT-5.5, reasoning | 0.708 | 136 |
| Gemini 3.5 Flash, reasoning | 0.585 | 117 |
| DeepSeek V4, reasoning | 0.579 | 136 |

Report Table 13 and local summary `local/results/results_frontier/no10c/frontier_metrics.csv`.
Unparsed answers are excluded; denominators differ. No fresh API verification.

---

## SHAP word attribution (RQ3)

Highlight the answer words contributing to the predicted score.
Plausibility = fraction of the top 20% SHAP words found in the reference answer.

| Pillar | Report Table 14 | Answers |
|---|---:|---:|
| Classical | 0.63 | 66 |
| BiLSTM | 0.60 | 66 |
| Encoder | 0.68 | 18 |
| LLM | 0.61 | 18 |

[Saved XAI summary matches after rounding](report_evidence.md#shap-plausibility). Unequal samples limit ranking.

---

## Teacher-facing prototype

Question + reference answer + student answer + maximum score + model selection.

Returns a score, optional readable Khmer SHAP highlights and written feedback.

- Open-source instruct LLM for feedback.
- Rule-based fallback when no server is reachable.
- Model availability depends on data, checkpoints and configured endpoints.

Report Table 11: score + SHAP approximately 2 / 13 / 10 / 76 seconds for Classical / BiLSTM / GTE / LLM.

[Prototype instructions](../../prototype/README.md). Public demo URL unverified.

---

## Limitations and conclusion

The final report identifies Qwen-KhmerGrader as the highest QWK performer and presents SHAP as a shared explanation method.

- One grader; small corpus and uneven subject coverage.
- Encoder uses a different test set.
- SHAP overlap is an automatic proxy; sample sizes differ.
- No classroom field study.
- Saved prediction provenance and the source score of 19/15 remain unresolved.

Next research priorities: a second grader, broader data and teacher evaluation.

[Report-to-evidence reference](report_evidence.md) | [Reproduction limits](report_evidence.md#remaining-evidence-and-release-boundaries)
