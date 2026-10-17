# Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education

Undergraduate final report by **Phork Norak**, supervised by **Dr. Khim Chamroeun**, Royal University of Phnom Penh. Bachelor of Engineering in Data Science and Engineering, June 2026.

This project studies automated short-answer grading (ASAG) using **real, teacher-graded Khmer examination answers**. The corpus was manually digitised from handwritten semester exam papers collected at two Cambodian high schools. It compares four model families and uses **SHAP word attribution (text highlighting)** to show which answer words contribute to a predicted score. The prototype combines a score, optional word highlighting and written feedback.

The study contributes a real Khmer short-answer corpus, a shared pipeline for comparing four grading families, and a teacher-assist prototype with SHAP explanations. Its motivation is the time and inconsistency involved in manual grading, together with the limited resources available for Khmer language processing.

Read the [final report](<docs/reports/Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education Report final.pdf>) or the [final presentation](docs/reports/Final_Presentation.pdf). The final report is the 67-page version with clearer wording and all reviewed corrections. [Report versions](docs/reports/README.md) explain how it was recovered and revised. Editable manuscript sources, older drafts and local audit builds stay local and are excluded from Git.

The benchmark tables below retain the report's findings. [Report-to-evidence notes](docs/guides/report_evidence.md) distinguish reported results, saved summaries and unresolved reproduction issues. This repository provides the source code and final documents; it does not include the private student dataset, predictions or model checkpoints.

## Research questions and approach

1. **RQ1: Which model family gives the best agreement with the human teacher?**
2. **RQ2: Which model is most useful for classroom deployment based on exact mark and within-one-point agreement?**
3. **RQ3: Can SHAP provide understandable word-level explanations across all model families?**

| Family | Approach |
|---|---|
| Classical | Character TF-IDF features with RBF-SVR |
| RNN | Character BiLSTM with additive attention; SHAP supplies explanations |
| Transformer | Multilingual mBERT, XLM-R and GTE encoders |
| LLM | KhmerGrader: QLoRA fine-tuning of Qwen, Gemma and SEA-LION bases |

**KhmerGrader** is the LLM family studied in the final report. See the [model descriptions and lineage](docs/guides/model_cards.md). Documentation covers this final-report version; subsequent experiments are outside its scope.

The pipeline normalises marks by the question maximum and maps predictions to five grades with `round(4 * predicted_score)` and to raw points with `round(predicted_score * max_score)`, bounded to valid ranges. SVR uses its regression objective, neural regression heads use MSE, and generative LLM fine-tuning supervises integer-score completions. Preprocessing strips invisible characters, normalises to NFC and removes punctuation; `segment` additionally uses khmernltk. There is no custom KCC reorder.

## Real examination data

### Collection and digitisation

The corpus comes from **handwritten semester 1 examination papers from the 2026 school year**. The researcher collected papers from two Cambodian high schools, one in a province and one in the capital city. These settings bring variation in student writing, answer quality and classroom context. The answers are authentic student work, with marks awarded by a teacher, rather than generated or crowd-sourced responses.

The collection process preserved the student answers and teacher marks, manually transcribed the Khmer handwriting into text, organised the question, reference, answer, awarded score and maximum score into structured records, and cleaned the text for the shared modelling pipeline. The original process diagram from the final presentation is reproduced below.

![Original final-presentation slide showing collection of handwritten papers, real student answers and marks, manual Khmer transcription, structured records and data preparation](docs/images/slides/real-exam-data-collection.png)

### Corpus and dataset variants

The final corpus contains **1,184 answers, 41 questions, 203 students, four subjects and two schools**. Subjects are Biology, History, Geography and Earth Science. Question maxima vary across 5, 6, 7, 8, 10, 12, 15 and 20 points.

| Variant | Answers | Description |
|---|---:|---|
| `full` | 1,184 | Full usable corpus |
| `no10c` | 909 | Excludes class-10C Biology, where inconsistent marks were identified |

Both variants retain grade 0. The reported split is 70/15/15 with seed 42. Classical, RNN and LLM headline rows use the 137-answer no10c test set; the encoder uses the 178-answer full test set.

### Reference answers and score normalisation

An official teacher-provided answer key was unavailable. For each question, the **highest-scoring student response serves as the reference answer**. This is a proxy: the best available response may still fall short of full marks. The reference should therefore be understood as student-derived content rather than a verified complete marking rubric.

Teacher marks are normalised by the question's maximum: `normalised_score = student_score / max_score`. Five-class labels use `round(4 * normalised_score)`, bounded to grades 0-4. This allows questions with different scoring scales to share one grading task.

### Real classroom variation and grading noise

The data includes identical answers receiving different marks, answers that repeat the question, and zero-score responses with no meaningful content. Biology, History and Geography account for most records, Earth Science has few, and grades are skewed toward high scores. These properties explain the use of two dataset variants and the reporting of macro-F1 alongside accuracy. Results measure agreement with a **single human grader** and can inherit that grader's inconsistencies.

Collection and reference construction are described in report Section 3.2 and the final presentation's dataset slides (PDF pages 8-12).

### Local data access

The student corpus and predictions are private. With approved access, the loader expects `local/data/dataset.csv` and `local/data/dataset_no_10c_biology.csv`, with `SchoolID`, `ClassID`, `Subject`, `StudentID`, `QuestionID`, `Question`, `Reference`, `Answer`, `Student Score`, `Max Score` and `Year`. The report's public-release wording does not establish permission or actual public availability. See [ethics and data access](docs/guides/ethics.md).

## Benchmarks from the final report

The following tables are reproduced directly from the latest report, including their original captions, values and formatting.

### Table 10: Per-pillar champion results

![Table 10 from the final report: per-pillar champion results and Qwen zero-shot reference](docs/images/report/table-10-pillar-benchmark.png)

### Table 12: KhmerGrader family

![Table 12 from the final report: KhmerGrader family fine-tuned test QWK](docs/images/report/table-12-khmergrader-family.png)

### Table 13: KhmerGrader and frontier models

![Table 13 from the final report: KhmerGrader versus frontier models in bare and reasoning settings](docs/images/report/table-13-frontier-benchmark.png)

The encoder uses a different test set from the other pillars; frontier metrics use each model's parsed subset. See [report-to-evidence notes](docs/guides/report_evidence.md) for saved-result provenance and the historical audit.

## Explainability and prototype

SHAP attributes the predicted score to answer words. Plausibility is the fraction of the top 20% SHAP words appearing in the reference answer, an automatic lexical-overlap proxy rather than a teacher evaluation.

Report Table 14 gives SHAP plausibility **0.63 / 0.60 / 0.68 / 0.61** for Classical / BiLSTM / Encoder / LLM. These match the saved XAI summary after rounding. Classical and BiLSTM use 66 answers each; encoder and LLM use 18 each, so the ranking is not a common-sample comparison. See [SHAP evidence](docs/guides/report_evidence.md#shap-plausibility).

Report Table 11 gives approximate score-plus-SHAP latency of **2 / 13 / 10 / 76 seconds per answer**, respectively. These are capped-explanation costs, not score-only timings; see [efficiency evidence](docs/guides/report_evidence.md#efficiency).

Run the Gradio prototype from the repository root:

```bash
pip install -r prototype/requirements.txt
python prototype/app.py
```

Model availability depends on private data, compatible checkpoints and configured endpoints. Written feedback uses an open-source instruct LLM through `FEEDBACK_LLM_*`, with rule-based fallback when no server is reachable. See [prototype setup](prototype/README.md) for isolated-environment instructions and model configuration.

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/PhorkNorak/Explainable-AI-for-Automated-Grading-of-Khmer-Language-Short-Answer-Questions-in-Education.git
cd Explainable-AI-for-Automated-Grading-of-Khmer-Language-Short-Answer-Questions-in-Education

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

For GPU experiments, install the PyTorch build that matches the CUDA version reported by
`nvidia-smi` before installing the remaining requirements. See the comments in
[`requirements.txt`](requirements.txt) for an example.



## Reproduction and evidence status

The [experiment registry](experiments/README.md) documents the implemented stages. Classical and individual BiLSTM work are CPU-feasible; encoder and LLM training belong on HPC. No new training or external API evaluation was performed to align this documentation.

A surviving source record has a score of 19 with maximum 15. Its score or maximum must be confirmed against the examination record before training or regenerating affected results. Do not silently clip or remove it. Follow the [evidence notes](docs/guides/report_evidence.md) and [isolated HPC handoff](docs/guides/hpc.md), preserving existing evidence. Detailed audit records remain local.

## Repository layout

```text
src/khmer_asag/   Core data loading, preprocessing, training, models and SHAP
experiments/     Experiment definitions and shared experiment helpers
prototype/       Gradio app and its requirements
scripts/         Pipeline runners, analysis and isolated HPC handoff
tests/           Scoring, SHAP, prototype, path and handoff checks
docs/guides/     Methods, model cards, evidence, ethics and presentation notes
docs/images/     README benchmark tables and collection diagram
docs/reports/    Final report and presentation PDFs
local/           Ignored data, results, checkpoints, logs and manuscripts
```

Install the package with `python -m pip install -e .` from the repository root after installing dependencies. Imports use `khmer_asag`, and direct experiment and prototype commands also work from a source checkout. Central paths are defined in `src/khmer_asag/paths.py`.

The private dataset belongs in `local/data/`. Existing run names are preserved under `local/results/`, while champion checkpoints are under `local/checkpoints/results/champions/`. Logs, builds, editable manuscripts and historical archives stay under `local/`. Moving these files does not release them or make historical saved paths current.

Public Git files include code, requirements, tests, active documentation, the four README images and the final PDFs. The whole `local/` folder, secrets, environments, model weights and LaTeX/bibliography files are ignored. See the [repository guide](docs/guides/repository_guide.md) for the exact inclusion policy.

- [Prototype setup](prototype/README.md)
- [Experiment registry](experiments/README.md)
- [Final report and presentation](docs/reports/README.md)
- [Evidence and reproduction limits](docs/guides/report_evidence.md)
- [HPC workflow](docs/guides/hpc.md)
- [Slides](docs/guides/slide_final.md), [speaker notes](docs/guides/script.md) and [defense questions](docs/guides/defense_qa.md)

## Intended use and licence

This is research software for teacher assistance. Results measure agreement with one grader, not independent correctness, fairness or demonstrated classroom benefit. There is no completed classroom field study. See [ethics and data access](docs/guides/ethics.md).

Source code is released under the [MIT License](LICENSE). This does not grant rights to the private student dataset, third-party weights or research papers; those retain their own access and licensing terms.
