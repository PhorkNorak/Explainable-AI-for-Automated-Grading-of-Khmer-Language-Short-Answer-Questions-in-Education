# Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

**Author:** Phork Norak<br>
**Supervisor:** Dr. Khim Chamroeun<br>
**Academic work:** Undergraduate report

This is a student research project. Feedback, corrections, and suggestions from researchers,
educators, and Khmer-language specialists are welcome through GitHub issues.

## Abstract

Automatic short-answer grading (ASAG) can reduce grading workload and provide students with faster
feedback, but most existing systems are designed for high-resource languages. Khmer remains a
low-resource language in natural language processing, and there is no established benchmark for
grading Khmer short answers. This study addresses that gap by developing a reproducible and
explainable Khmer ASAG benchmark from 1,184 teacher-graded answers written by 203 students from two
Cambodian high schools, one in a province and one in a city. The corpus covers 41 questions from
Biology, History, Geography, and Earth Science. After identifying and removing an inconsistently
graded Biology subset, 909 answers were used in the primary experiments.

The study compares four model families within a unified grading pipeline: a classical TF-IDF and
support-vector regression model, a BiLSTM with attention, multilingual Transformer encoders, and the
Pintu (ពិន្ទុ, “score” in Khmer) family of open-source large language models, consisting of Qwen 3.5 4B, Gemma 4 E4B, and
Gemma-SEA-LION v4.5 E2B, fine-tuned using QLoRA. The models were evaluated using Quadratic Weighted
Kappa (QWK), classification metrics, and exact and within-one-point agreement with the teacher's
marks. Four recent frontier language models, GPT-5.5, Claude Opus 4.8, Gemini 3.5 Flash, and DeepSeek
V4, were additionally evaluated using zero-shot prompting. To support transparent grading, SHAP word
attribution was used to highlight the answer words that influenced each model's prediction.

The fine-tuned Pintu-Qwen3.5-4B achieved the strongest overall performance, with a QWK of 0.850,
compared with 0.802 for the classical model, 0.789 for the BiLSTM, and 0.777 for the Transformer
encoder. It reproduced the teacher's exact mark for 67% of the test answers and produced a mark
within one point for 80%. It also outperformed the best zero-shot frontier model, Claude Opus 4.8,
which achieved a QWK of 0.734. Across the four model families, SHAP explanations achieved
reference-overlap plausibility scores ranging from approximately 0.60 to 0.68.

The results indicate that fine-tuning a relatively small open-source model on in-domain Khmer data
can be more effective than relying on much larger general-purpose models. The study also delivers a
prototype that returns a predicted score, highlighted word contributions, and written feedback.
Nevertheless, the findings measure agreement with a single teacher on a limited corpus and require
validation with additional graders, schools, and subjects.

**Keywords:** Automatic Short-Answer Grading; Explainable AI; SHAP; Khmer Language Processing;
Low-Resource NLP; Large Language Models; QLoRA Fine-Tuning; Quadratic Weighted Kappa.

## Highlights

- Four model families evaluated with the same scoring and preprocessing pipeline.
- Khmer-aware text cleaning and optional word segmentation with `khmer-nltk`.
- Reproducible 70/15/15 train, validation, and test split using seed 42.
- Ordinal evaluation with Quadratic Weighted Kappa (QWK), accuracy, macro-F1, and point-level agreement.
- One SHAP word-attribution workflow across all model families.
- A Gradio prototype that returns a predicted score, word highlights, and feedback.
- CPU baselines and GPU-ready Transformer and QLoRA experiment scripts.

## Results

All values below are the uncalibrated test results reported in the final report. QWK is Quadratic
Weighted Kappa, Acc is accuracy, F1 is macro-F1, Exact is exact integer-mark agreement, and ±1 is
agreement within one point.

### Main model comparison

| Model | QWK | Acc | F1 | Exact | ±1 |
|---|---:|---:|---:|---:|---:|
| Classical, TF-IDF + RBF-SVR | 0.802 | 0.65 | 0.47 | 0.28 | 0.72 |
| RNN, BiLSTM with attention | 0.789 | 0.69 | 0.53 | 0.34 | 0.70 |
| Transformer, GTE multilingual encoder | 0.777 | 0.68 | 0.51 | 0.49 | 0.66 |
| **LLM, Pintu-Qwen3.5-4B fine-tuned** | **0.850** | **0.81** | **0.80** | **0.67** | **0.80** |
| Qwen base, zero-shot reference | 0.529 | 0.42 | 0.30 | 0.25 | 0.43 |

The fine-tuned Pintu-Qwen3.5-4B achieved the best result on every reported grading metric. Its QWK
increased from 0.529 without fine-tuning to 0.850 after fine-tuning, a gain of approximately 0.32.
The Transformer champion was evaluated on the corpus before the 10C Biology subset was removed, so its
cross-family comparison is indicative rather than a strict head-to-head result.

### Pintu model family

| Fine-tuned model | Base size | QWK |
|---|---:|---:|
| **Pintu-Qwen3.5-4B** | 4B | **0.850** |
| Pintu-Gemma4-E4B | 4B | 0.844 |
| Pintu-SEA-LION-v4.5-E2B | 2.3B | 0.713 |

The two 4B models produced similar QWK results, while the smaller SEA-LION model scored lower. All
three models were fine-tuned using QLoRA on the Khmer grading data.

### Fine-tuned Pintu versus frontier models

The following comparison uses the same curated test set of 137 answers. Frontier models were prompted
zero-shot using both a bare-integer setting and a reasoning-enabled setting. The results are a
version-pinned API snapshot evaluated on 2026-06-24.

| Model | Setting | QWK | Acc | F1 | Exact | ±1 |
|---|---|---:|---:|---:|---:|---:|
| **Pintu-Qwen3.5-4B** | **fine-tuned** | **0.850** | **0.81** | **0.80** | **0.67** | **0.80** |
| GPT-5.5 | bare | 0.684 | 0.59 | 0.53 | 0.48 | 0.67 |
| GPT-5.5 | reasoning | 0.708 | 0.58 | 0.50 | 0.45 | 0.65 |
| Claude Opus 4.8 | bare | 0.662 | 0.54 | 0.37 | 0.35 | 0.63 |
| Claude Opus 4.8 | reasoning | 0.734 | 0.62 | 0.61 | 0.43 | 0.69 |
| Gemini 3.5 Flash | bare | 0.555 | 0.60 | 0.45 | 0.46 | 0.61 |
| Gemini 3.5 Flash | reasoning | 0.585 | 0.58 | 0.42 | 0.45 | 0.68 |
| DeepSeek V4 | bare | 0.553 | 0.48 | 0.38 | 0.34 | 0.52 |
| DeepSeek V4 | reasoning | 0.579 | 0.49 | 0.37 | 0.31 | 0.56 |

Pinned versions: `gpt-5.5-20260423`, `claude-opus-4.8`, `gemini-3.5-flash-20260519`, and
`deepseek-v4-flash-20260423`. Frontier metrics use each model's successfully parsed responses.

Pintu-Qwen3.5-4B outperformed every tested frontier model. The strongest frontier result was Claude
Opus 4.8 with reasoning at QWK 0.734. The fine-tuned 4B Pintu led it by 0.116 QWK and also achieved
higher accuracy, macro-F1, exact agreement, and within-one-point agreement.

### Explainability results with different evaluation sample sizes

SHAP word attribution was applied across all four model families. Plausibility is the fraction of the
top 20% of highlighted answer words that also appear in the reference answer.

| Model | SHAP plausibility | Evaluated answers |
|---|---:|---:|
| Classical, SVR | 0.63 | 66 |
| RNN, BiLSTM | 0.60 | 66 |
| Transformer, GTE | 0.68 | 18 |
| LLM, Qwen | 0.61 | 18 |

**Comparison note:** Classical and BiLSTM were evaluated on 66 answers, while Transformer and LLM were
evaluated on capped samples of 18 answers. These results demonstrate that SHAP can produce plausible
word-level explanations across all four model families. They should not be used to rank explanation
quality because the evaluation sample sizes are different.

### Other findings

- Removing the inconsistently graded 10C Biology subset increased the classical model's QWK by about
  0.044.
- A smaller text-cleaning refinement changed QWK by no more than 0.003.
- Adding the maximum possible score as an input feature improved exact agreement for the neural models
  by about 4.5 percentage points on the 909-answer dataset.
- Post-hoc calibration was model-dependent and is reported only as an ablation, not as a headline result.

## Repository structure

```text
.
├── config.py                 experiment paths, model registry, and hyperparameters
├── data.py                   dataset loading and deterministic splitting
├── preprocess.py             Khmer text cleaning and segmentation
├── train.py                  shared model-training functions
├── evaluate.py               agreement and point-level metrics
├── models/                   classical, BiLSTM, dual-encoder, and cross-encoder models
├── xai/                      SHAP attribution, plausibility, and Khmer rendering
├── experiments/              experiment and analysis entry points
├── prototype/                Gradio demonstration application
├── run_pipeline.sh           full resumable research pipeline
└── requirements.txt          Python dependencies
```

The undergraduate report, paper draft, supporting documents, student data, generated predictions, model weights,
logs, publication drafts, and document exports are intentionally excluded from the public repository.

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/PhorkNorak/Explainable-AI-for-Automated-Grading-of-Khmer-Language-Short-Answer-Questions-in-Education.git
cd Explainable-AI-for-Automated-Grading-of-Khmer-Language-Short-Answer-Questions-in-Education

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

For GPU experiments, install the PyTorch build that matches the CUDA version reported by
`nvidia-smi` before installing the remaining requirements. See the comments in
[`requirements.txt`](requirements.txt) for an example.

## Dataset setup

The research corpus contains real answers written by school students and is not stored in this GitHub
repository. A separate de-identified release package has been prepared under the CC BY-NC 4.0 dataset
licence. School identifiers in that package have been replaced, while the original pseudonymous
student and class codes are retained; the private source CSVs must not be published.

To run the project with an approved copy of the data, place these files locally:

```text
data/dataset.csv
data/dataset_no_10c_biology.csv
```

The CSV loader expects these columns: `SchoolID`, `ClassID`, `Subject`, `StudentID`, `QuestionID`,
`Question`, `Reference`, `Answer`, `Student Score`, `Max Score`, and `Year`. The complete `data/` folder
is ignored by Git.

## Quick start

Run the classical baseline on the curated dataset:

```bash
python experiments/exp01_tfidf_baseline.py --datasets no10c
```

Check experiment status:

```bash
python experiments/check_progress.py
```

Run the full resumable pipeline on a CUDA-capable machine:

```bash
bash run_pipeline.sh
```

The full pipeline includes classical, BiLSTM, Transformer, QLoRA, SHAP, ablation, and reporting steps.
It writes generated artifacts to ignored `results*` and `logs` directories.

## Main experiment commands

```bash
# Classical models
python experiments/exp01_tfidf_baseline.py --datasets no10c
python experiments/exp03_maxscore_feature.py --datasets no10c
python experiments/exp04_bucket_svr.py --datasets no10c

# Neural models
python experiments/exp05_bilstm.py --datasets no10c
python experiments/exp06_transformer.py --datasets no10c

# Open-source LLM fine-tuning
python experiments/exp08_llm_finetune.py --models qwen35_4b --epochs 10 --datasets no10c --input qar

# Merge the three tested QAR adapters into complete BF16 Pintu models
python experiments/merge_pintu_models.py --models qwen gemma sealion

# Explainability and summary metrics
python experiments/exp09_xai.py --families classical bilstm --dataset no10c
python experiments/exp10_significance.py
```

Transformer and LLM experiments require a CUDA GPU and access to their base models. The classical
pipeline runs on CPU. See [`experiments/README.md`](experiments/README.md) for the experiment registry.

## Prototype

Install and start the Gradio application:

```bash
pip install -r prototype/requirements.txt
python prototype/app.py
```

Then open <http://127.0.0.1:7860>. The application supports local or OpenAI-compatible endpoints for
open-source grading and feedback models, with a rule-based feedback fallback. Configuration details are
in [`prototype/README.md`](prototype/README.md). Never commit API keys or local model paths.

## Reproducibility

- Default random seed: `42`
- Default split: 70% training, 15% validation, 15% test
- Primary metric: Quadratic Weighted Kappa
- Model selection: validation QWK
- Headline results: uncalibrated test values
- Saved outputs: generated under `results*` and excluded from Git

Exact results depend on the approved dataset, hardware, package versions, and access to the listed base
models. Closed-model comparisons are optional dated baselines and are not required to run the core
open-source pipeline.

## Responsible use

This system is intended to assist teachers, not replace them in high-stakes grading. The reported scores
measure agreement with one teacher on a limited corpus. They do not establish universal correctness or
fairness across schools, subjects, or student groups. Any real deployment should include human review,
additional graders, privacy controls, and evaluation in the target classroom setting.

The dataset is governed by its separate CC BY-NC 4.0 licence rather than the software's MIT licence.
Any model trained from student answers still requires a privacy and memorization review before public
distribution.

## Citation

If you use this code, please cite the project metadata in [`CITATION.cff`](CITATION.cff). GitHub can
also generate a citation from the repository's **Cite this repository** menu.

## License

The source code is released under the [MIT License](LICENSE). This license does not grant permission to
use the private student dataset, third-party model weights, or third-party research papers. Those items
remain subject to their own access and licensing terms.
