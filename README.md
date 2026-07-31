# Explainable Khmer Short-Answer Grading

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

An explainable automatic short-answer grading (ASAG) pipeline for Khmer educational text. The project
compares classical machine learning, recurrent neural networks, multilingual Transformer encoders, and
fine-tuned open-source large language models under one evaluation workflow. SHAP word attribution is
used to show which words influenced each predicted score.

This repository contains the research code and a Gradio prototype developed for the bachelor thesis
*Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education*.

## Highlights

- Four model families evaluated with the same scoring and preprocessing pipeline.
- Khmer-aware text cleaning and optional word segmentation with `khmer-nltk`.
- Reproducible 70/15/15 train, validation, and test split using seed 42.
- Ordinal evaluation with Quadratic Weighted Kappa (QWK), accuracy, macro-F1, and point-level agreement.
- One SHAP word-attribution workflow across all model families.
- A Gradio prototype that returns a predicted score, word highlights, and feedback.
- CPU baselines and GPU-ready Transformer and QLoRA experiment scripts.

## Verified results

The current uncalibrated champion results are:

| Model family | Representative model | Test size | QWK |
|---|---|---:|---:|
| Classical | TF-IDF + RBF-SVR | 137 | 0.802 |
| RNN | BiLSTM with attention | 137 | 0.789 |
| Transformer | GTE multilingual encoder | 178 | 0.777 |
| LLM | Qwen-KhmerGrader-4B, QLoRA fine-tuned | 137 | **0.850** |
| LLM baseline | Qwen 4B, zero-shot | 137 | 0.529 |

Qwen-KhmerGrader-4B matched the teacher's exact mark on 67% of the test answers and produced a mark
within one point on 80%. The Transformer result uses the full-corpus split, while the other champion
models use the curated `no10c` split. Cross-family differences should therefore be treated as
indicative rather than as a strict head-to-head comparison.

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
├── paper/                    paper source and figure-generation script
├── docs/                     ethics, model cards, and supporting documentation
├── data/README.md            required local dataset schema
├── run_pipeline.sh           full resumable research pipeline
└── requirements.txt          Python dependencies
```

The thesis source, student data, generated predictions, model weights, logs, and document exports are
intentionally excluded from the public repository.

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

The research corpus contains real answers written by school students and is not included in this
public repository pending confirmation of its release and reuse terms. Do not publish the CSV files
without the required consent and privacy review.

To run the project with an approved copy of the data, place these files locally:

```text
data/dataset.csv
data/dataset_no_10c_biology.csv
```

The required columns and validation guidance are documented in [`data/README.md`](data/README.md).
Both files are ignored by Git.

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

See [`docs/ethics.md`](docs/ethics.md) for the full data-governance statement and
[`docs/model_cards.md`](docs/model_cards.md) for model lineage and limitations.

## Citation

If you use this code, please cite the project metadata in [`CITATION.cff`](CITATION.cff). GitHub can
also generate a citation from the repository's **Cite this repository** menu.

## License

The source code is released under the [MIT License](LICENSE). This license does not grant permission to
use the private student dataset, third-party model weights, or third-party research papers. Those items
remain subject to their own access and licensing terms.
