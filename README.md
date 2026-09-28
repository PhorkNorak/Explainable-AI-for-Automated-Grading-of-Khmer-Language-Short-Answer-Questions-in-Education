# Khmer short-answer grading: audit status

This undergraduate research project implements four grading families and SHAP word attribution. The audit reproduced 628 saved test-prediction files, but did not validate a current four-champion selection or the historical SHAP summaries.

| Saved configuration | Test n | QWK | Exact raw | Within 1 raw | Raw MAE |
|---|---:|---:|---:|---:|---:|
| SVR segment/ra | 137 | 0.782 | 0.175 | 0.708 | 1.496 |
| BiLSTM clean/ra | 137 | 0.666 | 0.394 | 0.613 | 1.803 |
| GTE dual/max feature | 178 | 0.820 | 0.573 | 0.770 | 0.994 |
| Pintu-Qwen3.5-4B | 137 | 0.843 | 0.657 | 0.832 | 0.927 |

These are named saved configurations, not newly selected champions. Training/preprocessing provenance is UNVERIFIED; the encoder uses the full test set.

See [audit record](docs/audit/README.md), [exact evidence](docs/audit/verified_prediction_examples.md), and [HPC handoff](docs/audit/HPC.md). Original README instructions are preserved in the audit archive.

The corpus is private. No training, external student-data evaluation, model download, or publishing is part of the local audit. For a new HPC run, use the isolated handoff rather than resuming legacy directories.

Prototype: install `prototype/requirements.txt` in a compatible environment, then run `python prototype/app.py`. Missing models are reported. Configure feedback only with the permitted open-source model endpoint; absent/unreachable endpoints use rule-based feedback.

Report: `thesis/main.tex`, XeLaTeX required. Paper source is absent. Markdown slides and notes are revised; existing PDF/PPTX exports are stale. No commit or push was performed.

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


## Data and local verification

The student corpus and saved predictions are private. With approved access, the loader expects local `data/dataset.csv` and `data/dataset_no_10c_biology.csv` containing `SchoolID`, `ClassID`, `Subject`, `StudentID`, `QuestionID`, `Question`, `Reference`, `Answer`, `Student Score`, `Max Score`, and `Year`.

A surviving source record has score 19 with maximum 15. Training and affected derived results are blocked until its score or maximum is confirmed against the original examination record. Do not silently clip or remove it. Existing data is preserved. Use the audit commands in [the audit record](docs/audit/README.md) for lightweight verification, and [HPC handoff](docs/audit/HPC.md) only after this data decision is resolved.

## Repository guide

- `data.py`, `preprocess.py`, and `evaluate.py`: loading, Khmer cleaning, and scoring.
- `models/`, `train.py`, and `experiments/`: model definitions, training, and experiment registry.
- `xai/`: SHAP estimation, overlap plausibility, and readable Khmer highlights.
- `prototype/`: teacher-facing Gradio application and endpoint configuration.
- `thesis/main.tex`: XeLaTeX report; `docs/slide_final.md` and `docs/script.md`: presentation and notes.
- `docs/audit/`: evidence ledger, verification outputs, unresolved decisions, and isolated HPC instructions.

## Intended use

This is research software for teacher assistance. Scores measure agreement with recorded grading decisions. They do not establish independent correctness, fairness, or classroom benefit. Feedback must use the permitted open-source LLM arrangement with a rule-based fallback. Data access, release consent, and downstream model permissions remain separate from the source-code licence.

## License

The source code is released under the [MIT License](LICENSE). This license does not grant permission to
use the private student dataset, third-party model weights, or third-party research papers. Those items
remain subject to their own access and licensing terms.
