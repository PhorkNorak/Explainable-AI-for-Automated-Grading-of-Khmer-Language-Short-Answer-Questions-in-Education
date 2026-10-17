> Audit status: handler smoke tests passed without startup model loading. Historical checkpoint provenance, full browser behaviour and GPU inference remain unverified. SHAP uses a seeded permutation estimator with a hard model-call budget (`PROTOTYPE_SHAP_MAX_EVALS`, default 100); short budgets can report explanation unavailable. See the [historical audit](../docs/guides/report_evidence.md).

# KhmerXScore: live prototype (Gradio)

A web app developed for Phork Norak's undergraduate report, *"Explainable AI for
Automated Grading of Khmer Language Short-Answer Questions in Education,"* under the supervision of
Dr. Khim Chamroeun. The teacher pastes the **question + reference answer +
student answer**, picks a **model pillar**, and gets:

- a **score** (raw points + percentage + 5-class grade),
- a **Khmer word-attribution heatmap** (text highlighting of the words that drove the grade), and
- short **written feedback** (which reference points are missing).

The [June 2026 final report](<../docs/reports/Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education.pdf>) calls the LLM family **KhmerGrader**. This guide describes the final-report system and its configurable backends. Report SHAP values match the saved summary after rounding; provenance limits are documented in the [evidence reference](../docs/guides/report_evidence.md).

It is a **teacher-assist** tool (human-in-the-loop), not an autonomous grader; it implements **SHAP word attribution**, the explanation approach described in the report. Historical report heatmaps are not evidence that the current estimator outputs have been reproduced.

## The four pillars and how each activates

All four pillars are wired behind one interface. Each one self-activates when its resource is
available; otherwise it stays in the selector and reports exactly why it is inactive (the status
line at the bottom of the app lists the state of every pillar).

| Pillar | Runs on | Activates when |
|---|---|---|
| **Classical (TF-IDF + SVR)** | CPU | requires the private corpus and working dependencies; refits at startup. **Demo pillar.** |
| **RNN (BiLSTM + Attention)** | CPU | the champion checkpoint `local/checkpoints/results/champions/rnn_clean_ra_bilstm_909/best.pt` is present |
| **Transformer (GTE dual-encoder)** | CPU (slow) / GPU | a fine-tuned checkpoint is present at `local/checkpoints/results/champions/encoder_clean_qar_dual_gte_maxfeat_1184/best.pt` (or `ENCODER_CKPT`) **and** `transformers` is installed |
| **LLM (Qwen-KhmerGrader-4B)** | endpoint (any) / GPU | `GRADER_LLM_BASE_URL` is set (endpoint), or `GRADER_LLM_INPROCESS=1` on a GPU |

> The encoder checkpoint and the LLM adapter are trained on HPC/GPU and are **not shipped** with the
> repo. The code is ready to consume them: drop the encoder `best.pt` into the champion dir, or point
> the LLM pillar at a served KhmerGrader. Until then those two pillars show a clear "inactive" note
> the Classical pillar also requires locally downloaded dataset files and compatible dependencies; see the [dataset card](../docs/guides/dataset.md).

## Run locally

From the repository root (`final_kxs/`), use a dedicated environment. On Windows PowerShell:

The previous environment was moved into `local/environments/prototype/` for preservation. Virtual environments can retain absolute interpreter paths, so recreate the environment at this location before using its activation script.

```powershell
python -m venv local/environments/prototype
.\local\environments\prototype\Scripts\Activate.ps1
python -m pip install -r prototype/requirements.txt
python prototype/app.py
```

On a Unix shell, activate an equivalent environment before running:

```bash
pip install -r prototype/requirements.txt
python prototype/app.py            # opens http://127.0.0.1:7860
```

The base install supports CPU code. Actual model availability depends on the private corpus, checkpoint compatibility, and any explicitly configured endpoint. The **Transformer** and **in-process LLM** paths need the optional extras
noted in `prototype/requirements.txt`.

Explanation: the app shows **SHAP word attribution** (text highlighting), the model-agnostic
highlighter described in the final report. The repaired code uses one seeded
Monte Carlo permutation estimator with a hard model-call budget per invocation. It does not
switch silently to another estimator. Historical heatmaps require a rerun. This method supports
all four pillars, including the non-differentiable
classical model. (For a `qar` pillar the question is folded into the model input, so the attribution
still perturbs only the student's answer words.) The explanation is a **toggle** ("Show explanation"):
turn it off for a fast score-only result, or on for the heatmap, which is slower, especially for the
LLM pillar where each SHAP evaluation is a full generation.

Written feedback comes from an **open-source instruct LLM** over an OpenAI-compatible endpoint
(`FEEDBACK_LLM_BASE_URL`), never a proprietary API; when no feedback server is reachable the app falls
back to deterministic **rule-based** feedback (the reference points missing from the answer), when its local text-processing dependencies are available.

## LLM grading pillar (open-source LLM)

The **LLM (Qwen-KhmerGrader-4B)** pillar grades with an open-source model. It has two backends:

**1. Endpoint (default, runs anywhere incl. this CPU machine).** Point it at any OpenAI-compatible
server (vLLM / Ollama / llama.cpp / LM Studio / OpenRouter / Together …) that serves a KhmerGrader
model. The app sends the same grading prompt used in fine-tuning and parses the integer score.

```bash
export GRADER_LLM_BASE_URL="http://localhost:8000/v1"   # your served KhmerGrader
export GRADER_LLM_MODEL="your-served-grader-model" # replace with your server model name
export GRADER_LLM_API_KEY="..."                          # only for hosted providers
python prototype/app.py
```

| Variable | Default | Notes |
|---|---|---|
| `GRADER_LLM_BASE_URL` | *(empty → pillar inactive)* | OpenAI-compatible base URL serving the grader |
| `GRADER_LLM_MODEL` | Set explicitly | Exact model name exposed by your server; the example is a placeholder |
| `GRADER_LLM_API_KEY` | `ollama` | real key only for hosted providers |

**2. In-process (GPU only).** Loads the base model + trained LoRA adapter locally instead of calling
an endpoint. Needs the optional GPU extras and a GPU Space.

```bash
export GRADER_LLM_INPROCESS=1
export GRADER_LLM_MODEL="Qwen/Qwen3.5-4B"                # base (HF id)
export GRADER_LLM_ADAPTER="/path/to/report-model-adapter"
python prototype/app.py
```

## Written feedback (open-source LLM)

The written feedback is generated by a **general open-source instruct model** (separate from the
KhmerGrader that produces the score), served over any OpenAI-compatible endpoint. Point it at a server
and the app drafts short Khmer feedback grounded in the question, reference, student answer, the score,
and the missing reference points. If no endpoint is set or reachable, it falls back to deterministic
rule-based feedback, when its text-processing dependencies are available. It never calls a proprietary grading API.

```bash
export FEEDBACK_LLM_BASE_URL="http://localhost:8000/v1"   # any OpenAI-compatible server
export FEEDBACK_LLM_MODEL="qwen2.5:7b-instruct"
export FEEDBACK_LLM_API_KEY="..."                          # only for hosted providers
python prototype/app.py
```

| Variable | Default | Notes |
|---|---|---|
| `FEEDBACK_LLM_BASE_URL` | *(empty → rule-based fallback)* | OpenAI-compatible base URL serving an instruct model |
| `FEEDBACK_LLM_MODEL` | `qwen2.5:7b-instruct` | model name on that server |
| `FEEDBACK_LLM_API_KEY` | `ollama` | real key only for hosted providers |

## Deployment packaging

The app imports project modules and depends on locally downloaded data or compatible model resources. A source-only checkout does not provide a fully populated demo. Do not upload the whole working directory or force-add private data/checkpoints as a deployment shortcut.

For a separately approved deployment, package the app and required source modules, use `prototype/app.py` as the Gradio entry point and install its requirements. Provision only resources whose access and distribution permissions have been confirmed. Configure `GRADER_LLM_*` and `FEEDBACK_LLM_*` for the permitted open-source models. The encoder and in-process LLM need the optional dependencies documented in `prototype/requirements.txt`; GPU work belongs on HPC or the deployment host.

No public demo URL is verified here. Existing binary report/presentation exports are unchanged. See [data access and ethics](../docs/guides/ethics.md).
