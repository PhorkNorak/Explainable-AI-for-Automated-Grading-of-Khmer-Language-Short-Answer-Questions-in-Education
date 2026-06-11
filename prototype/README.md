# KhmerXScore — live prototype (Gradio)

A teacher-facing web app for the thesis *"Explainable AI for Automated Grading of Khmer
Language Short-Answer Questions."* The teacher pastes the **question + reference answer +
student answer**, picks a **model pillar**, and gets:

- a **score** (raw points + percentage + 5-class grade),
- a **Khmer word-attribution heatmap** (text highlighting of the words that drove the grade), and
- short **written feedback** (which reference points are missing).

It is a **teacher-assist** tool (human-in-the-loop), not an autonomous grader; the same
faithfulness-checked **word-attribution** explanation reported in the thesis is what the teacher sees.

## Run locally

From the repository root (`final_kxs/`):

```bash
pip install -r prototype/requirements.txt
python prototype/app.py            # opens http://127.0.0.1:7860
```

The **Classical** pillar trains on startup (CPU, a few seconds). The **RNN** pillar loads
the shipped champion checkpoint (`results/champions/rnn_clean_ra_bilstm_895/best.pt`) if
present. The **Transformer** and **LLM** pillars appear in the selector but need a GPU +
their weights, so on a CPU machine the app reports that clearly instead of failing.

Explanation: the app shows **word attribution** (text highlighting) computed by **occlusion**
(leave-one-out), the model-agnostic, faithfulness-checked highlighter used throughout the thesis.
It needs no extra install and works for every pillar, including the non-differentiable classical
model. For the neural pillars the same highlighting can also be produced from **attention** (BiLSTM)
or **Integrated Gradients** (Transformer); see the thesis explainability chapter.

## AI-generated feedback (open-source LLM)

The **AI feedback** checkbox generates short, Khmer, teacher-style feedback with an
**open-source LLM** over an **OpenAI-compatible** HTTP endpoint — so it works with Ollama, vLLM,
llama.cpp's server, LM Studio, OpenRouter, Together, etc. If no LLM server is reachable, the app
silently falls back to the built-in rule-based feedback (so it always works offline).

Easiest local setup is **[Ollama](https://ollama.com)** (free, runs the model locally):

```bash
ollama pull qwen2.5:7b-instruct      # a capable, Khmer-aware open model (or gemma2, llama3.1, ...)
ollama serve                          # exposes http://localhost:11434/v1
python prototype/app.py               # the app auto-targets Ollama by default
```

Configure via environment variables (all optional; defaults target local Ollama):

| Variable | Default | Notes |
|---|---|---|
| `FEEDBACK_LLM_BASE_URL` | `http://localhost:11434/v1` | Any OpenAI-compatible base URL |
| `FEEDBACK_LLM_MODEL` | `qwen2.5:7b-instruct` | Model name on that server |
| `FEEDBACK_LLM_API_KEY` | `ollama` | Real key only for hosted providers (OpenRouter/Together/…) |

Example pointing at a hosted open-model provider:

```bash
export FEEDBACK_LLM_BASE_URL="https://openrouter.ai/api/v1"
export FEEDBACK_LLM_MODEL="qwen/qwen-2.5-72b-instruct"
export FEEDBACK_LLM_API_KEY="sk-or-..."
```

The feedback is grounded only in the reference answer and is labelled AI-generated; it is a
teacher-assist suggestion to review, not a final comment.

## Deploy free on Hugging Face Spaces

The app imports the project code (`config`, `data`, `models/`, `xai/`) and reads
`data/*.csv` + `results/champions/`, so the **whole `final_kxs/` folder** is the Space.

1. Create a new **Gradio** Space.
2. Push the `final_kxs/` contents to it (the `*.pt` checkpoints are git-ignored by the
   project `.gitignore` — for the RNN pillar to work on the Space, force-add
   `results/champions/rnn_clean_ra_bilstm_895/best.pt`, e.g. `git add -f <that file>`).
3. Put this YAML front-matter at the **top of the Space's root `README.md`** so it picks up
   the right entry point and SDK:

   ```yaml
   ---
   title: KhmerXScore
   sdk: gradio
   app_file: prototype/app.py
   python_version: "3.10"
   ---
   ```

4. Set the Space's `requirements.txt` to `prototype/requirements.txt` (or copy its contents
   to a root `requirements.txt`).

For the Transformer/LLM pillars, use a **GPU** Space hardware tier and add
`transformers`, `accelerate`, `peft`, `bitsandbytes` (and `unsloth` for the LLM) — the same
extras listed in the project's top-level `requirements.txt`.

> After deploying, paste the public Space URL into the thesis (§4.5 / prototype slide).
