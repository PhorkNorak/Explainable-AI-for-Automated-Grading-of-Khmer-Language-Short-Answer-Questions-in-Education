"""KhmerXScore — live teacher-facing prototype (Gradio).

A teacher pastes the Question + Reference answer + Student answer, picks a model
pillar, and gets back:
  * a score (raw points + percentage + 5-class grade),
  * a Khmer word-attribution heatmap (text highlighting of the words that drove the grade), and
  * short written feedback (which reference points are missing).

The Classical pillar (TF-IDF + RBF-SVR) trains at startup and runs fully on CPU. The
RNN pillar loads the shipped champion checkpoint if present. The Transformer and LLM
pillars are wired into the selector but only activate where their weights + GPU are
available (otherwise the app reports that clearly).

Run locally:  python prototype/app.py      (opens http://127.0.0.1:7860)
Deploy:       Hugging Face Space (sdk: gradio, app_file: prototype/app.py) — see README.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr                                            # noqa: E402
import config as C                                            # noqa: E402
import data                                                   # noqa: E402
from preprocess import preprocess, strip_invisibles, strip_punctuation  # noqa: E402
from xai.attributions import word_importance                  # noqa: E402
from xai.render_html import heatmap_html_fragment             # noqa: E402

DATA_CSV = os.path.join(_ROOT, "data", "dataset_no_10c_biology.csv")


def readable_tokens(text):
    """Readable Khmer word units of the ORIGINAL text (for display + feedback).

    Strips invisibles and punctuation but does NOT apply KCC reordering, so the words
    render the way a teacher reads them (the model still scores the KCC-normalised text
    internally). Falls back to whitespace split if khmernltk is unavailable.
    """
    if not text:
        return []
    t = strip_punctuation(strip_invisibles(str(text)))
    try:
        import khmernltk
        return [w for w in khmernltk.word_tokenize(t) if w.strip()]
    except Exception:
        return [w for w in t.split() if w.strip()]


# ───────────────────────── model pillars ─────────────────────────
class ClassicalGrader:
    name = "Classical (TF-IDF + SVR)"
    preprocess_mode = "segment"

    def __init__(self):
        from models.classical import TFIDFSVR
        C.RAW_CSV = DATA_CSV; C.DROP_SCORE_ZERO = True; C.SPLIT_MODE = "random"
        df = data.load_dataframe(DATA_CSV)
        tr, _, _ = data.split_dataframe(df)
        trp = data.apply_preprocess(tr, self.preprocess_mode)
        a, b = data.build_text_lists(trp, "ra")
        self.model = TFIDFSVR().fit(a, b, trp["normalized_score"].values)

    def predict_one(self, ans_proc, ref_proc):
        return float(self.model.predict([ans_proc], [ref_proc])[0])

    def predict_batch(self, texts, ref_proc):
        return np.asarray(self.model.predict(list(texts), [ref_proc] * len(texts)), float)


class BiLSTMGrader:
    name = "RNN (BiLSTM + Attention)"
    preprocess_mode = "clean"

    def __init__(self):
        import torch
        from models.bilstm import BiLSTMScorer
        ckpt = os.path.join(_ROOT, "results", "champions",
                            "rnn_clean_ra_bilstm_895", "best.pt")
        state = torch.load(ckpt, map_location="cpu")
        vocab_size = int(state["emb.weight"].shape[0])
        C.RAW_CSV = DATA_CSV; C.DROP_SCORE_ZERO = True; C.SPLIT_MODE = "random"
        df = data.load_dataframe(DATA_CSV)
        tr, _, _ = data.split_dataframe(df)
        trp = data.apply_preprocess(tr, self.preprocess_mode)
        a, b = data.build_text_lists(trp, "ra")
        char2id = data.build_char_vocab(a + b)
        if len(char2id) != vocab_size:        # mapping would be misaligned → refuse
            raise RuntimeError(f"vocab size {len(char2id)} != checkpoint {vocab_size}")
        model = BiLSTMScorer(vocab_size=vocab_size)
        model.load_state_dict(state)
        model.train(False)
        self.torch, self.model, self.char2id, self.max_len = torch, model, char2id, 256

    def _encode(self, text):
        ids = [self.char2id.get(c, 1) for c in text[: self.max_len]]
        n = len(ids); pad = self.max_len - n
        return (self.torch.tensor(ids + [0] * pad).unsqueeze(0),
                self.torch.tensor([1] * n + [0] * pad).unsqueeze(0))

    def predict_one(self, ans_proc, ref_proc):
        ia, ma = self._encode(ans_proc); ib, mb = self._encode(ref_proc)
        with self.torch.no_grad():
            return float(self.model(ia, ma, ib, mb).item())

    def predict_batch(self, texts, ref_proc):
        return np.asarray([self.predict_one(t, ref_proc) for t in texts], float)


# Pillars that need GPU + downloaded/fine-tuned weights: wired into the selector but
# not loadable in a lightweight CPU environment.
GPU_PILLARS = {
    "Transformer (mBERT / XLM-R / GTE)": "the fine-tuned multilingual encoder weights",
    "LLM (Qwen 3.5 4B, QLoRA)": "the QLoRA adapter + 4-bit base model",
}


def _load_models():
    available, notes = {}, []
    try:
        available["Classical (TF-IDF + SVR)"] = ClassicalGrader()
        notes.append("Classical: ready (trained on startup, CPU).")
    except Exception as e:  # pragma: no cover
        notes.append(f"Classical: failed to load ({type(e).__name__}).")
    try:
        available["RNN (BiLSTM + Attention)"] = BiLSTMGrader()
        notes.append("RNN: ready (champion checkpoint loaded, CPU).")
    except Exception as e:
        notes.append(f"RNN: unavailable ({type(e).__name__}); checkpoint/vocab not loadable here.")
    for name, what in GPU_PILLARS.items():
        notes.append(f"{name}: requires {what} (GPU) — not loaded in this environment.")
    return available, notes


MODELS, LOAD_NOTES = _load_models()
ALL_CHOICES = (list(MODELS.keys())
               + [n for n in GPU_PILLARS if n not in MODELS])
DEFAULT_MODEL = "Classical (TF-IDF + SVR)" if "Classical (TF-IDF + SVR)" in MODELS else (
    ALL_CHOICES[0] if ALL_CHOICES else "")


# ───────────────────────── grading + feedback ─────────────────────────
def _feedback(yhat, answer, reference):
    if yhat >= 0.85:
        band = "Excellent — the answer covers the key content of the reference."
    elif yhat >= 0.60:
        band = "Good — most of the key content is present."
    elif yhat >= 0.40:
        band = "Partial — some key content is present, but important points are missing."
    else:
        band = "Needs improvement — the key content of the reference is largely missing."
    ans_set = set(readable_tokens(answer))
    seen, missing = set(), []
    for w in readable_tokens(reference):           # readable, original-order keywords
        if len(w) > 1 and w not in ans_set and w not in seen:
            seen.add(w); missing.append(w)
    tip = ("\n\n**Reference points not found in the answer:** " + "，".join(missing[:8])
           if missing else "\n\nThe answer touches the main reference points.")
    return (f"**Feedback.** {band}{tip}\n\n"
            f"*Teacher-assist only — please review the highlighted words and the suggestion "
            f"before finalizing the grade.*")


# ── AI feedback via an OPEN-SOURCE LLM (OpenAI-compatible endpoint) ──
# Works with Ollama (default), vLLM, llama.cpp server, LM Studio, OpenRouter, Together, etc.
# Configure via env vars; if no server is reachable the app falls back to the rule-based feedback.
FEEDBACK_LLM_BASE_URL = os.environ.get("FEEDBACK_LLM_BASE_URL", "http://localhost:11434/v1")
FEEDBACK_LLM_MODEL = os.environ.get("FEEDBACK_LLM_MODEL", "qwen2.5:7b-instruct")
FEEDBACK_LLM_API_KEY = os.environ.get("FEEDBACK_LLM_API_KEY", "ollama")  # Ollama ignores it

_FEEDBACK_SYS = (
    "You are an experienced, encouraging Khmer secondary-school teacher. Write SHORT feedback "
    "(3-4 sentences) IN KHMER to the student about their short answer, grounded ONLY in the "
    "reference answer provided. Say what they got right, what key point is missing or wrong, and "
    "one concrete suggestion to improve. Be kind and specific; do not invent facts beyond the "
    "reference; do not restate the numeric score."
)


def _ai_feedback(question, reference, answer, raw, max_score):
    """Khmer teacher-style feedback from an open-source LLM. Returns None on any failure
    (no `requests` installed, no LLM server reachable, bad response, ...)."""
    user = (f"Question: {question}\n\nReference answer (correct): {reference}\n\n"
            f"Student answer: {answer}\n\nTeacher score: {raw}/{max_score}.\n\n"
            "Write the feedback in Khmer now.")
    try:
        import requests
        r = requests.post(
            f"{FEEDBACK_LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {FEEDBACK_LLM_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": FEEDBACK_LLM_MODEL,
                  "messages": [{"role": "system", "content": _FEEDBACK_SYS},
                               {"role": "user", "content": user}],
                  "temperature": 0.3, "max_tokens": 320, "stream": False},
            timeout=60,
        )
        r.raise_for_status()
        txt = (r.json()["choices"][0]["message"]["content"] or "").strip()
        return txt or None
    except Exception:
        return None


def build_feedback(use_ai, question, reference, answer, raw, max_score, yhat):
    if use_ai:
        ai = _ai_feedback(question, reference, answer, raw, max_score)
        if ai:
            return (f"**Feedback (AI · {FEEDBACK_LLM_MODEL}).**\n\n{ai}\n\n"
                    f"<small><i>AI-generated, teacher-style — review before sharing with the "
                    f"student.</i></small>")
    return _feedback(yhat, answer, reference)


def grade(question, reference, answer, max_score, model_name, use_ai):
    if not (answer or "").strip() or not (reference or "").strip():
        return ("⚠️ Please enter both a **reference answer** and a **student answer**.", "", "")
    grader = MODELS.get(model_name)
    if grader is None:
        what = GPU_PILLARS.get(model_name, "additional weights")
        return (f"**{model_name}** is wired into the system but needs {what} (a GPU), so it is "
                f"not active in this lightweight demo. Please pick the **Classical** or **RNN** "
                f"pillar here, or run this pillar on a GPU deployment.", "", "")

    mode = grader.preprocess_mode
    ans_proc, ref_proc = preprocess(answer, mode), preprocess(reference, mode)
    yhat = float(np.clip(grader.predict_one(ans_proc, ref_proc), 0.0, 1.0))
    try:
        max_score = int(max_score)
    except (TypeError, ValueError):
        max_score = 10
    max_score = max(1, max_score)
    raw = int(round(yhat * max_score))
    grade5 = int(round(yhat * 4))

    score_md = (f"### Score: **{raw} / {max_score}**  &nbsp; "
                f"(model estimate {yhat * 100:.0f}%, grade {grade5}/4)\n\n"
                f"Model: **{grader.name}**\n\n"
                f"<small><i>This is the model's <b>predicted grade</b> — a learned estimate of how the "
                f"teacher would score this content, not an exact-match check. Even a verbatim copy of "
                f"the reference may not read 100%, because the model imitates the teacher's grading "
                f"rather than measuring text overlap.</i></small>")

    # Explanation: attribute over READABLE original word units (the model still scores
    # the KCC-normalised text via preprocess inside the predictors below).
    disp = readable_tokens(answer)
    if len(disp) < 2:
        return (score_md, "<i>Answer too short to explain.</i>",
                build_feedback(use_ai, question, reference, answer, raw, max_score, yhat))
    ans_disp = " ".join(disp)

    def pred_one(a_disp, b=ref_proc):
        return grader.predict_one(preprocess(a_disp, mode), b)

    def pred_batch(texts, b=ref_proc):
        return np.asarray([grader.predict_one(preprocess(t, mode), b) for t in texts], float)

    # LOO (Leave-One-Out) is the sole word attribution method — model-agnostic and
    # faithfulness-verified across all four model families (ERASER comprehensiveness/sufficiency).
    words, imp = word_importance(pred_one, ans_disp, ref_proc, "segment")
    heat = heatmap_html_fragment(words, imp, f"Word attribution (highlighted words) — {grader.name}")
    return score_md, heat, build_feedback(use_ai, question, reference, answer, raw, max_score, yhat)


# ───────────────────────── example from the real data ─────────────────────────
def _example_row():
    try:
        df = data.load_dataframe(DATA_CSV)
        r = df.iloc[0]
        return [str(r["Question"]), str(r["Reference"]), str(r["Answer"]), int(r["Max Score"])]
    except Exception:
        return ["", "", "", 10]


# ───────────────────────── UI ─────────────────────────
def build_demo():
    with gr.Blocks(title="KhmerXScore — Explainable Khmer Short-Answer Grading") as demo:
        gr.Markdown(
            "# KhmerXScore — Explainable Grading of Khmer Short Answers\n"
            "Enter the **question**, the **reference answer**, and the **student answer**, "
            "choose a model pillar, then press **Grade**. You get a score, a Khmer "
            "**word-attribution** heatmap (the highlighted words that drove the grade), and "
            "short feedback. *Teacher-assist tool — "
            "the teacher stays in the loop.*")
        with gr.Row():
            with gr.Column(scale=1):
                question = gr.Textbox(label="Question (optional)", lines=2)
                reference = gr.Textbox(label="Reference answer", lines=4)
                answer = gr.Textbox(label="Student answer", lines=4)
                with gr.Row():
                    max_score = gr.Number(label="Max score", value=10, precision=0, minimum=1)
                    model_name = gr.Dropdown(ALL_CHOICES, value=DEFAULT_MODEL, label="Model pillar")
                use_ai = gr.Checkbox(value=True, label="AI feedback (open-source LLM; falls back "
                                     "to rule-based if no LLM server)")
                btn = gr.Button("Grade", variant="primary")
            with gr.Column(scale=1):
                score_out = gr.Markdown(label="Score")
                heat_out = gr.HTML(label="Why this score (word importance)")
                fb_out = gr.Markdown(label="Feedback")
        gr.Examples(examples=[_example_row()],
                    inputs=[question, reference, answer, max_score],
                    label="Example (from the dataset)")
        gr.Markdown("<small>" + "  ·  ".join(LOAD_NOTES) + "</small>")
        btn.click(grade,
                  inputs=[question, reference, answer, max_score, model_name, use_ai],
                  outputs=[score_out, heat_out, fb_out])
    return demo


if __name__ == "__main__":
    # theme passed here (Gradio 6 moved it out of the Blocks constructor)
    build_demo().launch(theme=gr.themes.Soft())
