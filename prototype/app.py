"""Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education.

A teacher pastes the Question + Reference answer + Student answer, picks a model
pillar, and gets back:
  * a score (raw points + percentage + 5-class grade),
  * a Khmer word-attribution heatmap (text highlighting of the words that drove the grade), and
  * short written feedback (which reference points are missing).

All four pillars are wired behind one interface and self-activate when their resource
is available, otherwise the app reports clearly why a pillar is inactive:
  * Classical (TF-IDF + RBF-SVR) trains at startup and runs fully on CPU. (Demo pillar.)
  * RNN (BiLSTM + attention) loads the shipped champion checkpoint, CPU.
  * Transformer (GTE dual-encoder) loads a fine-tuned checkpoint if present (CPU-capable).
  * LLM (Qwen-KhmerGrader-4B) grades via an OpenAI-compatible endpoint (default) or
    in-process on a GPU.

Run locally:  python prototype/app.py      (opens http://127.0.0.1:7860)
Deploy:       Hugging Face Space (sdk: gradio, app_file: prototype/app.py); see README.
"""

from __future__ import annotations

import json
import os
import re
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
from xai.explainers import shap_importance                    # noqa: E402
from xai.render_html import heatmap_html_fragment             # noqa: E402

DATA_CSV = os.path.join(_ROOT, "data", "dataset_no_10c_biology.csv")


def readable_tokens(text):
    """Readable Khmer word units of the ORIGINAL text (for display + feedback).

    Strips invisibles and punctuation, then segments with khmernltk, so the words
    render the way a teacher reads them. Falls back to whitespace split if khmernltk
    is unavailable.
    """
    if not text:
        return []
    t = strip_punctuation(strip_invisibles(str(text)))
    try:
        import khmernltk
        return [w for w in khmernltk.word_tokenize(t) if w.strip()]
    except Exception:
        return [w for w in t.split() if w.strip()]


def _compose_side_a(input_format, question_proc, answer_proc):
    """Side A for a two-sided grader, matching data.build_pair.

    ra  -> answer only;  qar -> question + answer (so the model sees the prompt).
    """
    if input_format == "qar":
        return (str(question_proc) + " " + str(answer_proc)).strip()
    return answer_proc


# ───────────────────────── model pillars ─────────────────────────
# Uniform contract for every pillar:
#   .name            display label
#   .preprocess_mode "segment" | "clean"
#   .input_format    "ra" | "qar"
#   .status          one-line "ready" detail (set in __init__)
#   .score(question_proc, reference_proc, answer_proc, max_score) -> float in [0,1]
# A failing __init__ leaves the pillar inactive; the raised message becomes the note.
class ClassicalGrader:
    name = "Classical (TF-IDF + SVR)"
    preprocess_mode = "segment"
    input_format = "ra"

    def __init__(self):
        from models.classical import TFIDFSVR
        C.RAW_CSV = DATA_CSV; C.DROP_SCORE_ZERO = False; C.SPLIT_MODE = "random"
        df = data.load_dataframe(DATA_CSV)
        tr, _, _ = data.split_dataframe(df)
        trp = data.apply_preprocess(tr, self.preprocess_mode)
        a, b = data.build_text_lists(trp, self.input_format)
        self.model = TFIDFSVR().fit(a, b, trp["normalized_score"].values)
        self.status = "trained on startup, CPU"

    def score(self, question_proc, reference_proc, answer_proc, max_score):
        side_a = _compose_side_a(self.input_format, question_proc, answer_proc)
        return float(self.model.predict([side_a], [reference_proc])[0])


class BiLSTMGrader:
    name = "RNN (BiLSTM + Attention)"
    preprocess_mode = "clean"
    input_format = "ra"

    def __init__(self):
        import torch
        from models.bilstm import BiLSTMScorer
        ckpt = os.path.join(_ROOT, "results", "champions",
                            "rnn_clean_ra_bilstm_909", "best.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"champion checkpoint not found at {ckpt}")
        state = torch.load(ckpt, map_location="cpu")
        vocab_size = int(state["emb.weight"].shape[0])
        C.RAW_CSV = DATA_CSV; C.DROP_SCORE_ZERO = False; C.SPLIT_MODE = "random"
        df = data.load_dataframe(DATA_CSV)
        tr, _, _ = data.split_dataframe(df)
        trp = data.apply_preprocess(tr, self.preprocess_mode)
        a, b = data.build_text_lists(trp, self.input_format)
        char2id = data.build_char_vocab(a + b)
        if len(char2id) != vocab_size:        # mapping would be misaligned → refuse
            raise RuntimeError(f"vocab size {len(char2id)} != checkpoint {vocab_size}")
        model = BiLSTMScorer(vocab_size=vocab_size)
        model.load_state_dict(state)
        model.train(False)
        self.torch, self.model, self.char2id, self.max_len = torch, model, char2id, 256
        self.status = "champion checkpoint, CPU"

    def _encode(self, text):
        ids = [self.char2id.get(c, 1) for c in text[: self.max_len]]
        n = len(ids); pad = self.max_len - n
        return (self.torch.tensor(ids + [0] * pad).unsqueeze(0),
                self.torch.tensor([1] * n + [0] * pad).unsqueeze(0))

    def score(self, question_proc, reference_proc, answer_proc, max_score):
        side_a = _compose_side_a(self.input_format, question_proc, answer_proc)
        ia, ma = self._encode(side_a); ib, mb = self._encode(reference_proc)
        with self.torch.no_grad():
            return float(self.model(ia, ma, ib, mb).item())


class EncoderGrader:
    """GTE dual-encoder. Loads a fine-tuned checkpoint if present; CPU-capable.

    The architecture (backbone, input format, max-score feature) is read from the
    run's config.json so it matches the trained head exactly. The checkpoint is not
    shipped by default (HPC-trained); drop a ``best.pt`` into the champion dir or set
    ENCODER_CKPT to activate this pillar.
    """
    name = "Transformer (GTE dual-encoder)"
    preprocess_mode = "clean"   # overwritten from config.json
    input_format = "qar"        # overwritten from config.json

    def __init__(self):
        # Check the checkpoint first so the inactive reason names the real blocker
        # (the weights), not a missing optional dependency.
        ckpt = os.environ.get("ENCODER_CKPT", os.path.join(
            _ROOT, "results", "champions",
            "encoder_clean_qar_dual_gte_maxfeat_1184", "best.pt"))
        if not os.path.exists(ckpt):
            raise RuntimeError(
                "checkpoint not found: export the fine-tuned GTE-dual best.pt to "
                f"{ckpt} (or set ENCODER_CKPT) to activate this pillar")
        import torch
        from transformers import AutoTokenizer
        from models.dual import DualEncoderScorer
        cfg_path = os.path.join(os.path.dirname(ckpt), "config.json")
        cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}
        backbone = cfg.get("backbone", C.TRANSFORMER_BACKBONES["gte"])
        self.preprocess_mode = cfg.get("preprocess", "clean")
        self.input_format = cfg.get("input", "qar")
        self.max_feat = bool(cfg.get("max_feat", False))
        self.tok = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
        self.model = DualEncoderScorer(backbone, max_feat_dim=1 if self.max_feat else 0)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.train(False)
        self.torch, self.max_len = torch, C.TXFMR_MAX_LEN
        self.status = f"checkpoint loaded ({os.path.basename(os.path.dirname(ckpt))}), CPU"

    def _enc(self, text):
        e = self.tok(text, max_length=self.max_len, padding="max_length",
                     truncation=True, return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    def score(self, question_proc, reference_proc, answer_proc, max_score):
        side_a = _compose_side_a(self.input_format, question_proc, answer_proc)
        ia, ma = self._enc(side_a); ib, mb = self._enc(reference_proc)
        kwargs = {}
        if self.max_feat:
            kwargs["max_score_feat"] = self.torch.tensor(
                [[float(max_score) / float(C.MAX_SCORE_NORMALIZER)]],
                dtype=self.torch.float32)
        with self.torch.no_grad():
            out = self.model(ia, ma, ib, mb, **kwargs)
        return float(out.reshape(-1)[0].item())


# ── LLM grading prompt (mirrors experiments/exp08_llm_finetune.py:format_prompt) ──
# Kept inline so the endpoint path stays self-contained and CPU-light (the experiment
# module pulls in the training harness). The in-process GPU path lazy-imports exp08.
def _llm_grading_prompt(question, reference, answer, max_score):
    return (
        f"Below is a Khmer short-answer grading task. Score the student's answer "
        f"on a scale from 0 to {int(max_score)}.\n\n"
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Student answer: {answer}\n\n"
        f"The score (integer from 0 to {int(max_score)}):"
    )


# Markers a reasoning model may emit before the final answer; keep only the tail.
_LLM_ANSWER_MARKERS = ("</think>", "<|channel|>final", "final answer", "Final answer",
                       "answer:", "Answer:", "score:", "Score:")


def _parse_llm_int(text, max_score):
    parsed = text or ""
    for marker in _LLM_ANSWER_MARKERS:
        if marker in parsed:
            parsed = parsed.split(marker)[-1]
    m = re.search(r"\d+", parsed)
    if m:
        return max(0, min(int(m.group()), int(max_score)))
    return int(max_score) // 2


# Endpoint config for the LLM *grader* (separate from the FEEDBACK_LLM_* vars: grading
# uses the fine-tuned KhmerGrader, feedback uses a general instruct model).
GRADER_LLM_BASE_URL = os.environ.get("GRADER_LLM_BASE_URL", "").strip()
GRADER_LLM_MODEL = os.environ.get("GRADER_LLM_MODEL", "qwen-khmergrader-4b")
GRADER_LLM_API_KEY = os.environ.get("GRADER_LLM_API_KEY", "ollama")


class LLMGrader:
    name = "LLM (Qwen-KhmerGrader-4B)"
    preprocess_mode = "clean"
    input_format = "qar"

    def __init__(self):
        if os.environ.get("GRADER_LLM_INPROCESS") == "1":
            self._init_inprocess()          # GPU only; may raise
            self.backend = "inprocess"
            self.status = f"in-process ({GRADER_LLM_MODEL}), GPU"
        elif GRADER_LLM_BASE_URL:
            self.backend = "endpoint"
            self.status = f"endpoint {GRADER_LLM_BASE_URL} ({GRADER_LLM_MODEL})"
        else:
            raise RuntimeError(
                "no grading backend configured: set GRADER_LLM_BASE_URL to an "
                "OpenAI-compatible endpoint serving a KhmerGrader model, or "
                "GRADER_LLM_INPROCESS=1 on a GPU (with GRADER_LLM_MODEL / "
                "GRADER_LLM_ADAPTER) to load the adapter in-process")

    # GPU-only: load base + trained LoRA adapter. Runs in the full repo env, so it is
    # safe to lazy-import the experiment helpers here.
    def _init_inprocess(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        base = GRADER_LLM_MODEL if "/" in GRADER_LLM_MODEL else "Qwen/Qwen3.5-4B"
        adapter = os.environ.get("GRADER_LLM_ADAPTER", "").strip()
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base, trust_remote_code=True, device_map="auto", torch_dtype="auto")
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        model.train(False)
        self.torch, self.tok, self.model = torch, tok, model

    def score(self, question_proc, reference_proc, answer_proc, max_score):
        if self.backend == "endpoint":
            return self._score_endpoint(question_proc, reference_proc, answer_proc, max_score)
        return self._score_inprocess(question_proc, reference_proc, answer_proc, max_score)

    def _score_endpoint(self, question_proc, reference_proc, answer_proc, max_score):
        import requests
        prompt = _llm_grading_prompt(question_proc, reference_proc, answer_proc, max_score)
        r = requests.post(
            f"{GRADER_LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {GRADER_LLM_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": GRADER_LLM_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.0, "max_tokens": 16, "stream": False},
            timeout=60,
        )
        r.raise_for_status()
        txt = (r.json()["choices"][0]["message"]["content"] or "").strip()
        return _parse_llm_int(txt, max_score) / max(int(max_score), 1)

    def _score_inprocess(self, question_proc, reference_proc, answer_proc, max_score):
        from experiments.exp08_llm_finetune import render_prompt_text, generate_score
        row = {"Question_proc": question_proc, "Reference_proc": reference_proc,
               "Answer_proc": answer_proc, "Max Score": int(max_score),
               "Student Score": 0}
        prompt = render_prompt_text(self.tok, row, with_answer=False)
        device = next(self.model.parameters()).device
        raw, _ = generate_score(self.model, self.tok, prompt, int(max_score), device)
        return raw / max(int(max_score), 1)


def _load_models():
    available, inactive, notes = {}, {}, []
    for cls in (ClassicalGrader, BiLSTMGrader, EncoderGrader, LLMGrader):
        try:
            g = cls()
            available[g.name] = g
            notes.append(f"{g.name}: ready ({g.status}).")
        except Exception as e:
            inactive[cls.name] = str(e) or type(e).__name__
            notes.append(f"{cls.name}: inactive ({type(e).__name__}).")
    return available, inactive, notes


MODELS, INACTIVE, LOAD_NOTES = _load_models()
ALL_CHOICES = list(MODELS.keys()) + list(INACTIVE.keys())
DEFAULT_MODEL = ("Classical (TF-IDF + SVR)" if "Classical (TF-IDF + SVR)" in MODELS
                 else (ALL_CHOICES[0] if ALL_CHOICES else ""))


# ───────────────────────── grading + feedback ─────────────────────────
# Feedback config (separate from the GRADER_LLM_* vars): grading uses the fine-tuned
# KhmerGrader; the WRITTEN feedback uses a general open-source instruct model served over
# an OpenAI-compatible endpoint (Ollama / vLLM / llama.cpp / LM Studio), never a
# proprietary grading API. When no endpoint is reachable we fall back to rule-based text.
FEEDBACK_LLM_BASE_URL = os.environ.get("FEEDBACK_LLM_BASE_URL", "").strip()
FEEDBACK_LLM_MODEL = os.environ.get("FEEDBACK_LLM_MODEL", "qwen2.5:7b-instruct")
FEEDBACK_LLM_API_KEY = os.environ.get("FEEDBACK_LLM_API_KEY", "ollama")


def _missing_reference_points(answer, reference):
    """Readable reference keywords (length > 1) absent from the answer, original order."""
    ans_set = set(readable_tokens(answer))
    seen, missing = set(), []
    for w in readable_tokens(reference):
        if len(w) > 1 and w not in ans_set and w not in seen:
            seen.add(w); missing.append(w)
    return missing


def _score_band(yhat):
    if yhat >= 0.85:
        return "Excellent. The answer covers the key content of the reference."
    if yhat >= 0.60:
        return "Good. Most of the key content is present."
    if yhat >= 0.40:
        return "Partial. Some key content is present, but important points are missing."
    return "Needs improvement. The key content of the reference is largely missing."


def _rule_feedback(yhat, missing):
    """Deterministic, offline feedback: a score band plus the missing reference points."""
    tip = ("\n\n**Reference points not found in the answer:** " + "，".join(missing[:8])
           if missing else "\n\nThe answer touches the main reference points.")
    return f"**Feedback.** {_score_band(yhat)}{tip}"


def _llm_feedback(question, reference, answer, raw, max_score, missing):
    """Short Khmer feedback from an open-source instruct model over an OpenAI-compatible
    endpoint. Returns None on any failure so the caller falls back to rule-based text."""
    if not FEEDBACK_LLM_BASE_URL:
        return None
    try:
        import requests
        miss = "，".join(missing[:8]) if missing else "(none)"
        prompt = (
            "You are a Cambodian secondary-school teacher assistant. Write SHORT, kind, "
            "constructive feedback in KHMER (2-3 sentences) for the student, based only on "
            "the information below. Say what is correct and which reference points are "
            "missing. Do not invent facts and do not change the score.\n\n"
            f"Question: {question}\n"
            f"Reference answer: {reference}\n"
            f"Student answer: {answer}\n"
            f"Score: {raw}/{max_score}\n"
            f"Reference points missing from the answer: {miss}\n\n"
            "Feedback (in Khmer):"
        )
        r = requests.post(
            f"{FEEDBACK_LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {FEEDBACK_LLM_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": FEEDBACK_LLM_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3, "max_tokens": 200, "stream": False},
            timeout=60,
        )
        r.raise_for_status()
        txt = (r.json()["choices"][0]["message"]["content"] or "").strip()
        return ("**Feedback.** " + txt) if txt else None
    except Exception:
        return None


def _feedback(question, yhat, raw, max_score, answer, reference):
    """Compose feedback: open-source LLM if an endpoint is reachable, else rule-based."""
    missing = _missing_reference_points(answer, reference)
    body = _llm_feedback(question, reference, answer, raw, max_score, missing)
    if not body:
        body = _rule_feedback(yhat, missing)
    return (f"{body}\n\n"
            f"*Teacher-assist only. Please review the highlighted words and the suggestion "
            f"before finalizing the grade.*")


def grade(question, reference, answer, max_score, model_name, explain=True):
    if not (answer or "").strip() or not (reference or "").strip():
        return ("⚠️ Please enter both a **reference answer** and a **student answer**.", "", "")
    grader = MODELS.get(model_name)
    if grader is None:
        reason = INACTIVE.get(model_name, "not available in this environment.")
        return (f"**{model_name}** is not active here: {reason}\n\n"
                f"Pick the **Classical** or **RNN** pillar, or provide the resource above and "
                f"restart the app.", "", "")

    mode, fmt = grader.preprocess_mode, grader.input_format
    q_proc = preprocess(question or "", mode)
    ref_proc = preprocess(reference, mode)
    ans_proc = preprocess(answer, mode)
    try:
        max_score = max(1, int(max_score))
    except (TypeError, ValueError):
        max_score = 10

    try:
        yhat = float(np.clip(grader.score(q_proc, ref_proc, ans_proc, max_score), 0.0, 1.0))
    except Exception as e:
        return (f"⚠️ **{grader.name}** could not grade this answer "
                f"({type(e).__name__}: {e}).", "", "")
    raw = int(round(yhat * max_score))
    grade5 = int(round(yhat * 4))

    score_md = (f"### Score: **{raw} / {max_score}**  &nbsp; "
                f"(model estimate {yhat * 100:.0f}%, grade {grade5}/4)\n\n"
                f"Model: **{grader.name}**\n\n"
                f"<small><i>This is the model's <b>predicted grade</b>: a learned estimate of how the "
                f"teacher would score this content, not an exact-match check. Even a verbatim copy of "
                f"the reference may not read 100%, because the model imitates the teacher's grading "
                f"rather than measuring text overlap.</i></small>")

    fb = _feedback(question, yhat, raw, max_score, answer, reference)

    # Explainable mode toggle: SHAP needs many re-scorings per answer, and for the LLM each one is a
    # full generation, so a teacher can switch the explanation off for a fast score-only result.
    if not explain:
        return (score_md, "<i>Explanation off. Tick <b>Show explanation</b> to see the "
                "SHAP word-attribution heatmap (slower, especially for the LLM).</i>", fb)

    # Explanation: attribute over READABLE original word units; the predictor re-runs the
    # model's own preprocessing inside score(). For qar pillars the question is prepended
    # inside score(), so the attribution still perturbs only the answer.
    disp = readable_tokens(answer)
    if len(disp) < 2:
        return (score_md, "<i>Answer too short to explain.</i>", fb)
    ans_disp = " ".join(disp)

    def pred_one(a_disp, b=ref_proc):
        return grader.score(q_proc, b, preprocess(a_disp, mode), max_score)

    # SHAP word attribution is the unified explanation method: model-agnostic (it only calls
    # the scoring function), so one method explains the classical SVR, the BiLSTM, the
    # encoder, and the LLM, matching the SHAP study reported in the thesis.
    try:
        words, imp = shap_importance(pred_one, ans_disp, ref_proc, "segment")
        heat = heatmap_html_fragment(words, imp,
                                     f"SHAP word attribution (highlighted words): {grader.name}")
    except Exception as e:
        heat = f"<i>Explanation unavailable ({type(e).__name__}).</i>"
    return score_md, heat, fb


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
    with gr.Blocks(title="Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education") as demo:
        gr.Markdown(
            "# Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education\n"
            "Enter the **question**, the **reference answer**, and the **student answer**, "
            "choose a model pillar, then press **Grade**. You get a score, a Khmer "
            "**word-attribution** heatmap (the highlighted words that drove the grade), and "
            "short feedback. *Teacher-assist tool. "
            "The teacher stays in the loop.*")
        with gr.Row():
            with gr.Column(scale=1):
                question = gr.Textbox(label="Question (optional)", lines=2)
                reference = gr.Textbox(label="Reference answer", lines=4)
                answer = gr.Textbox(label="Student answer", lines=4)
                with gr.Row():
                    max_score = gr.Number(label="Max score", value=10, precision=0, minimum=1)
                    model_name = gr.Dropdown(ALL_CHOICES, value=DEFAULT_MODEL, label="Model pillar")
                explain = gr.Checkbox(value=True, label="Show explanation (SHAP heatmap; slower, "
                                                        "especially for the LLM)")
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
                  inputs=[question, reference, answer, max_score, model_name, explain],
                  outputs=[score_out, heat_out, fb_out])
    return demo


if __name__ == "__main__":
    # theme passed here (Gradio 6 moved it out of the Blocks constructor)
    build_demo().launch(theme=gr.themes.Soft(primary_hue="blue"))
