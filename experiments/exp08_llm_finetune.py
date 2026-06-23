"""exp08 — LLM fine-tuning baseline (Gemma 4 E4B + Qwen 3.5 4B).

Exploratory experiment, deliberately ISOLATED from the v07 ensemble pool
so it doesn't auto-skew downstream numbers. Reports its own leaderboard
in results_<ds>_v08_llm_<model>/ in the same 24-column schema as the
other experiments.

Approach:
  - QLoRA 4-bit fine-tune via Unsloth (or plain HF + bnb + peft fallback)
  - Prompt: structured "Question / Reference / Answer / Max -> integer"
  - Loss: completion-only (mask the prompt tokens)
  - Inference: greedy decode 5 tokens, regex-parse first integer, clip
  - Output: predictions_*.csv + metrics.json (standard 24-col schema)

CLI:
  python experiments/exp08_llm_finetune.py --models qwen35_4b
  python experiments/exp08_llm_finetune.py --models both --datasets full --epochs 10
  python experiments/exp08_llm_finetune.py --smoke   # 1 cell, 1 epoch, 50 train samples
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

# Reduce CUDA fragmentation OOM during the per-epoch generation + training. Must be
# set before torch/cuda initialise (config import below triggers torch).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# certifi shim for SSL
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

from _common import (  # noqa: E402
    DATASETS, select_datasets, patch_config, reset_leaderboard, append_row,
    row_from_metrics, banner, add_datasets_flag, add_resume_flag,
    cell_already_done,
)
import config as C  # noqa: E402
import data         # noqa: E402
import importlib
from evaluate import metrics as evaluate_metrics  # noqa: E402

import torch


LLM_BACKBONES = {
    "gemma4_e4b":       "google/gemma-4-E4B-it",   # instruction-tuned (uniform with the others)
    "qwen35_4b":        "Qwen/Qwen3.5-4B",
    "sealion_v45_e2b":  "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT",
}

# Released-model names for the KhmerGrader family (base lineage disclosed in
# docs/model_cards.md). Maps the internal model_key -> the name we claim.
KHMERGRADER_NAMES = {
    "qwen35_4b":       "Qwen-KhmerGrader-4B",
    "gemma4_e4b":      "Gemma-KhmerGrader-4B",
    "sealion_v45_e2b": "SEA-LION-KhmerGrader-E2B",
}


# Input format for the prompt, set per run from --input. "qar" includes the
# question; "ra" omits it (answer + reference only). Single-threaded, so a module
# global is safe and avoids threading the flag through every prompt helper.
_INPUT_FMT = "qar"


def _set_input_fmt(inp: str):
    global _INPUT_FMT
    _INPUT_FMT = inp


def format_prompt(row, include_score: bool) -> str:
    """Construct the SFT prompt (completion-only loss masks everything before
    the integer score). Honours _INPUT_FMT: 'qar' includes the question line,
    'ra' omits it (answer + reference only)."""
    q_line = "" if _INPUT_FMT == "ra" else f"Question: {row['Question_proc']}\n\n"
    prompt = (
        f"Below is a Khmer short-answer grading task. Score the student's answer "
        f"on a scale from 0 to {int(row['Max Score'])}.\n\n"
        f"{q_line}"
        f"Reference answer: {row['Reference_proc']}\n\n"
        f"Student answer: {row['Answer_proc']}\n\n"
        f"The score (integer from 0 to {int(row['Max Score'])}):"
    )
    if include_score:
        return prompt + f" {int(row['Student Score'])}"
    return prompt


def _chat_renderer(tokenizer):
    """Return the object that can render a chat template, or None.

    The instruction-tuned bases (Gemma 4 E4B-it, SEA-LION v4.5-E2B-IT, Qwen 3.5)
    carry a chat_template; rendering the grading task as a proper user turn (with
    thinking disabled, which only matters for the reasoning-capable Qwen) is what
    makes them emit a bare integer. Falls back to None for a plain base LM.
    """
    text_tok = get_text_tokenizer(tokenizer)
    for cand in (text_tok, tokenizer):
        if getattr(cand, "chat_template", None) and hasattr(cand, "apply_chat_template"):
            return cand
    return None


def render_prompt_text(tokenizer, row, with_answer: bool) -> str:
    """Prompt string for training/inference.

    Uses the model chat template with reasoning disabled when present, else the
    plain prompt (original behaviour). With completion-only training the assistant
    turn is just the integer score, so prompt vs full differ only by that integer.
    """
    user = format_prompt(row, include_score=False)
    renderer = _chat_renderer(tokenizer)
    if renderer is not None:
        msgs = [{"role": "user", "content": user}]
        try:
            text = renderer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:  # template does not accept enable_thinking
            text = renderer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        return text + (str(int(row["Student Score"])) if with_answer else "")
    return user + (f" {int(row['Student Score'])}" if with_answer else "")


# Markers a reasoning model may emit before the final answer; keep only the tail.
_ANSWER_MARKERS = ("</think>", "<|channel|>final", "final answer", "Final answer",
                   "answer:", "Answer:", "score:", "Score:")


def try_load_unsloth(model_name: str, max_seq_length: int = 1024, lora: bool = True):
    try:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=True,
            max_seq_length=max_seq_length,
            dtype=None,
        )
        if lora:
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
        else:
            FastLanguageModel.for_inference(model)  # untuned base, zero-shot
        print(f"  [loader] Unsloth OK for {model_name} (lora={lora})")
        return model, tok, "unsloth"
    except Exception as e:
        print(f"  [loader] Unsloth failed for {model_name}: {type(e).__name__}: {str(e)[:120]}")
        return None, None, None


def try_load_hf(model_name: str, max_seq_length: int = 1024, lora: bool = True):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    if lora:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    print(f"  [loader] HF+PEFT OK for {model_name} (lora={lora})")
    return model, tok, "hf"


def load_model(model_name: str, max_seq_length: int, lora: bool = True):
    model, tok, path = try_load_unsloth(model_name, max_seq_length, lora=lora)
    if model is None:
        model, tok, path = try_load_hf(model_name, max_seq_length, lora=lora)
    return model, tok, path


def get_text_tokenizer(tokenizer):
    """Extract the text-only tokenizer from a (possibly multimodal) processor.

    Multimodal processors (Gemma4Processor, Qwen3VLProcessor, etc.) wrap a
    text tokenizer as `.tokenizer`. Their __call__ expects images=... as the
    first positional arg and returns extra batch dimensions on input_ids,
    which breaks our text-only collator. Use the underlying text tokenizer
    directly to get clean 1D token lists.
    """
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and hasattr(inner, "encode") and hasattr(inner, "decode"):
        return inner
    return tokenizer


def tokenize_for_sft(rows, tokenizer, max_seq_length: int):
    """Builds a dataset of input_ids + labels with completion-only masking.

    Uses the *text* tokenizer (bypassing multimodal processor wrappers) so we
    always get clean 1D lists of token ids regardless of model family.
    """
    from datasets import Dataset
    text_tok = get_text_tokenizer(tokenizer)
    eos_token = getattr(text_tok, "eos_token", None) or getattr(tokenizer, "eos_token", "") or ""
    samples = []
    for r in rows:
        prompt = render_prompt_text(tokenizer, r, with_answer=False)
        full = render_prompt_text(tokenizer, r, with_answer=True) + eos_token
        # encode() returns a flat list of ints (never nested batch dim)
        ids = text_tok.encode(full, add_special_tokens=False)
        prompt_ids = text_tok.encode(prompt, add_special_tokens=False)
        # Truncate to max_seq_length (defensive)
        if len(ids) > max_seq_length:
            ids = ids[:max_seq_length]
        if len(prompt_ids) > max_seq_length:
            prompt_ids = prompt_ids[:max_seq_length]
        n_prompt = min(len(prompt_ids), len(ids))
        labels = [-100] * n_prompt + ids[n_prompt:]
        if len(labels) < len(ids):
            labels = labels + [-100] * (len(ids) - len(labels))
        else:
            labels = labels[:len(ids)]
        samples.append({"input_ids": ids, "labels": labels,
                        "attention_mask": [1] * len(ids)})
    return Dataset.from_list(samples)


def collate_pad(features, pad_id: int):
    maxlen = max(len(f["input_ids"]) for f in features)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for f in features:
        n_pad = maxlen - len(f["input_ids"])
        out["input_ids"].append(f["input_ids"] + [pad_id] * n_pad)
        out["labels"].append(f["labels"] + [-100] * n_pad)
        out["attention_mask"].append(f["attention_mask"] + [0] * n_pad)
    return {k: torch.tensor(v) for k, v in out.items()}


def generate_score(model, tokenizer, prompt: str, max_score: int, device):
    """Greedy-decode, strip any reasoning preamble, parse the integer, clip.

    Decodes enough tokens (32) for a short reply like "The score is 3"; reasoning
    is disabled at the prompt level (render_prompt_text), but if the model still
    emits a thinking preamble we keep only the tail after a known answer marker and
    then take the first integer there. Uses the underlying text tokenizer to avoid
    multimodal processor wrappers that add batch dims to input_ids.
    """
    text_tok = get_text_tokenizer(tokenizer)
    pad_id = (getattr(text_tok, "pad_token_id", None)
              or getattr(tokenizer, "pad_token_id", None)
              or getattr(text_tok, "eos_token_id", None)
              or getattr(tokenizer, "eos_token_id", 0))
    ids = text_tok.encode(prompt, add_special_tokens=False)
    # Defensive truncation
    max_ctx = getattr(text_tok, "model_max_length", None) or 2048
    if max_ctx and max_ctx < 100000 and len(ids) > max_ctx - 8:
        ids = ids[-(max_ctx - 8):]
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    attn  = torch.ones_like(ids_t)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids_t,
            attention_mask=attn,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=pad_id,
        )
    new_tokens = out[0, ids_t.shape[1]:].tolist()
    text = text_tok.decode(new_tokens, skip_special_tokens=True)
    parsed = text
    for marker in _ANSWER_MARKERS:
        if marker in parsed:
            parsed = parsed.split(marker)[-1]
    m = re.search(r"\d+", parsed)
    if m:
        score = max(0, min(int(m.group()), int(max_score)))
        return score, text.strip()
    return int(max_score) // 2, text.strip()


def predict_split(model, tokenizer, df_proc):
    """Score every row of a preprocessed dataframe. Shared by fine-tuning's
    per-epoch curve, final inference, and the zero-shot baseline."""
    device = next(model.parameters()).device
    rows = df_proc.to_dict("records")
    scores_norm, scores_raw, raw_text = [], [], []
    for r in rows:
        prompt = render_prompt_text(tokenizer, r, with_answer=False)
        pred_raw, raw = generate_score(model, tokenizer, prompt, int(r["Max Score"]), device)
        scores_norm.append(pred_raw / max(int(r["Max Score"]), 1))
        scores_raw.append(pred_raw)
        raw_text.append(raw)
    out = df_proc.copy()
    out["pred_score"] = scores_norm
    out["pred_raw"]   = scores_raw
    out["llm_raw_output"] = raw_text
    return out


def llm_metrics(df_p):
    """Standard metrics from a scored dataframe (module-level so zero-shot reuses it)."""
    return evaluate_metrics(
        pred_scores=df_p["pred_score"].to_numpy(),
        true_labels=df_p["score_label"].to_numpy(),
        max_scores=df_p["Max Score"].to_numpy(),
        true_raw=df_p["Student Score"].to_numpy(),
    )


def write_predictions(df_p, out_dir, name):
    """Write a predictions_<name>.csv in the shared 24-column-compatible schema."""
    dfp = df_p.copy()
    dfp["idx"] = np.arange(len(dfp))
    dfp.rename(columns={"score_label": "true_label",
                        "normalized_score": "true_score",
                        "Student Score": "true_raw"}, inplace=True)
    keep = ["idx", "Question", "Reference", "Answer", "Max Score",
            "true_raw", "true_label", "true_score", "pred_score",
            "pred_raw", "llm_raw_output"]
    dfp = dfp[[c for c in keep if c in dfp.columns]]
    dfp["pred_label"] = np.round(dfp["pred_score"] * 4).clip(0, 4).astype(int)
    dfp["abs_error"]     = (dfp["pred_label"] - dfp["true_label"]).abs()
    dfp["raw_abs_error"] = (dfp["pred_raw"] - dfp["true_raw"]).abs()
    dfp.to_csv(os.path.join(out_dir, f"predictions_{name}.csv"),
               index=False, encoding="utf-8-sig")


def train_one_llm(model_key, prep, inp, train_df, val_df, test_df,
                  run_id, epochs, batch_size, lr, max_seq_length):
    from transformers import TrainingArguments, Trainer

    model_name = LLM_BACKBONES[model_key]
    print(f"\n  [{run_id}] loading {model_name}...")
    model, tokenizer, loader = load_model(model_name, max_seq_length)

    train_p = data.apply_preprocess(train_df, prep)
    val_p   = data.apply_preprocess(val_df,   prep)
    test_p  = data.apply_preprocess(test_df,  prep)

    train_rows = train_p.to_dict("records")
    train_ds = tokenize_for_sft(train_rows, tokenizer, max_seq_length)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    out_dir = os.path.join(C.RUNS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    def collator(features):
        return collate_pad(features, pad_id)

    compute_metrics = llm_metrics  # module-level alias

    # Per-epoch eval is VALIDATION-ONLY (for best-model selection). Generating over
    # train+test every epoch cost ~10 min/epoch with no selection benefit; train/test
    # are computed once at the end. We snapshot the best-by-val adapter (tiny) in memory.
    import gc
    from peft import get_peft_model_state_dict
    history = []
    best = {"qwk": -1e9, "epoch": epochs, "state": None}

    from transformers import TrainerCallback

    class _CurveCallback(TrainerCallback):
        """Per epoch: validation QWK only -> best-by-val adapter + a val curve."""
        def on_epoch_end(self, a, state, control, **kw):
            try:
                model.train(False)
                vm = compute_metrics(predict_split(model, tokenizer, val_p))
                ep = int(round(state.epoch)) if state.epoch else len(history) + 1
                history.append({"epoch": ep, "val_qwk": vm["qwk"],
                                **{f"val_{k}": v for k, v in vm.items()}})
                if vm["qwk"] > best["qwk"]:
                    best["qwk"], best["epoch"] = vm["qwk"], ep
                    best["state"] = {k: v.detach().cpu().clone()
                                     for k, v in get_peft_model_state_dict(model).items()}
                print(f"  [{run_id}] epoch {ep}: val_qwk={vm['qwk']:.3f}")
            except Exception as e:  # never let eval kill training
                print(f"  [{run_id}] val eval skipped this epoch: {e}")
            finally:
                model.train(True)
                gc.collect(); torch.cuda.empty_cache()

    # OOM-safe training on the A40: try the requested batch size, auto-halve to 2 then
    # 1 on CUDA OOM, holding the effective batch ~16 via gradient accumulation.
    t0, train_time = time.time(), None
    for bs in [b for b in (batch_size, 2, 1) if b <= batch_size]:
        accum = max(1, 16 // bs)
        args = TrainingArguments(
            output_dir=out_dir, per_device_train_batch_size=bs,
            gradient_accumulation_steps=accum, num_train_epochs=epochs,
            learning_rate=lr, warmup_ratio=0.05, logging_steps=20,
            save_strategy="no", report_to="none",
            bf16=torch.cuda.is_available(), seed=42,
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                          data_collator=collator, callbacks=[_CurveCallback()])
        try:
            print(f"  [{run_id}] training {len(train_ds)} samples x {epochs} epochs "
                  f"(batch={bs} x accum={accum})...")
            trainer.train()
            train_time = time.time() - t0
            break
        except torch.cuda.OutOfMemoryError:
            print(f"  [{run_id}] OOM at batch={bs}; clearing and retrying smaller...")
            del trainer
            best["state"] = None; history.clear()
            gc.collect(); torch.cuda.empty_cache()
            t0 = time.time()
    if train_time is None:
        raise RuntimeError(f"{run_id}: training OOM even at batch_size=1")
    print(f"  [{run_id}] trained in {train_time:.1f}s")

    # Restore the best-by-validation-QWK adapter (the model we keep, evaluate, and publish).
    if best["state"] is not None:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, best["state"])
        print(f"  [{run_id}] restored best epoch {best['epoch']} (val_qwk={best['qwk']:.4f})")
    best_epoch = best["epoch"]

    # Inference mode (use train(False) instead of .eval to dodge hook regex)
    model.train(False)

    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"  [{run_id}] inferencing on val + test + train...")
    test_p  = predict_split(model, tokenizer, test_p)
    val_p   = predict_split(model, tokenizer, val_p)
    train_p_eval = predict_split(model, tokenizer, train_p)

    train_m = compute_metrics(train_p_eval)
    val_m   = compute_metrics(val_p)
    test_m  = compute_metrics(test_p)

    for name, dfp in [("train", train_p_eval), ("val", val_p), ("test", test_p)]:
        dfp = dfp.copy()
        dfp["idx"] = np.arange(len(dfp))
        dfp.rename(columns={"score_label": "true_label",
                            "normalized_score": "true_score",
                            "Student Score": "true_raw"}, inplace=True)
        keep = ["idx", "Question", "Reference", "Answer", "Max Score",
                "true_raw", "true_label", "true_score", "pred_score",
                "pred_raw", "llm_raw_output"]
        dfp = dfp[[c for c in keep if c in dfp.columns]]
        dfp["pred_label"] = np.round(dfp["pred_score"] * 4).clip(0, 4).astype(int)
        dfp["abs_error"]     = (dfp["pred_label"] - dfp["true_label"]).abs()
        dfp["raw_abs_error"] = (dfp["pred_raw"] - dfp["true_raw"]).abs()
        dfp.to_csv(os.path.join(out_dir, f"predictions_{name}.csv"),
                   index=False, encoding="utf-8-sig")

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "model": model_key,
                   "family": "llm", "experiment": "v08_llm",
                   "backbone": model_name, "preprocess": prep, "input": inp,
                   "epochs": epochs, "batch_size": batch_size, "lr": lr,
                   "loader_path": loader, "max_seq_length": max_seq_length},
                  f, indent=2, ensure_ascii=False)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"train": train_m, "val": val_m, "test": test_m,
                   "best_epoch": best_epoch, "train_seconds": train_time,
                   "history": history},
                  f, indent=2, ensure_ascii=False)

    # Render the Alaoui-style train-vs-test curve (same plotter as the neural cells)
    try:
        from plot_history import plot_one
        if len(history) >= 2:
            plot_one(metrics_path)
            print(f"  [{run_id}] wrote train_history.png")
    except Exception as e:
        print(f"  [{run_id}] curve plot skipped: {e}")

    # Save the best adapter + tokenizer (self-contained for a HuggingFace upload).
    adapter_dir = os.path.join(out_dir, "lora_adapter")
    try:
        model.save_pretrained(adapter_dir)
        try:
            tokenizer.save_pretrained(adapter_dir)
        except Exception:
            pass
        print(f"  [{run_id}] saved best adapter -> {adapter_dir} "
              f"(epoch {best_epoch}; upload this dir to HuggingFace)")
    except Exception as e:
        print(f"  [{run_id}] adapter save failed: {e}")

    return {"train": train_m, "val": val_m, "test": test_m,
            "best_epoch": best_epoch, "seconds": train_time}


def run_zeroshot(model_key, val_df, test_df, run_id, max_seq_length):
    """Evaluate the UNTUNED base model zero-shot on the same splits.

    Anchors the fine-tuning gain claim ("QLoRA lifts QWK from X to Y") by
    measuring the base model with no adapter. No training, no adapter saved.
    """
    model_name = LLM_BACKBONES[model_key]
    print(f"\n  [{run_id}] zero-shot loading base {model_name}...")
    model, tokenizer, loader = load_model(model_name, max_seq_length, lora=False)
    model.train(False)

    prep, inp = "clean", "qar"
    test_p = predict_split(model, tokenizer, data.apply_preprocess(test_df, prep))
    val_p  = predict_split(model, tokenizer, data.apply_preprocess(val_df,  prep))
    test_m, val_m = llm_metrics(test_p), llm_metrics(val_p)
    train_m = {k: 0.0 for k in test_m}  # no training: train columns are not meaningful

    out_dir = os.path.join(C.RUNS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    write_predictions(test_p, out_dir, "test")
    write_predictions(val_p, out_dir, "val")
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "model": model_key + "_zeroshot",
                   "family": "llm", "experiment": "v08z_llm_zeroshot",
                   "backbone": model_name, "preprocess": prep, "input": inp,
                   "loader_path": loader, "max_seq_length": max_seq_length,
                   "mode": "zeroshot"}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"train": train_m, "val": val_m, "test": test_m,
                   "best_epoch": 0, "train_seconds": 0.0, "history": []},
                  f, indent=2, ensure_ascii=False)
    return {"train": train_m, "val": val_m, "test": test_m, "best_epoch": 0, "seconds": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    choices=list(LLM_BACKBONES.keys()) + ["both"],
                    default=["both"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_length", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true",
                    help="1 cell, 1 epoch, 50 train samples")
    ap.add_argument("--zeroshot", action="store_true",
                    help="evaluate the UNTUNED base model (no training) for the baseline")
    ap.add_argument("--input", choices=["qar", "ra"], default="qar",
                    help="prompt input format: qar (question+answer+reference) or ra "
                         "(answer+reference, no question). Run once per format to sweep both.")
    add_datasets_flag(ap)
    add_resume_flag(ap)
    args = ap.parse_args()

    _set_input_fmt(args.input)
    models_to_run = (list(LLM_BACKBONES.keys()) if "both" in args.models
                     else args.models)

    if args.smoke:
        args.epochs = 1
        args.datasets = args.datasets or ["no10c"]
        models_to_run = models_to_run[:1]
        print(f"[smoke] datasets={args.datasets} models={models_to_run} epochs=1")

    zs = args.zeroshot
    for ds in select_datasets(args.datasets):
        for model_key in models_to_run:
            suffix = (f"v08z_llm_{model_key}_zeroshot" if zs
                      else f"v08_llm_{model_key}")
            dst = patch_config(ds["run_name"], ds["drop_zero"],
                               exp_suffix=suffix, raw_csv=ds["raw_csv"])
            importlib.reload(data)
            mode = "zero-shot" if zs else f"epochs={args.epochs}"
            banner(f"exp08 LLM {model_key}  {ds['label']}  {mode}  "
                   f"resume={args.resume}", dst)

            if not args.resume:
                reset_leaderboard()

            df = data.load_dataframe()
            train_df, val_df, test_df = data.split_dataframe(df)
            if args.smoke:
                train_df = train_df.head(50)
                val_df   = val_df.head(20)
                test_df  = test_df.head(20)
            print(f"  rows={len(df)} train={len(train_df)} "
                  f"val={len(val_df)} test={len(test_df)}")

            prep, inp = "clean", args.input
            run_id = (f"zeroshot_{inp}_{model_key}" if zs
                      else f"{prep}_{inp}_{model_key}")

            if args.resume and cell_already_done(run_id):
                print(f"  [{run_id}] SKIP (already done)")
                continue

            try:
                t0 = time.time()
                if zs:
                    result = run_zeroshot(model_key, val_df, test_df, run_id,
                                          max_seq_length=args.max_seq_length)
                else:
                    result = train_one_llm(
                        model_key, prep, inp,
                        train_df, val_df, test_df, run_id,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                        max_seq_length=args.max_seq_length,
                    )
                dt = time.time() - t0
                trm, vm, tm = result["train"], result["val"], result["test"]
                print(f"  [{run_id}] train_qwk={trm.get('qwk', 0):.4f} -> "
                      f"test_qwk={tm['qwk']:.4f}  test_acc={tm['accuracy']:.4f}  "
                      f"raw_w1={tm.get('raw_within1', 0):.4f}  ({dt:.1f}s)")
                row = row_from_metrics(
                    run_id=run_id, prep=prep, inp=inp,
                    model_id=(model_key + "_zeroshot" if zs else model_key),
                    family="llm",
                    train_m=trm, val_m=vm, test_m=tm,
                    best_epoch=result.get("best_epoch", 0), seconds=dt,
                )
                append_row(row)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [{run_id}] FAILED: {type(e).__name__}: {e}")

    print("\n[*] exp08 done")


if __name__ == "__main__":
    main()
