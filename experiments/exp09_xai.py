"""exp09 — Explainable-AI study across the four model families.

For each family it (1) fits/loads that family's champion model on a common dataset,
(2) builds a uniform ``predict_fn(answer, reference) -> score``, (3) produces SHAP
word attribution as the single unified explanation, and (4) scores the explanation by
its plausibility (overlap of the top SHAP words with the reference answer).

All families are anchored on the SAME dataset/split so their explanations are
directly comparable (RQ5: the accuracy–explainability trade-off).

Local (CPU, no extra deps):   --families classical bilstm
HPC   (GPU):                  --families encoder llm     (needs transformers+GPU / unsloth+peft)

Outputs (under results_xai/<dataset>/):
    faithfulness_leaderboard.csv   one row per family (plausibility, qwk, acc)
    shap_global/<family>.csv       Kumar-style global most-important words
    heatmaps/<family>/*.png        word-importance pictures for sampled answers
    rationales/<family>/*.json     LLM rationale cards (llm family only)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np

# Make the project root importable (same pattern as the other experiments).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C            # noqa: E402
import data                   # noqa: E402
import preprocess             # noqa: E402
from evaluate import metrics  # noqa: E402
from xai.explainers import tokenize_answer, occlusion_importance, shap_importance  # noqa: E402
from xai.plausibility import plausibility        # noqa: E402
from xai.render import render_word_heatmap         # noqa: E402
from xai.render_html import render_word_heatmap_html, render_gallery_html  # noqa: E402


DATASET_CSV = {
    "full": "dataset.csv",
    "no10c": "dataset_no_10c_biology.csv",
}
DATASET_DROP0 = {"full": False, "no10c": False}

# Champion config per family (the best cell of each pillar).
CHAMPIONS = {
    # refreshed champion cells under the corrected cleaning (classical=segment_ra, bilstm=clean_ra)
    "classical": {"preprocess": "segment", "input": "ra"},
    "bilstm":    {"preprocess": "clean",   "input": "ra"},
    "encoder":   {"preprocess": "clean",   "input": "ra", "arch": "dual", "backbone": "gte"},
    "llm":       {"preprocess": "clean",   "input": "qar", "model": "qwen35_4b"},
}

# SHAP-only leaderboard: plausibility is the single quantitative anchor (ERASER
# faithfulness removed). Global most-important words are written separately to
# results_xai/<dataset>/shap_global/<family>.csv (Kumar-style global importance).
LEADERBOARD_HEADER = [
    "family", "champion", "dataset", "test_qwk", "test_accuracy",
    "explainer", "n_explained", "fraction", "plausibility", "seconds",
]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _set_dataset(ds: str):
    C.RAW_CSV = os.path.join(C.PROJECT_ROOT, "data", DATASET_CSV[ds])
    C.DROP_SCORE_ZERO = DATASET_DROP0[ds]


def _sample_test(test_p, n: int, seed: int = 42):
    """Pick ~n test rows, balanced across the 5 score labels."""
    if n <= 0 or n >= len(test_p):
        return list(range(len(test_p)))
    rng = np.random.default_rng(seed)
    per = max(1, n // 5)
    idx = []
    for lab in sorted(test_p["score_label"].unique()):
        pool = test_p.index[test_p["score_label"] == lab].tolist()
        take = min(per, len(pool))
        idx.extend(rng.choice(pool, size=take, replace=False).tolist())
    return sorted(idx)[:n]


def _qwk_acc(predict_fn, test_p, input_fmt):
    """Whole-test QWK/accuracy from predict_fn (consistency check on the champion)."""
    a_col = "Answer_proc"
    preds = []
    for _, row in test_p.iterrows():
        ans = str(row[a_col])
        if input_fmt == "qar":
            ans = str(row["Question_proc"]) + " " + ans
        preds.append(float(predict_fn(ans, str(row["Reference_proc"]))))
    preds = np.asarray(preds)
    m = metrics(preds, test_p["score_label"].values,
                max_scores=test_p["Max Score"].values,
                true_raw=test_p["Student Score"].values)
    return m["qwk"], m["accuracy"]


def _write_global_topwords(word_abs_imp, out_root, family, top=30):
    """Kumar-style global importance: rank answer words by summed |SHAP| across the
    explained answers and write results_xai/<dataset>/shap_global/<family>.csv."""
    out_dir = os.path.join(out_root, "shap_global")
    os.makedirs(out_dir, exist_ok=True)
    ranked = sorted(word_abs_imp.items(), key=lambda kv: kv[1], reverse=True)[:top]
    path = os.path.join(out_dir, f"{family}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "word", "sum_abs_shap"])
        for r, (word, val) in enumerate(ranked, 1):
            w.writerow([r, word, round(float(val), 5)])
    print(f"[exp09] {family}: wrote global top-words -> {path}")
    return path


# ────────────────────────────────────────────────────────────────────────────
# Family champions → predict_fn + native explainer
# ────────────────────────────────────────────────────────────────────────────


def _build_classical(train_df, test_df, cfg):
    """Fit the TF-IDF+SVR champion; return (predict_fn, native_explainer, test_p)."""
    from models.classical import TFIDFSVR

    mode, fmt = cfg["preprocess"], cfg["input"]
    train_p = data.apply_preprocess(train_df, mode)
    test_p = data.apply_preprocess(test_df, mode)
    train_a, train_b = data.build_text_lists(train_p, fmt)
    model = TFIDFSVR()
    model.fit(train_a, train_b, train_p["normalized_score"].values)

    def predict_fn(answer, reference):
        return float(model.predict([answer], [reference])[0])

    def explain(answer, reference):
        # native explanation for classical == occlusion
        return occlusion_importance(predict_fn, answer, reference, mode)

    return predict_fn, explain, test_p, "occlusion"


def _build_bilstm(train_df, val_df, test_df, cfg, out_root, max_epochs=12):
    """Train the BiLSTM+Attention champion in-memory on CPU; expose attention.

    Records a per-epoch train/val/test history and writes the Alaoui-style
    train-vs-test curve to results_xai/<dataset>/curves/bilstm/train_history.png.
    """
    import random
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    import torch.nn as nn
    from models.bilstm import BiLSTMScorer

    # Reproducible: seed init + DataLoader shuffle so faithfulness numbers are stable.
    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)

    mode, fmt = cfg["preprocess"], cfg["input"]
    train_p = data.apply_preprocess(train_df, mode)
    val_p = data.apply_preprocess(val_df, mode)
    test_p = data.apply_preprocess(test_df, mode)

    texts = []
    for _, row in train_p.iterrows():
        a, b = data.build_pair(row, fmt)
        texts += [a, b]
    char2id = data.build_char_vocab(texts)

    train_ds = data.CharPairDataset(train_p, char2id, fmt)
    val_ds = data.CharPairDataset(val_p, char2id, fmt)
    test_ds = data.CharPairDataset(test_p, char2id, fmt)
    train_loader = DataLoader(train_ds, batch_size=C.BILSTM_BATCH, shuffle=True)
    train_infer = DataLoader(train_ds, batch_size=C.BILSTM_BATCH, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=C.BILSTM_BATCH, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=C.BILSTM_BATCH, shuffle=False)

    device = "cpu"
    model = BiLSTMScorer(vocab_size=len(char2id)).to(device)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=C.BILSTM_LR)
    loss_fn = nn.MSELoss()
    fkeys = ["input_ids_a", "attention_mask_a", "input_ids_b", "attention_mask_b"]

    def _infer(loader, df_p):
        model.train(False)
        ps, ls = [], []
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model(**{k: batch[k] for k in fkeys})
                tot += float(loss_fn(out, batch["score"])) * out.shape[0]; n += out.shape[0]
                ps.append(out.cpu().numpy()); ls.append(batch["label"].cpu().numpy())
        m = metrics(np.concatenate(ps), np.concatenate(ls),
                    max_scores=df_p["Max Score"].values, true_raw=df_p["Student Score"].values)
        return m, tot / max(n, 1)

    history = []
    best_qwk, best_state, best_epoch, patience, no_improve = -1e9, None, -1, C.BILSTM_PATIENCE, 0
    for ep in range(1, max_epochs + 1):
        model.train(True)
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(**{k: batch[k] for k in fkeys})
            loss = loss_fn(out, batch["score"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        tr_m, tr_loss = _infer(train_infer, train_p)
        va_m, va_loss = _infer(val_loader, val_p)
        te_m, te_loss = _infer(test_loader, test_p)
        history.append({
            "epoch": ep, "train_loss": tr_loss, "val_loss": va_loss, "test_loss": te_loss,
            **{f"train_{k}": v for k, v in tr_m.items()},
            **{f"test_{k}": v for k, v in te_m.items()},
            **va_m,
        })
        print(f"    [bilstm] epoch {ep:2d}  qwk(tr/va/te)="
              f"{tr_m['qwk']:.3f}/{va_m['qwk']:.3f}/{te_m['qwk']:.3f}")
        if va_m["qwk"] > best_qwk:
            best_qwk, best_epoch = va_m["qwk"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.train(False)

    # Write the train-vs-test curve (same 2x2 plotter as the neural grid cells).
    try:
        import json
        from plot_history import plot_one
        cur_dir = os.path.join(out_root, "curves", "bilstm")
        os.makedirs(cur_dir, exist_ok=True)
        tr_f, _ = _infer(train_infer, train_p)
        va_f, _ = _infer(val_loader, val_p)
        te_f, _ = _infer(test_loader, test_p)
        mpath = os.path.join(cur_dir, "metrics.json")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump({"train": tr_f, "val": va_f, "test": te_f,
                       "best_epoch": best_epoch, "history": history}, f, indent=2)
        if len(history) >= 2:
            plot_one(mpath)
            print(f"    [bilstm] wrote {os.path.join(cur_dir, 'train_history.png')}")
    except Exception as e:
        print(f"    [bilstm] curve plot skipped: {e}")

    max_len = 256

    def _enc(text):
        ids = [char2id.get(ch, 1) for ch in str(text)[:max_len]]
        n = len(ids)
        ids = ids + [0] * (max_len - n)
        mask = [1] * n + [0] * (max_len - n)
        return (torch.tensor(ids).unsqueeze(0), torch.tensor(mask).unsqueeze(0))

    def predict_fn(answer, reference):
        ia, ma = _enc(answer)
        ib, mb = _enc(reference)
        with torch.no_grad():
            out = model(ia, ma, ib, mb)
        return float(out.reshape(-1)[0].item())

    def explain(answer, reference):
        return occlusion_importance(predict_fn, answer, reference, mode)

    return predict_fn, explain, test_p, "occlusion"


# ────────────────────────────────────────────────────────────────────────────
# Encoder champion (GPU): re-fit the GTE dual encoder in-memory, like the BiLSTM
# ────────────────────────────────────────────────────────────────────────────


def _build_encoder(train_df, val_df, test_df, cfg, out_root, max_epochs=8):
    """Re-fit the GTE dual-encoder champion in-memory (GPU) and expose predict_fn.

    The champion ships no weights, so (like the BiLSTM) we re-fit on the same
    split. Faithfulness probes the model's own behaviour, so a representative
    re-fit suffices. Champion cell is input=ra with no max-score feature, so the
    encoder is a plain regressor returning a normalized score in [0, 1] and fits
    the uniform predict_fn(answer, reference) contract directly.
    """
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    import torch.nn as nn
    from transformers import AutoTokenizer
    from models.dual import DualEncoderScorer

    np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    mode, fmt = cfg["preprocess"], cfg["input"]
    backbone = C.TRANSFORMER_BACKBONES[cfg["backbone"]]
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)

    train_p = data.apply_preprocess(train_df, mode)
    val_p = data.apply_preprocess(val_df, mode)
    test_p = data.apply_preprocess(test_df, mode)

    train_loader = DataLoader(data.PairDataset(train_p, tokenizer, fmt),
                              batch_size=C.TXFMR_BATCH, shuffle=True)
    val_loader = DataLoader(data.PairDataset(val_p, tokenizer, fmt),
                            batch_size=C.TXFMR_BATCH, shuffle=False)

    device = C.DEVICE
    model = DualEncoderScorer(backbone, max_feat_dim=0).to(device)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=C.TXFMR_LR)
    loss_fn = nn.MSELoss()
    fkeys = ["input_ids_a", "attention_mask_a", "input_ids_b", "attention_mask_b"]

    def _val_qwk():
        model.train(False); ps = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                ps.append(model(**{k: batch[k] for k in fkeys}).detach().cpu().numpy())
        return metrics(np.concatenate(ps), val_p["score_label"].values,
                       max_scores=val_p["Max Score"].values,
                       true_raw=val_p["Student Score"].values)["qwk"]

    best_qwk, best_state, no_improve = -1e9, None, 0
    for ep in range(1, max_epochs + 1):
        model.train(True)
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(**{k: batch[k] for k in fkeys})
            loss = loss_fn(out, batch["score"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        vq = _val_qwk()
        print(f"    [encoder] epoch {ep:2d}  val_qwk={vq:.3f}")
        if vq > best_qwk:
            best_qwk, no_improve = vq, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= C.TXFMR_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.train(False)

    max_len = C.TXFMR_MAX_LEN

    def _enc(text):
        e = tokenizer(text, max_length=max_len, padding="max_length",
                      truncation=True, return_tensors="pt")
        return e["input_ids"].to(device), e["attention_mask"].to(device)

    def predict_fn(answer, reference):
        ia, ma = _enc(answer); ib, mb = _enc(reference)
        with torch.no_grad():
            out = model(ia, ma, ib, mb)
        return float(out.reshape(-1)[0].item())

    def explain(answer, reference):
        return occlusion_importance(predict_fn, answer, reference, mode)

    return predict_fn, explain, test_p, "occlusion"


# ────────────────────────────────────────────────────────────────────────────
# LLM champion (GPU): load base + fine-tuned LoRA adapter, occlude the answer
# ────────────────────────────────────────────────────────────────────────────


def _load_llm_finetuned(model_key, dataset, adapter_path=None, max_seq_length=1024):
    """Load the base model + the fine-tuned KhmerGrader LoRA adapter for inference.

    Auto-discovers the adapter from the exp08 run dir or a champion dir; falls back
    to the base model (with a warning) if none is found.
    """
    import glob
    import importlib
    L = importlib.import_module("exp08_llm_finetune")
    base = L.LLM_BACKBONES[model_key]
    if adapter_path is None:
        cands = [os.path.join(_ROOT, f"results_{dataset}_v08_llm_{model_key}",
                              "runs", f"clean_qar_{model_key}", "lora_adapter")]
        cands += glob.glob(os.path.join(_ROOT, "results", "champions",
                                        f"llm_clean_qar_{model_key}_*", "lora_adapter"))
        adapter_path = next((p for p in cands if os.path.isdir(p)), None)
    model, tok, _ = L.load_model(base, max_seq_length=max_seq_length, lora=False)
    if adapter_path and os.path.isdir(adapter_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  [llm] loaded fine-tuned adapter: {adapter_path}")
    else:
        print("  [llm] WARNING: no fine-tuned adapter found; faithfulness will be "
              "measured on the BASE model (set --llm-adapter to point at lora_adapter/)")
    model.train(False)
    return L, model, tok


def run_llm_family(family, dataset, test_df, cfg, sample_n, fraction, out_root,
                   adapter_path=None, explainers=("occlusion",), shap_max_evals=None):
    """LOO faithfulness for the fine-tuned LLM.

    The LLM grades from a structured prompt (Question / Reference / Answer / Max),
    so unlike the regressor pillars its score depends on max_score. We therefore
    occlude only the STUDENT ANSWER, hold the question/reference/max fixed, and
    build a per-instance predict_fn (capturing that row's question + max_score)
    that returns a normalized score in [0, 1]. SHAP attribution + plausibility only
    (ERASER faithfulness removed); the LLM answer is occluded, Q/R/Max held fixed.
    """
    t0 = time.time()
    model_key = cfg.get("model", "qwen35_4b")
    mode = cfg["preprocess"]
    try:
        L, model, tok = _load_llm_finetuned(model_key, dataset, adapter_path)
    except Exception as e:
        print(f"[exp09] llm load failed: {type(e).__name__}: {e}")
        return None
    device = next(model.parameters()).device

    test_p = data.apply_preprocess(test_df, mode)

    def score_norm(qproc, refproc, ansproc, max_score):
        row = {"Question_proc": qproc, "Reference_proc": refproc,
               "Answer_proc": ansproc, "Max Score": int(max_score), "Student Score": 0}
        prompt = L.render_prompt_text(tok, row, with_answer=False)
        raw, _ = L.generate_score(model, tok, prompt, int(max_score), device)
        return raw / max(int(max_score), 1)

    # Whole-test QWK consistency check (structured prompt, per-row max_score).
    preds = [score_norm(str(r["Question_proc"]), str(r["Reference_proc"]),
                        str(r["Answer_proc"]), r["Max Score"])
             for _, r in test_p.iterrows()]
    m = metrics(np.asarray(preds), test_p["score_label"].values,
                max_scores=test_p["Max Score"].values,
                true_raw=test_p["Student Score"].values)
    qwk, acc = m["qwk"], m["accuracy"]
    print(f"[exp09] llm champion on {dataset}: test_qwk={qwk:.4f} acc={acc:.4f}")

    idxs = _sample_test(test_p, sample_n)
    heat_dir = os.path.join(out_root, "heatmaps", family)

    def _explain_for(imp_name):
        """One SHAP-only row for the LLM under explainer ``imp_name`` (``shap`` is the
        headline; ``occlusion`` kept only as a debug option). Renders SHAP heatmaps +
        global top-words; plausibility is the single quantitative metric (ERASER removed)."""
        plaus = []
        gabs = defaultdict(float)
        gallery_items, rendered, n_ok = [], 0, 0
        print(f"[exp09] llm [{imp_name}]: explaining {len(idxs)} answers ...")
        for idx in idxs:
            row = test_p.loc[idx]
            Q, R, M = str(row["Question_proc"]), str(row["Reference_proc"]), int(row["Max Score"])

            def pf(answer_proc, reference_proc, _Q=Q, _M=M):  # occlude answer, hold Q/R/Max
                return score_norm(_Q, reference_proc, answer_proc, _M)

            if imp_name == "shap":
                words, imp = shap_importance(pf, str(row["Answer_proc"]), R, mode,
                                             max_evals=shap_max_evals)
            else:
                words, imp = occlusion_importance(pf, str(row["Answer_proc"]), R, mode)
            if len(words) == 0:
                continue
            n_ok += 1
            plaus.append(plausibility(words, imp, R, mode, fraction))
            for w_, v_ in zip(words, imp):
                gabs[w_.strip()] += abs(float(v_))

            if rendered < 8:  # heatmap over the ORIGINAL Khmer answer (SHAP-coloured)
                qraw = str(row["Question"])

                def raw_pf(ans_text, ref_text, _q=qraw, _M=M):
                    return score_norm(preprocess.preprocess(_q, mode),
                                      preprocess.preprocess(ref_text, mode),
                                      preprocess.preprocess(ans_text, mode), _M)

                rw, ri = shap_importance(raw_pf, str(row["Answer"]), str(row["Reference"]),
                                         "raw", max_evals=shap_max_evals)
                if len(rw) > 0:
                    title = (f"{family} | true={int(row['score_label'])} "
                             f"pred={int(round(pf(str(row['Answer_proc']), R) * 4))} "
                             f"| SHAP word attribution on original answer")
                    render_word_heatmap(rw, ri, os.path.join(heat_dir, f"sample_{rendered:02d}.png"), title)
                    render_word_heatmap_html(rw, ri, os.path.join(heat_dir, f"sample_{rendered:02d}.html"), title)
                    gallery_items.append({"words": rw, "importance": ri, "caption": title})
                    rendered += 1

        if gallery_items:
            render_gallery_html(gallery_items, os.path.join(heat_dir, f"{family}_gallery.html"),
                                f"{family} — Khmer word-importance (SHAP, browser-shaped)")
        if n_ok == 0:
            return None
        if imp_name == "shap":
            _write_global_topwords(gabs, out_root, family)
        return {
            "family": family,
            "champion": "_".join(str(v) for v in cfg.values()),
            "dataset": dataset,
            "test_qwk": round(qwk, 4),
            "test_accuracy": round(acc, 4),
            "explainer": imp_name,
            "n_explained": n_ok,
            "fraction": fraction,
            "plausibility": round(float(np.mean(plaus)) if plaus else 0.0, 4),
            "seconds": round(time.time() - t0, 1),
        }

    rows = [r for r in (_explain_for(name) for name in explainers) if r]
    return rows or None


# ────────────────────────────────────────────────────────────────────────────
# Family runner
# ────────────────────────────────────────────────────────────────────────────


def run_family(family, dataset, sample_n, fraction, out_root, adapter_path=None,
               explainers=("occlusion",), shap_max_evals=None):
    cfg = CHAMPIONS[family]
    fmt = cfg["input"]
    df = data.load_dataframe()
    train_df, val_df, test_df = data.split_dataframe(df)
    t0 = time.time()

    if family == "llm":
        return run_llm_family(family, dataset, test_df, cfg, sample_n, fraction,
                              out_root, adapter_path=adapter_path, explainers=explainers,
                              shap_max_evals=shap_max_evals)

    if family == "classical":
        predict_fn, explain, test_p, explainer = _build_classical(train_df, test_df, cfg)
    elif family == "bilstm":
        predict_fn, explain, test_p, explainer = _build_bilstm(train_df, val_df, test_df, cfg, out_root)
    elif family == "encoder":
        predict_fn, explain, test_p, explainer = _build_encoder(train_df, val_df, test_df, cfg, out_root)
    else:
        raise ValueError(family)

    qwk, acc = _qwk_acc(predict_fn, test_p, fmt)
    print(f"[exp09] {family} champion on {dataset}: test_qwk={qwk:.4f} acc={acc:.4f}")

    mode = cfg["preprocess"]
    idxs = _sample_test(test_p, sample_n)

    # Materialise the sampled (row, answer, reference) once.
    inst = []
    for idx in idxs:
        row = test_p.loc[idx]
        ans = str(row["Answer_proc"])
        if fmt == "qar":
            ans = str(row["Question_proc"]) + " " + ans
        inst.append((row, ans, str(row["Reference_proc"])))

    # Heatmaps (once, native model): ORIGINAL Khmer answer coloured by SHAP importance.
    # We emit BOTH a PNG (matplotlib; quick preview, but matplotlib does NOT shape Khmer) and
    # an HTML file (browser-shaped → correct Khmer). Use the HTML/gallery for the deck & paper.
    heat_dir = os.path.join(out_root, "heatmaps", family)
    rendered = 0
    gallery_items = []
    for j, (row, ans, ref) in enumerate(inst):
        if rendered >= 8:
            break
        q_raw = str(row["Question"]) if fmt == "qar" else ""

        def raw_predict(ans_text, ref_text, _m=mode, _q=q_raw):
            a_proc = preprocess.preprocess(ans_text, _m)
            if fmt == "qar":
                a_proc = preprocess.preprocess(_q, _m) + " " + a_proc
            return predict_fn(a_proc, preprocess.preprocess(ref_text, _m))

        rw, ri = shap_importance(raw_predict, str(row["Answer"]), str(row["Reference"]),
                                 "raw", max_evals=shap_max_evals)
        if len(rw) > 0:
            title = (f"{family} | true={int(row['score_label'])} "
                     f"pred={int(round(float(predict_fn(ans, ref)) * 4))} "
                     f"| SHAP word attribution on original answer")
            render_word_heatmap(rw, ri, os.path.join(heat_dir, f"sample_{j:02d}.png"), title)
            render_word_heatmap_html(rw, ri, os.path.join(heat_dir, f"sample_{j:02d}.html"), title)
            gallery_items.append({"words": rw, "importance": ri, "caption": title})
            rendered += 1
    if gallery_items:
        render_gallery_html(gallery_items,
                            os.path.join(heat_dir, f"{family}_gallery.html"),
                            f"{family} — Khmer word-importance (SHAP, browser-shaped)")

    imp_fns = {
        "occlusion": explain,
        "shap": (lambda a, r: shap_importance(predict_fn, a, r, mode, max_evals=shap_max_evals)),
    }
    variants = [(name, imp_fns[name]) for name in explainers if name in imp_fns]

    out_rows = []
    for vname, vfn in variants:
        print(f"[exp09] {family} [{vname}]: explaining {len(inst)} answers ...")
        plaus = []
        gabs = defaultdict(float)   # global sum(|attribution|) per answer word
        n_ok = 0
        for row, ans, ref in inst:
            words, imp = vfn(ans, ref)
            if len(words) == 0:
                continue
            n_ok += 1
            plaus.append(plausibility(words, imp, ref, mode, fraction))
            for w_, v_ in zip(words, imp):
                gabs[w_.strip()] += abs(float(v_))
        if vname == "shap":
            _write_global_topwords(gabs, out_root, family)
        out_rows.append({
            "family": family,
            "champion": "_".join(str(v) for v in cfg.values()),
            "dataset": dataset,
            "test_qwk": round(qwk, 4),
            "test_accuracy": round(acc, 4),
            "explainer": vname,
            "n_explained": n_ok,
            "fraction": fraction,
            "plausibility": round(float(np.mean(plaus)) if plaus else 0.0, 4),
            "seconds": round(time.time() - t0, 1),
        })
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+",
                    default=["classical", "bilstm"],
                    choices=["classical", "bilstm", "encoder", "llm"])
    ap.add_argument("--dataset", default="no10c",
                    choices=["full", "no10c"])
    ap.add_argument("--sample", type=int, default=80,
                    help="number of test answers to explain (balanced by score)")
    ap.add_argument("--fraction", type=float, default=0.2,
                    help="top-k fraction of words for faithfulness/plausibility")
    ap.add_argument("--explainers", nargs="+", default=["shap"],
                    choices=["occlusion", "shap"],
                    help="attribution method(s); SHAP is the headline (Kumar 2020). "
                         "occlusion is kept only as the internal mechanism / a debug option")
    ap.add_argument("--shap-max-evals", type=int, default=None,
                    help="cap SHAP evaluations per answer (lower = faster; for the "
                         "LLM try ~2*num_words). Default: max(2*words+1, 100).")
    ap.add_argument("--llm-adapter", default=None,
                    help="path to the fine-tuned LoRA adapter dir (lora_adapter/); "
                         "auto-discovered from the exp08 run if omitted")
    args = ap.parse_args()

    _set_dataset(args.dataset)
    out_root = os.path.join(C.PROJECT_ROOT, "results_xai", args.dataset)
    os.makedirs(out_root, exist_ok=True)

    rows = []
    for fam in args.families:
        try:
            r = run_family(fam, args.dataset, args.sample, args.fraction, out_root,
                           adapter_path=args.llm_adapter, explainers=tuple(args.explainers),
                           shap_max_evals=args.shap_max_evals)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[exp09] family '{fam}' FAILED: {type(e).__name__}: {e}")
            r = None
        if r:
            rows.extend(r)

    if rows:
        lb = os.path.join(out_root, "faithfulness_leaderboard.csv")
        existing = []
        new_keys = {(r["family"], r["explainer"]) for r in rows}
        if os.path.exists(lb):
            with open(lb, encoding="utf-8") as f:
                existing = [r for r in csv.DictReader(f)
                            if (r["family"], r["explainer"]) not in new_keys]
        with open(lb, "w", newline="", encoding="utf-8") as f:
            # extrasaction="ignore" so any legacy rows (old ERASER columns) re-serialise
            # cleanly under the new plausibility-only header.
            w = csv.DictWriter(f, fieldnames=LEADERBOARD_HEADER, extrasaction="ignore")
            w.writeheader()
            for r in existing:
                w.writerow(r)
            for r in rows:
                w.writerow(r)
        print(f"\n[exp09] wrote {lb}")
        for r in rows:
            print(f"  {r['family']:8s} [{r['explainer']:9s}] n={r['n_explained']:>3} "
                  f"qwk={float(r['test_qwk']):.3f} plausibility={float(r['plausibility']):.3f}")


if __name__ == "__main__":
    main()
