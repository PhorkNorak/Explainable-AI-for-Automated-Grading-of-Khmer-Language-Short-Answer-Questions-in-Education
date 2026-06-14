"""exp09 — Explainable-AI study across the four model families.

For each family it (1) fits/loads that family's champion model on a common dataset,
(2) builds a uniform ``predict_fn(answer, reference) -> score``, (3) produces LOO
(Leave-One-Out occlusion) word attribution as the single unified explanation, and
(4) scores the explanation with model-agnostic faithfulness (ERASER comprehensiveness
& sufficiency, vs a random-removal baseline) and a reference-overlap plausibility proxy.

All families are anchored on the SAME dataset/split so their explanations are
directly comparable (RQ5: the accuracy–explainability trade-off).

Local (CPU, no extra deps):   --families classical bilstm
HPC   (GPU):                  --families encoder llm     (needs transformers+GPU / unsloth+peft)

Outputs (under results_xai/<dataset>/):
    faithfulness_leaderboard.csv   one row per family (comp, suff, gap, plausibility, qwk, acc)
    heatmaps/<family>/*.png        word-importance pictures for sampled answers
    rationales/<family>/*.json     LLM rationale cards (llm family only)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

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
from xai.explainers import tokenize_answer, occlusion_importance  # noqa: E402
from xai.faithfulness import faithfulness_report  # noqa: E402
from xai.plausibility import plausibility        # noqa: E402
from xai.render import render_word_heatmap         # noqa: E402
from xai.render_html import render_word_heatmap_html, render_gallery_html  # noqa: E402


DATASET_CSV = {
    "full": "dataset.csv",
    "no10c": "dataset_no_10c_biology.csv",
    "no10c_no0": "dataset_no_10c_biology.csv",
}
DATASET_DROP0 = {"full": False, "no10c": False, "no10c_no0": True}

# Champion config per family (the best cell of each pillar).
CHAMPIONS = {
    # refreshed champion cells under the corrected cleaning (classical=segment_ra, bilstm=clean_ra)
    "classical": {"preprocess": "segment", "input": "ra"},
    "bilstm":    {"preprocess": "clean",   "input": "ra"},
    "encoder":   {"preprocess": "clean",   "input": "ra", "arch": "dual", "backbone": "gte"},
    "llm":       {"preprocess": "clean",   "input": "qar", "model": "qwen35_4b"},
}

LEADERBOARD_HEADER = [
    "family", "champion", "dataset", "test_qwk", "test_accuracy",
    "explainer", "n_explained", "fraction",
    "comprehensiveness", "sufficiency", "comprehensiveness_random",
    "faithfulness_gap", "aopc_comprehensiveness", "aopc_sufficiency",
    "plausibility", "seconds",
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
                   adapter_path=None):
    """LOO faithfulness for the fine-tuned LLM.

    The LLM grades from a structured prompt (Question / Reference / Answer / Max),
    so unlike the regressor pillars its score depends on max_score. We therefore
    occlude only the STUDENT ANSWER, hold the question/reference/max fixed, and
    build a per-instance predict_fn (capturing that row's question + max_score)
    that returns a normalized score in [0, 1]. Aggregation matches faithfulness.py
    (ERASER comprehensiveness/sufficiency + AOPC vs a random-removal baseline).
    """
    from xai.faithfulness import (comprehensiveness, sufficiency,
                                  comprehensiveness_random, DEFAULT_KGRID)
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
    comp, suff, comp_rand, aopc_c, aopc_s, plaus = [], [], [], [], [], []
    heat_dir = os.path.join(out_root, "heatmaps", family)
    gallery_items, rendered = [], 0

    for idx in idxs:
        row = test_p.loc[idx]
        Q, R, M = str(row["Question_proc"]), str(row["Reference_proc"]), int(row["Max Score"])

        def pf(answer_proc, reference_proc, _Q=Q, _M=M):  # occlude answer, hold Q/R/Max
            return score_norm(_Q, reference_proc, answer_proc, _M)

        words, imp = occlusion_importance(pf, str(row["Answer_proc"]), R, mode)
        if len(words) == 0:
            continue
        comp.append(comprehensiveness(pf, words, imp, R, mode, fraction))
        suff.append(sufficiency(pf, words, imp, R, mode, fraction))
        comp_rand.append(comprehensiveness_random(pf, words, R, mode, fraction))
        aopc_c.append(float(np.mean([comprehensiveness(pf, words, imp, R, mode, k) for k in DEFAULT_KGRID])))
        aopc_s.append(float(np.mean([sufficiency(pf, words, imp, R, mode, k) for k in DEFAULT_KGRID])))
        plaus.append(plausibility(words, imp, R, mode, fraction))

        if rendered < 8:  # heatmap over the ORIGINAL Khmer answer (readable tokens)
            qraw = str(row["Question"])

            def raw_pf(ans_text, ref_text, _q=qraw, _M=M):
                return score_norm(preprocess.preprocess(_q, mode),
                                  preprocess.preprocess(ref_text, mode),
                                  preprocess.preprocess(ans_text, mode), _M)

            rw, ri = occlusion_importance(raw_pf, str(row["Answer"]), str(row["Reference"]), "raw")
            if len(rw) > 0:
                title = (f"{family} | true={int(row['score_label'])} "
                         f"pred={int(round(pf(str(row['Answer_proc']), R) * 4))} "
                         f"| word attribution on original answer")
                render_word_heatmap(rw, ri, os.path.join(heat_dir, f"sample_{rendered:02d}.png"), title)
                render_word_heatmap_html(rw, ri, os.path.join(heat_dir, f"sample_{rendered:02d}.html"), title)
                gallery_items.append({"words": rw, "importance": ri, "caption": title})
                rendered += 1

    if gallery_items:
        render_gallery_html(gallery_items, os.path.join(heat_dir, f"{family}_gallery.html"),
                            f"{family} — Khmer word-importance (occlusion, browser-shaped)")
    if not comp:
        return None
    comp_m, rand_m = float(np.mean(comp)), float(np.mean(comp_rand))
    return [{
        "family": family,
        "champion": "_".join(str(v) for v in cfg.values()),
        "dataset": dataset,
        "test_qwk": round(qwk, 4),
        "test_accuracy": round(acc, 4),
        "explainer": "occlusion",
        "n_explained": len(comp),
        "fraction": fraction,
        "comprehensiveness": round(comp_m, 4),
        "sufficiency": round(float(np.mean(suff)), 4),
        "comprehensiveness_random": round(rand_m, 4),
        "faithfulness_gap": round(comp_m - rand_m, 4),
        "aopc_comprehensiveness": round(float(np.mean(aopc_c)), 4),
        "aopc_sufficiency": round(float(np.mean(aopc_s)), 4),
        "plausibility": round(float(np.mean(plaus)) if plaus else 0.0, 4),
        "seconds": round(time.time() - t0, 1),
    }]


# ────────────────────────────────────────────────────────────────────────────
# Family runner
# ────────────────────────────────────────────────────────────────────────────


def run_family(family, dataset, sample_n, fraction, out_root, adapter_path=None):
    cfg = CHAMPIONS[family]
    fmt = cfg["input"]
    df = data.load_dataframe()
    train_df, val_df, test_df = data.split_dataframe(df)
    t0 = time.time()

    if family == "llm":
        return run_llm_family(family, dataset, test_df, cfg, sample_n, fraction,
                              out_root, adapter_path=adapter_path)

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

    # Heatmaps (once, native model): ORIGINAL Khmer answer coloured by occlusion importance.
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

        rw, ri = occlusion_importance(raw_predict, str(row["Answer"]), str(row["Reference"]), "raw")
        if len(rw) > 0:
            title = (f"{family} | true={int(row['score_label'])} "
                     f"pred={int(round(float(predict_fn(ans, ref)) * 4))} "
                     f"| word attribution on original answer")
            render_word_heatmap(rw, ri, os.path.join(heat_dir, f"sample_{j:02d}.png"), title)
            render_word_heatmap_html(rw, ri, os.path.join(heat_dir, f"sample_{j:02d}.html"), title)
            gallery_items.append({"words": rw, "importance": ri, "caption": title})
            rendered += 1
    if gallery_items:
        render_gallery_html(gallery_items,
                            os.path.join(heat_dir, f"{family}_gallery.html"),
                            f"{family} — Khmer word-importance (occlusion, browser-shaped)")

    variants = [("occlusion", explain)]

    out_rows = []
    for vname, vfn in variants:
        per_instance, plaus = [], []
        for row, ans, ref in inst:
            words, imp = vfn(ans, ref)
            if len(words) == 0:
                continue
            per_instance.append((words, imp, ref))
            plaus.append(plausibility(words, imp, ref, mode, fraction))
        fr = faithfulness_report(predict_fn, per_instance, mode, fraction)
        out_rows.append({
            "family": family,
            "champion": "_".join(str(v) for v in cfg.values()),
            "dataset": dataset,
            "test_qwk": round(qwk, 4),
            "test_accuracy": round(acc, 4),
            "explainer": vname,
            "n_explained": fr.get("n", 0),
            "fraction": fraction,
            "comprehensiveness": round(fr.get("comprehensiveness", 0.0), 4),
            "sufficiency": round(fr.get("sufficiency", 0.0), 4),
            "comprehensiveness_random": round(fr.get("comprehensiveness_random", 0.0), 4),
            "faithfulness_gap": round(fr.get("faithfulness_gap", 0.0), 4),
            "aopc_comprehensiveness": round(fr.get("aopc_comprehensiveness", 0.0), 4),
            "aopc_sufficiency": round(fr.get("aopc_sufficiency", 0.0), 4),
            "plausibility": round(float(np.mean(plaus)) if plaus else 0.0, 4),
            "seconds": round(time.time() - t0, 1),
        })
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+",
                    default=["classical", "bilstm"],
                    choices=["classical", "bilstm", "encoder", "llm"])
    ap.add_argument("--dataset", default="no10c_no0",
                    choices=["full", "no10c", "no10c_no0"])
    ap.add_argument("--sample", type=int, default=80,
                    help="number of test answers to explain (balanced by score)")
    ap.add_argument("--fraction", type=float, default=0.2,
                    help="top-k fraction of words for faithfulness/plausibility")
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
                           adapter_path=args.llm_adapter)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[exp09] family '{fam}' FAILED: {type(e).__name__}: {e}")
            r = None
        if r:
            rows.extend(r)

    if rows:
        lb = os.path.join(out_root, "faithfulness_leaderboard.csv")
        existing = []
        if os.path.exists(lb):
            with open(lb, encoding="utf-8") as f:
                existing = [r for r in csv.DictReader(f) if r["family"] not in args.families]
        with open(lb, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LEADERBOARD_HEADER)
            w.writeheader()
            for r in existing:
                w.writerow(r)
            for r in rows:
                w.writerow(r)
        print(f"\n[exp09] wrote {lb}")
        for r in rows:
            print(f"  {r['family']:8s} [{r['explainer']:9s}] gap={r['faithfulness_gap']:+.3f} "
                  f"AOPC-comp={r['aopc_comprehensiveness']:+.3f} "
                  f"AOPC-suff={r['aopc_sufficiency']:+.3f} plaus={r['plausibility']:.3f}")


if __name__ == "__main__":
    main()
