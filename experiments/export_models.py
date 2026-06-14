"""Export publishable best checkpoints for the non-LLM pillars.

Bundles each champion into publish/<ModelName>/ with the weights plus the exact
config/vocab needed to load them, a minimal predict example, and a HuggingFace
model card. These are custom-architecture / scikit-learn models (not AutoModels),
so each card points to the GitHub repo for the model class and full inference code.

Each export runs a sanity QWK check on the test split and prints it, so you can
confirm the published artifact reproduces the champion before uploading.

  python experiments/export_models.py            # classical + bilstm (CPU, local)
  python experiments/export_models.py --encoder   # also the GTE dual encoder (GPU)

The non-LLM pillars (classical, bilstm) are the user's own models trained from
scratch on the Khmer corpus -> MIT. The encoder embeds GTE weights (Apache 2.0),
noted in its card.
"""

import argparse
import importlib
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import config as C
import data
from evaluate import metrics

PUB = os.path.join(_ROOT, "publish")


def _champ_split():
    """The non-LLM champions live on no10c_no0 (895): classical=segment_ra,
    bilstm=clean_ra, encoder=clean_ra. Deterministic (seed 42)."""
    C.RAW_CSV = os.path.join(C.PROJECT_ROOT, "data", "dataset_no_10c_biology.csv")
    C.DROP_SCORE_ZERO = True
    importlib.reload(data)
    df = data.load_dataframe()
    return data.split_dataframe(df)


def _write_card(out, name, arch, license_, desc, deps, usage):
    body = (
        f"---\nlicense: {license_}\nlibrary_name: khmer-asag\ntags:\n"
        f"  - khmer\n  - automated-short-answer-grading\n  - education\n---\n\n"
        f"# {name}\n\n{desc}\n\n"
        f"- Architecture: {arch}\n- Task: Khmer short-answer grading "
        f"(predict a normalized score in [0,1]; raw points = round(score x max_score))\n"
        f"- Training data: a single-trained-teacher Khmer secondary-school corpus "
        f"(no10c_no0 split, 895 answers)\n- {deps}\n\n"
        f"## Usage\n\n```python\n{usage}\n```\n\n"
        f"Part of the **KhmerGrader** benchmark. Model class and full inference "
        f"pipeline (Khmer preprocessing: NFC, strip invisibles, KCC reorder, "
        f"khmernltk segmentation) are in the project repository. Intended for "
        f"assistive grading with a human in the loop, not high-stakes autonomous use.\n"
    )
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(body)


def _qwk_on_test(pred_fn, test_p, fmt):
    preds = []
    for _, row in test_p.iterrows():
        a = str(row["Answer_proc"])
        if fmt == "qar":
            a = str(row["Question_proc"]) + " " + a
        preds.append(float(pred_fn(a, str(row["Reference_proc"]))))
    m = metrics(np.asarray(preds), test_p["score_label"].values,
                max_scores=test_p["Max Score"].values,
                true_raw=test_p["Student Score"].values)
    return m["qwk"]


def export_classical():
    import joblib
    from models.classical import TFIDFSVR
    name = "KhmerGrader-Classical-SVR"
    out = os.path.join(PUB, name)
    os.makedirs(out, exist_ok=True)
    mode, fmt = "segment", "ra"
    tr, _, te = _champ_split()
    trp = data.apply_preprocess(tr, mode)
    tep = data.apply_preprocess(te, mode)
    a, b = data.build_text_lists(trp, fmt)
    model = TFIDFSVR()
    model.fit(a, b, trp["normalized_score"].values)
    joblib.dump(model, os.path.join(out, "model.joblib"))
    json.dump({"preprocess": mode, "input": fmt,
               "output": "normalized_score in [0,1]; raw = round(score * max_score)"},
              open(os.path.join(out, "config.json"), "w"), indent=2)

    def pf(ans, ref):
        return float(model.predict([ans], [ref])[0])
    qwk = _qwk_on_test(pf, tep, fmt)
    _write_card(out, name, "TF-IDF (char_wb 2-4 gram) + RBF-SVR", "mit",
                "Classical Khmer short-answer grader: character TF-IDF features over "
                "the answer and reference, RBF-SVR regression to a normalized score.",
                "Dependencies: scikit-learn (MIT). Trained from scratch (no base model).",
                "import joblib\nm = joblib.load('model.joblib')\n"
                "score = float(m.predict([answer_proc], [reference_proc])[0])  # 0..1\n"
                "raw = round(score * max_score)")
    print(f"[classical] wrote {out}  (test QWK={qwk:.4f}; champion ~0.795)")


def export_bilstm():
    import torch
    from models.bilstm import BiLSTMScorer
    name = "KhmerGrader-BiLSTM"
    out = os.path.join(PUB, name)
    os.makedirs(out, exist_ok=True)
    champ = os.path.join(_ROOT, "results", "champions", "rnn_clean_ra_bilstm_895")
    mode, fmt = "clean", "ra"
    tr, _, te = _champ_split()
    trp = data.apply_preprocess(tr, mode)
    tep = data.apply_preprocess(te, mode)

    # Rebuild the exact char2id from the champion train split (best.pt has no vocab).
    texts = []
    for _, row in trp.iterrows():
        x, y = data.build_pair(row, fmt)
        texts += [x, y]
    char2id = data.build_char_vocab(texts)
    json.dump(char2id, open(os.path.join(out, "char2id.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)
    shutil.copy(os.path.join(champ, "best.pt"), os.path.join(out, "best.pt"))
    shutil.copy(os.path.join(champ, "config.json"), os.path.join(out, "model_config.json"))

    model = BiLSTMScorer(vocab_size=len(char2id))
    state = torch.load(os.path.join(out, "best.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.train(False)
    max_len = 256

    def _enc(text):
        ids = [char2id.get(ch, 1) for ch in str(text)[:max_len]]
        n = len(ids)
        ids = ids + [0] * (max_len - n)
        mask = [1] * n + [0] * (max_len - n)
        return torch.tensor([ids]), torch.tensor([mask])

    def pf(ans, ref):
        ia, ma = _enc(ans); ib, mb = _enc(ref)
        with torch.no_grad():
            return float(model(ia, ma, ib, mb).reshape(-1)[0])
    qwk = _qwk_on_test(pf, tep, fmt)
    _write_card(out, name, "Character BiLSTM + additive attention", "mit",
                "RNN Khmer short-answer grader: character-level BiLSTM with additive "
                "attention over the answer and reference, regressing a normalized score.",
                "Dependencies: PyTorch (BSD). Trained from scratch (no base model). "
                "Load best.pt into BiLSTMScorer(vocab_size=len(char2id)).",
                "import json, torch\nfrom models.bilstm import BiLSTMScorer\n"
                "char2id = json.load(open('char2id.json'))\n"
                "m = BiLSTMScorer(vocab_size=len(char2id))\n"
                "m.load_state_dict(torch.load('best.pt', map_location='cpu')); m.train(False)\n"
                "# encode answer/reference with char2id (0=pad, 1=unk, max_len=256), then m(ids_a, mask_a, ids_b, mask_b)")
    flag = "" if qwk > 0.6 else "  [WARN vocab/order mismatch? expected ~0.845]"
    print(f"[bilstm] wrote {out}  (test QWK={qwk:.4f}; champion ~0.845){flag}")


def export_encoder(max_epochs=8):
    """Re-fit the GTE dual encoder champion and save its weights (GPU)."""
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    import torch.nn as nn
    from transformers import AutoTokenizer
    from models.dual import DualEncoderScorer
    name = "KhmerGrader-Encoder-GTE"
    out = os.path.join(PUB, name)
    os.makedirs(out, exist_ok=True)
    mode, fmt = "clean", "ra"
    backbone = C.TRANSFORMER_BACKBONES["gte"]
    np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    tok = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    tr, va, te = _champ_split()
    trp = data.apply_preprocess(tr, mode)
    vap = data.apply_preprocess(va, mode)
    tep = data.apply_preprocess(te, mode)
    train_loader = DataLoader(data.PairDataset(trp, tok, fmt), batch_size=C.TXFMR_BATCH, shuffle=True)
    val_loader = DataLoader(data.PairDataset(vap, tok, fmt), batch_size=C.TXFMR_BATCH, shuffle=False)
    device = C.DEVICE
    model = DualEncoderScorer(backbone, max_feat_dim=0).to(device)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=C.TXFMR_LR)
    loss_fn = nn.MSELoss()
    fk = ["input_ids_a", "attention_mask_a", "input_ids_b", "attention_mask_b"]

    def _vq():
        model.train(False); ps = []
        with torch.no_grad():
            for b in val_loader:
                b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
                ps.append(model(**{k: b[k] for k in fk}).detach().cpu().numpy())
        return metrics(np.concatenate(ps), vap["score_label"].values,
                       max_scores=vap["Max Score"].values, true_raw=vap["Student Score"].values)["qwk"]

    best_q, best_state, bad = -1e9, None, 0
    for ep in range(1, max_epochs + 1):
        model.train(True)
        for b in train_loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(**{k: b[k] for k in fk}), b["score"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        vq = _vq()
        print(f"  [encoder] epoch {ep:2d} val_qwk={vq:.3f}")
        if vq > best_q:
            best_q, bad = vq, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= C.TXFMR_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.train(False)
    torch.save(model.state_dict(), os.path.join(out, "encoder.pt"))
    tok.save_pretrained(out)
    json.dump({"backbone": backbone, "preprocess": mode, "input": fmt,
               "max_feat_dim": 0, "max_len": C.TXFMR_MAX_LEN,
               "output": "normalized_score in [0,1]"},
              open(os.path.join(out, "config.json"), "w"), indent=2)
    max_len = C.TXFMR_MAX_LEN

    def _enc(text):
        e = tok(text, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        return e["input_ids"].to(device), e["attention_mask"].to(device)

    def pf(ans, ref):
        ia, ma = _enc(ans); ib, mb = _enc(ref)
        with torch.no_grad():
            return float(model(ia, ma, ib, mb).reshape(-1)[0])
    qwk = _qwk_on_test(pf, tep, fmt)
    _write_card(out, name, "GTE-multilingual dual encoder + MLP head", "apache-2.0",
                "Transformer Khmer short-answer grader: a shared GTE-multilingual encoder "
                "over answer and reference, 4-way interaction, MLP head to a normalized score.",
                "Base model: Alibaba-NLP/gte-multilingual-base (Apache 2.0); weights here "
                "are fine-tuned from it, so Apache 2.0 attribution applies. Load encoder.pt "
                "into DualEncoderScorer(backbone, max_feat_dim=0).",
                "import torch\nfrom models.dual import DualEncoderScorer\n"
                "m = DualEncoderScorer('Alibaba-NLP/gte-multilingual-base', max_feat_dim=0)\n"
                "m.load_state_dict(torch.load('encoder.pt', map_location='cpu')); m.train(False)\n"
                "# tokenize answer & reference (max_len=256) with the bundled tokenizer, then m(ids_a, mask_a, ids_b, mask_b)")
    print(f"[encoder] wrote {out}  (test QWK={qwk:.4f}; champion ~0.820)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", action="store_true", help="also export the GTE dual encoder (GPU)")
    ap.add_argument("--only", choices=["classical", "bilstm", "encoder"], default=None)
    args = ap.parse_args()
    os.makedirs(PUB, exist_ok=True)
    if args.only:
        {"classical": export_classical, "bilstm": export_bilstm, "encoder": export_encoder}[args.only]()
    else:
        export_classical()
        export_bilstm()
        if args.encoder:
            export_encoder()
    print(f"\n[*] publish-ready folders under {PUB}/")


if __name__ == "__main__":
    main()
