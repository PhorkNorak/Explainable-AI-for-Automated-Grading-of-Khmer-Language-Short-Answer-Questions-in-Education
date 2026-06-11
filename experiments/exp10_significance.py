"""exp10 — champion point metrics + the random-vs-unseen-question leakage comparison.

Outputs (results_stats/):
  champion_metrics.csv  point metrics (QWK / Cohen kappa / accuracy / macro-F1 / within-1)
                        for each saved champion (classical, RNN, encoder, LLM).
  split_compare.csv     classical champion test QWK, random vs unseen-question split,
                        mean +/- std over config.SEEDS.

Runs locally on CPU. Neural/LLM unseen-question re-runs are HPC follow-ups (see README).
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C            # noqa: E402
import data                   # noqa: E402
from evaluate import metrics  # noqa: E402

CHAMPS = {
    # classical + RNN refreshed under the corrected cleaning; encoder + LLM remain on
    # the prior cleaning (GPU/network — re-run pending).
    "classical": "results/champions/classical_segment_ra_tfidf_svr_cal_895",
    "rnn":       "results/champions/rnn_clean_ra_bilstm_895",
    "encoder":   "results/champions/encoder_clean_qar_dual_gte_maxfeat_1184",
    "llm":       "results/champions/llm_clean_qar_qwen35_4b_909",
}
METRICS = ["qwk", "cohen_kappa", "accuracy", "f1_macro", "raw_within1"]


def _load_predictions(path):
    d = pd.read_csv(path, encoding="utf-8-sig")
    return (d["pred_score"].to_numpy(float), d["true_label"].to_numpy(int),
            d["Max Score"].to_numpy(int), d["true_raw"].to_numpy(int))


def champion_metrics(out_dir):
    rows = []
    for fam, d in CHAMPS.items():
        p = os.path.join(C.PROJECT_ROOT, d, "predictions_test.csv")
        if not os.path.exists(p):
            print(f"[exp10] missing {p}; skipping {fam}"); continue
        ps, tl, ms, tr = _load_predictions(p)
        m = metrics(ps, tl, max_scores=ms, true_raw=tr)
        row = {"family": fam, "n": len(ps)}
        row.update({met: round(float(m[met]), 3) for met in METRICS})
        rows.append(row)
        print(f"  {fam:9s} QWK={row['qwk']} acc={row['accuracy']} "
              f"kappa={row['cohen_kappa']} F1={row['f1_macro']} within1={row['raw_within1']}")
    fields = ["family", "n"] + METRICS
    with open(os.path.join(out_dir, "champion_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return rows


def _fit_classical_qwk(mode, seed):
    from models.classical import TFIDFSVR
    C.RAW_CSV = os.path.join(C.PROJECT_ROOT, "data", "dataset_no_10c_biology.csv")
    C.DROP_SCORE_ZERO = True   # 895 variant
    C.SPLIT_MODE = mode
    df = data.load_dataframe()
    tr, va, te = data.split_dataframe(df, seed=seed)
    trp = data.apply_preprocess(tr, "segment"); tep = data.apply_preprocess(te, "segment")
    a, b = data.build_text_lists(trp, "ra"); ta, tb = data.build_text_lists(tep, "ra")
    m = TFIDFSVR().fit(a, b, trp["normalized_score"].values)
    pred = m.predict(ta, tb)
    return metrics(pred, tep["score_label"].values,
                   max_scores=tep["Max Score"].values,
                   true_raw=tep["Student Score"].values)["qwk"]


def split_compare(out_dir):
    rows = []
    for mode in ("random", "question"):
        qwks = [_fit_classical_qwk(mode, seed) for seed in C.SEEDS]
        rows.append({"split": mode, "mean_qwk": f"{np.mean(qwks):.3f}",
                     "std_qwk": f"{np.std(qwks):.3f}", "n_seeds": len(C.SEEDS),
                     "per_seed": " ".join(f"{q:.3f}" for q in qwks)})
        print(f"  classical [{mode:8s}] QWK mean={np.mean(qwks):.3f} +/- {np.std(qwks):.3f}")
    with open(os.path.join(out_dir, "split_compare.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "mean_qwk", "std_qwk", "n_seeds", "per_seed"])
        w.writeheader(); w.writerows(rows)
    if len(rows) == 2:
        delta = float(rows[0]["mean_qwk"]) - float(rows[1]["mean_qwk"])
        print(f"  >>> leakage gap (random - unseen-question) = {delta:+.3f} QWK")
    return rows


def main():
    out_dir = os.path.join(C.PROJECT_ROOT, "results_stats")
    os.makedirs(out_dir, exist_ok=True)
    print("=== champion point metrics (random-split predictions) ===")
    champion_metrics(out_dir)
    print("\n=== random vs unseen-question split (classical champion, multi-seed) ===")
    split_compare(out_dir)
    print("\n[exp10] wrote results_stats/ -> champion_metrics.csv, split_compare.csv")


if __name__ == "__main__":
    main()
