"""exp10 — champion point metrics.

Outputs (results_stats/):
  champion_metrics.csv  point metrics (QWK / Cohen kappa / accuracy / macro-F1 / within-1)
                        for each saved champion (classical, RNN, encoder, LLM).

Runs locally on CPU from the saved champion predictions; needs no re-training.
"""

from __future__ import annotations

import csv
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C            # noqa: E402
from evaluate import metrics  # noqa: E402

CHAMPS = {
    # NB: paths must be reconciled with the actual champion dirs the no10c re-run
    # produces (the winning preprocess/input may shift). Default variant is now no10c (909).
    "classical": "results/champions/classical_segment_ra_tfidf_svr_909",
    "rnn":       "results/champions/rnn_clean_ra_bilstm_909",
    "encoder":   "results/champions/encoder_clean_qar_dual_gte_maxfeat_1184",
    "llm":       "results/champions/llm_clean_qar_qwen35_4b_909",
    # Qwen zero-shot base -> the headline tab:champs reference row (untrained baseline).
    # Skipped automatically if predictions_test.csv is absent (exp08 --zeroshot writes it).
    "llm_zeroshot": "results/champions/zeroshot_qar_qwen35_4b_909",
}
METRICS = ["qwk", "cohen_kappa", "accuracy",
           "precision_macro", "recall_macro", "f1_macro", "raw_within1"]


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
              f"kappa={row['cohen_kappa']} P={row['precision_macro']} R={row['recall_macro']} "
              f"F1={row['f1_macro']} within1={row['raw_within1']}")
    fields = ["family", "n"] + METRICS
    with open(os.path.join(out_dir, "champion_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return rows


def main():
    out_dir = os.path.join(C.PROJECT_ROOT, "results_stats")
    os.makedirs(out_dir, exist_ok=True)
    print("=== champion point metrics (random-split predictions) ===")
    champion_metrics(out_dir)
    print("\n[exp10] wrote results_stats/ -> champion_metrics.csv")


if __name__ == "__main__":
    main()
