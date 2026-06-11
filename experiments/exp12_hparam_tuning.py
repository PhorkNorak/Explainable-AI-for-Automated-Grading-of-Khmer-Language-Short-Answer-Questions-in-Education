"""exp12 — classical hyperparameter tuning (validation-selected).

Standard practice: search a small grid and select the configuration with the best
*validation* QWK, then report its *test* QWK (no test-set peeking). For the classical
champion (TF-IDF + RBF-SVR), the key hyperparameters are the SVR penalty **C** (the
analog of a neural learning rate) and the TF-IDF **max-features**. Neural/LLM/transformer
learning-rate sweeps use the same val-selection rule and run on HPC (see README).

Output: results_stats/hparam_tuning.csv   (CPU-only; classical champion config segment_ra)
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C            # noqa: E402
import data                   # noqa: E402
from evaluate import metrics  # noqa: E402
from models.classical import TFIDFSVR  # noqa: E402

# Champion classical config + search grid.
PREP, INP = "segment", "ra"
C_GRID = [0.1, 1.0, 10.0]
MAXFEAT_GRID = [5000, 15000, 30000]
DATASETS = [("no10c_no0", "dataset_no_10c_biology.csv", True),
            ("no10c", "dataset_no_10c_biology.csv", False),
            ("full", "dataset.csv", False)]


def _prep(df, mode):
    p = data.apply_preprocess(df, mode)
    return data.build_text_lists(p, INP), p


def _fit(train_p_lists, train_p, test_lists, test_p, c, max_feat):
    C.TFIDF_MAX_FEAT = max_feat          # TFIDFSVR reads this at construction
    m = TFIDFSVR(C_svr=c)
    (tr_a, tr_b) = train_p_lists
    m.fit(tr_a, tr_b, train_p["normalized_score"].values)
    (te_a, te_b) = test_lists
    pred = m.predict(te_a, te_b)
    return metrics(pred, test_p["score_label"].values,
                   max_scores=test_p["Max Score"].values,
                   true_raw=test_p["Student Score"].values)["qwk"]


def main():
    out_dir = os.path.join(C.PROJECT_ROOT, "results_stats")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for run_name, csv_name, drop0 in DATASETS:
        C.RAW_CSV = os.path.join(C.PROJECT_ROOT, "data", csv_name)
        C.DROP_SCORE_ZERO = drop0
        C.SPLIT_MODE = "random"
        df = data.load_dataframe(C.RAW_CSV)
        tr, va, te = data.split_dataframe(df)
        tr_lists, trp = _prep(tr, PREP)
        va_lists, vap = _prep(va, PREP)
        te_lists, tep = _prep(te, PREP)

        best = None
        for c in C_GRID:
            for mf in MAXFEAT_GRID:
                val_qwk = _fit(tr_lists, trp, va_lists, vap, c, mf)
                if best is None or val_qwk > best[0]:
                    best = (val_qwk, c, mf)
        # default config QWK (C=1, max_feat=15000) for the "tuned vs default" comparison
        def_qwk_test = _fit(tr_lists, trp, te_lists, tep, 1.0, 15000)
        best_val, best_c, best_mf = best
        best_qwk_test = _fit(tr_lists, trp, te_lists, tep, best_c, best_mf)
        rows.append({"dataset": run_name, "best_C": best_c, "best_max_feat": best_mf,
                     "val_qwk": round(best_val, 4),
                     "tuned_test_qwk": round(best_qwk_test, 4),
                     "default_test_qwk(C1,15k)": round(def_qwk_test, 4),
                     "delta_vs_default": round(best_qwk_test - def_qwk_test, 4)})
        print(f"  {run_name:10s} best C={best_c} max_feat={best_mf} "
              f"val={best_val:.4f} tuned_test={best_qwk_test:.4f} "
              f"default_test={def_qwk_test:.4f} (Δ{best_qwk_test-def_qwk_test:+.4f})")
    path = os.path.join(out_dir, "hparam_tuning.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[exp12] wrote {path}")


if __name__ == "__main__":
    main()
