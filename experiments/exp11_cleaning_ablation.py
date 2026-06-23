"""exp11 — format-noise robustness ablation.

Quantifies the effect of the cleaning refinement (stripping zero-width/format/
control characters and bullet markers, plus NFC normalisation) on the classical
champion, to show the residual noise removed by the refinement is negligible — so the
released headline numbers (produced with the pre-refinement cleaning) are robust.

For each dataset variant it fits `clean_ra_tfidf_svr` twice:
  * OLD cleaning  = strip punctuation only            (no invisible strip, no NFC)
  * NEW cleaning  = strip invisibles + NFC + strip punctuation (current preprocess)
and reports test QWK + the delta.

Output: results_stats/cleaning_ablation.csv   (CPU-only)
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
import preprocess as P        # noqa: E402
from evaluate import metrics  # noqa: E402
from models.classical import TFIDFSVR  # noqa: E402

DATASETS = [
    ("full", "dataset.csv", False),
    ("no10c", "dataset_no_10c_biology.csv", False),
]


def _old_clean(text: str) -> str:
    """Pre-refinement clean: NO invisible stripping (the behaviour that produced
    the released headline numbers)."""
    if not text or not isinstance(text, str):
        return ""
    return P.strip_punctuation(text.strip())


def _new_clean(text: str) -> str:
    return P.preprocess(text, "clean")


def _fit_score(train_df, test_df, clean_fn):
    """clean_ra champion: side_a = answer, side_b = reference, both cleaned."""
    tr_a = [clean_fn(t) for t in train_df["Answer"]]
    tr_b = [clean_fn(t) for t in train_df["Reference"]]
    te_a = [clean_fn(t) for t in test_df["Answer"]]
    te_b = [clean_fn(t) for t in test_df["Reference"]]
    m = TFIDFSVR().fit(tr_a, tr_b, train_df["normalized_score"].values)
    pred = m.predict(te_a, te_b)
    qwk = metrics(pred, test_df["score_label"].values,
                  max_scores=test_df["Max Score"].values,
                  true_raw=test_df["Student Score"].values)["qwk"]
    return qwk


def main():
    out_dir = os.path.join(C.PROJECT_ROOT, "results_stats")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for run_name, csv_name, drop0 in DATASETS:
        C.RAW_CSV = os.path.join(C.PROJECT_ROOT, "data", csv_name)
        C.DROP_SCORE_ZERO = drop0
        C.SPLIT_MODE = "random"
        # NB: pass the path explicitly — load_dataframe's default arg is frozen at
        # import, so mutating C.RAW_CSV alone would not take effect here.
        df = data.load_dataframe(C.RAW_CSV)
        tr, _, te = data.split_dataframe(df)
        old_q = _fit_score(tr, te, _old_clean)
        new_q = _fit_score(tr, te, _new_clean)
        rows.append({
            "dataset": run_name,
            "old_qwk": round(old_q, 4), "new_qwk": round(new_q, 4),
            "delta": round(new_q - old_q, 4),
        })
        print(f"  {run_name:10s} old QWK={old_q:.4f}  new QWK={new_q:.4f}  "
              f"delta={new_q-old_q:+.4f}")
    path = os.path.join(out_dir, "cleaning_ablation.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "old_qwk", "new_qwk", "delta"])
        w.writeheader(); w.writerows(rows)
    print(f"\n[exp11] wrote {path}")


if __name__ == "__main__":
    main()
