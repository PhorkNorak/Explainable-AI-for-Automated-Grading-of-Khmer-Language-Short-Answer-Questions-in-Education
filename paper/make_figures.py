"""Render the full figure suite for the thesis/paper from the real result files.

No GPU, no re-training, no network. Reads only CSVs and champion prediction files
already on disk and writes PNGs into paper/figures/ (copied to thesis/figures/).

Groups:
  A dataset        data/dataset.csv                          (label/subject/max-score/length)
  B accuracy       results_stats/champion_metrics.csv        (QWK + multi-metric)
  C deployment     results/champions/*/predictions_test.csv  (exact/within-1, confusion, scatter)
  D explainability SHAP plausibility is a table, not a figure; heatmaps screenshotted separately
  E ablations      cleaning_ablation / hparam_tuning / calibration

Every value traces to a result file; nothing is hardcoded.

Run:  python paper/make_figures.py
"""

import os
import shutil

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
FIG = os.path.join(_HERE, "figures")
THESIS_FIG = os.path.join(_ROOT, "thesis", "figures")
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------- style
# Consistent pillar order, labels, and colors across every comparison figure.
PILLAR_ORDER = ["classical", "rnn", "encoder", "llm"]
PILLAR_LABELS = {"classical": "Classical", "rnn": "RNN", "encoder": "Transformer", "llm": "LLM"}
PILLAR_COLORS = {"classical": "#4c72b0", "rnn": "#dd8452", "encoder": "#55a868", "llm": "#c44e52"}

# The four pillar champions and their test-set prediction files.
CHAMP_PREDS = {
    "classical": "results/champions/classical_segment_ra_tfidf_svr_909/predictions_test.csv",
    "rnn":       "results/champions/rnn_clean_ra_bilstm_909/predictions_test.csv",
    "encoder":   "results/champions/encoder_clean_qar_dual_gte_maxfeat_1184/predictions_test.csv",
    "llm":       "results/champions/llm_clean_qar_qwen35_4b_909/predictions_test.csv",
}

# KhmerGrader LLM family: internal key -> released name. Used by fig_llm_finetune_gain.
LLM_KEYS = [("qwen35_4b", "Qwen-\nKhmerGrader-4B"),
            ("gemma4_e4b", "Gemma-\nKhmerGrader-4B"),
            ("sealion_v45_e2b", "SEA-LION-\nKhmerGrader-E2B")]
LLM_DATASETS = ["no10c", "full"]   # preference order for a fair pairing

GREEN = "#2f855a"
RED = "#c53030"
SERIES_A = "#4c72b0"   # first series in any 2-series grouped bar
SERIES_B = "#dd8452"   # second series

_written = []


def _rd(path):
    """Read a result CSV (utf-8-sig strips the BOM some files carry)."""
    return pd.read_csv(os.path.join(_ROOT, path), encoding="utf-8-sig")


def _save(fig, name):
    out = os.path.join(FIG, name)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _written.append(name)
    print("  wrote", name)


# ============================================================== Group A: dataset
def _load_full_corpus():
    """The full 1184-answer corpus, label logic mirrored from data.load_dataframe
    (DROP_SCORE_ZERO=False). Kept inline so this script needs no torch/khmernltk."""
    df = pd.read_csv(os.path.join(_ROOT, "data", "dataset.csv"), encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["Subject"] = df["Subject"].astype(str).str.strip().replace({"History ": "History"})
    df = df.dropna(subset=["Question", "Reference", "Answer"]).reset_index(drop=True)
    norm = df["Student Score"] / df["Max Score"]
    df["score_label"] = (norm * 4).round().clip(0, 4).astype(int)
    return df


def fig_label_dist():
    df = _load_full_corpus()
    counts = df["score_label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    bars = ax.bar(counts.index, counts.values, color="#4c72b0", width=0.65)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, str(int(v)), ha="center", fontsize=9)
    ax.set_xlabel("Ordinal grade label (round(4 x score / max))")
    ax.set_ylabel("Number of answers")
    ax.set_title(f"Grade label distribution (n={len(df)}, skewed to full credit)")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_label_dist.png")


def fig_subject_dist():
    df = _load_full_corpus()
    counts = df["Subject"].value_counts()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(range(len(counts)), counts.values, color="#55a868", width=0.6)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, str(int(v)), ha="center", fontsize=9)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, fontsize=9)
    ax.set_ylabel("Number of answers")
    ax.set_title("Answers per subject")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_subject_dist.png")


def fig_maxscore_dist():
    df = _load_full_corpus()
    counts = df["Max Score"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(range(len(counts)), counts.values, color="#dd8452", width=0.6)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, str(int(v)), ha="center", fontsize=9)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(int(s)) for s in counts.index], fontsize=9)
    ax.set_xlabel("Question max score")
    ax.set_ylabel("Number of answers")
    ax.set_title("Answers per question max score")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_maxscore_dist.png")


def fig_answer_length():
    df = _load_full_corpus()
    lengths = df["Answer"].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hist(lengths, bins=30, color="#8172b3", edgecolor="white")
    med = lengths.median()
    ax.axvline(med, color=RED, ls="--", lw=1.2, label=f"median {int(med)} chars")
    ax.set_xlabel("Answer length (characters)")
    ax.set_ylabel("Number of answers")
    ax.set_title("Answer length distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_answer_length.png")


def write_dataset_descriptive():
    """Emit the raw counts behind the dataset figures as a tidy CSV for a LaTeX table."""
    df = _load_full_corpus()
    lengths = df["Answer"].astype(str).str.len()
    rows = [("total", "answers", len(df)),
            ("total", "questions", df["QuestionID"].nunique()),
            ("total", "students", df["StudentID"].nunique()),
            ("total", "subjects", df["Subject"].nunique())]
    for lab, c in df["score_label"].value_counts().sort_index().items():
        rows.append(("label", str(int(lab)), int(c)))
    for sub, c in df["Subject"].value_counts().items():
        rows.append(("subject", sub, int(c)))
    for ms, c in df["Max Score"].value_counts().sort_index().items():
        rows.append(("max_score", str(int(ms)), int(c)))
    for stat, val in [("min", lengths.min()), ("q1", lengths.quantile(.25)),
                      ("median", lengths.median()), ("q3", lengths.quantile(.75)),
                      ("max", lengths.max()), ("mean", round(lengths.mean(), 1))]:
        rows.append(("answer_len_chars", stat, val))
    out = os.path.join(_ROOT, "results_stats", "dataset_descriptive.csv")
    pd.DataFrame(rows, columns=["category", "key", "value"]).to_csv(out, index=False)
    print("  wrote results_stats/dataset_descriptive.csv")


# ============================================================== Group B: accuracy
def _champion_metrics():
    df = _rd("results_stats/champion_metrics.csv").set_index("family")
    return df.reindex(PILLAR_ORDER)


def _zeroshot_qwk():
    """QWK of the Qwen zero-shot base from champion_metrics.csv (None if absent),
    used to draw the fine-tuning-lift reference line on fig_qwk_pillars."""
    try:
        df = _rd("results_stats/champion_metrics.csv")
        row = df[df["family"] == "llm_zeroshot"]
        return float(row["qwk"].iloc[0]) if len(row) else None
    except Exception:
        return None


def fig_qwk_pillars():
    df = _champion_metrics()
    vals = df["qwk"].values.astype(float)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    bars = ax.bar(range(len(vals)), vals,
                  color=[PILLAR_COLORS[p] for p in PILLAR_ORDER], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    lo, hi = vals.min(), vals.max()
    ax.axhline(lo, color="gray", ls=":", lw=0.9)
    ax.axhline(hi, color="gray", ls=":", lw=0.9)
    ax.text(-0.4, 0.93, f"QWK band {hi - lo:.3f}", ha="left", fontsize=9, color="gray")
    # Qwen zero-shot reference line: the fine-tuning lift, visible in the headline figure.
    zs = _zeroshot_qwk()
    if zs is not None:
        ax.axhline(zs, color=RED, ls="--", lw=1.1)
        ax.text(len(vals) - 0.55, zs + 0.012, f"Qwen zero-shot {zs:.2f}",
                ha="right", fontsize=8, color=RED)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([PILLAR_LABELS[p] for p in PILLAR_ORDER])
    ax.set_ylabel("Test QWK")
    ax.set_ylim(0, 1.0)
    ax.set_title("Primary metric: QWK comparable across pillars (narrow band)")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_qwk_pillars.png")


def _train_qwk_acc(fam):
    """Train QWK and accuracy for a pillar (test comes from champion_metrics.csv).
    Sources: champion metrics.json for rnn/encoder/llm; the uncalibrated champion
    leaderboard row (segment_ra) for classical, since its champion dir is calibrated."""
    import json
    mj = {
        "rnn":     "results/champions/rnn_clean_ra_bilstm_909/metrics.json",
        "encoder": "results/champions/encoder_clean_qar_dual_gte_maxfeat_1184/metrics.json",
        "llm":     "results/champions/llm_clean_qar_qwen35_4b_909/metrics.json",
    }
    if fam in mj:
        d = json.load(open(os.path.join(_ROOT, mj[fam]), encoding="utf-8"))
        return float(d["train"]["qwk"]), float(d["train"]["accuracy"])
    # classical: uncalibrated champion = segment_ra in v03_maxfeat
    df = _rd("results/leaderboards/no10c_v03_maxfeat.csv")
    row = df[df["run_id"] == "segment_ra_tfidf_svr_maxfeat"].iloc[0]
    return float(row["train_qwk"]), float(row["train_accuracy"])


def fig_metrics_grouped():
    """Train vs test QWK and accuracy by pillar (generalization gap; no Cohen kappa,
    no macro-F1 since train F1 is not saved for most pillars)."""
    test = _champion_metrics()
    qwk_tr, qwk_te, acc_tr, acc_te = [], [], [], []
    for p in PILLAR_ORDER:
        trq, tra = _train_qwk_acc(p)
        qwk_tr.append(trq); qwk_te.append(float(test.loc[p, "qwk"]))
        acc_tr.append(tra); acc_te.append(float(test.loc[p, "accuracy"]))
    x = np.arange(len(PILLAR_ORDER))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, (tr, te, title) in zip(
            axes, [(qwk_tr, qwk_te, "QWK"), (acc_tr, acc_te, "Accuracy")]):
        ax.bar(x - w / 2, tr, w, label="Train", color=SERIES_A)
        ax.bar(x + w / 2, te, w, label="Test", color=SERIES_B)
        for i in range(len(x)):
            ax.text(x[i] - w / 2, tr[i] + 0.01, f"{tr[i]:.2f}", ha="center", fontsize=7)
            ax.text(x[i] + w / 2, te[i] + 0.01, f"{te[i]:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([PILLAR_LABELS[p] for p in PILLAR_ORDER], fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8)
    fig.suptitle("Train vs test by pillar (generalization gap)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig_metrics_grouped.png")


def fig_prf_pillars():
    """Macro precision / recall / F1 per pillar (champion_metrics.csv). These three
    are otherwise table-only; the figure shows the LLM's class-balance advantage."""
    df = _champion_metrics()
    cols = [("precision_macro", "Precision"), ("recall_macro", "Recall"),
            ("f1_macro", "macro-F1")]
    x = np.arange(len(PILLAR_ORDER))
    w = 0.26
    colors = [SERIES_A, SERIES_B, "#55a868"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for j, (col, lab) in enumerate(cols):
        vals = df[col].values.astype(float)
        bars = ax.bar(x + (j - 1) * w, vals, w, label=lab, color=colors[j])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([PILLAR_LABELS[p] for p in PILLAR_ORDER])
    ax.set_ylabel("Macro-averaged score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Macro precision / recall / F1 by pillar (LLM leads on rare grades)")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_prf_pillars.png")


# ============================================================== Group C: deployment
def _load_preds():
    out = {}
    for fam, rel in CHAMP_PREDS.items():
        p = os.path.join(_ROOT, rel)
        if os.path.exists(p):
            out[fam] = pd.read_csv(p, encoding="utf-8-sig")
    return out


def fig_deployment():
    preds = _load_preds()
    fams = [p for p in PILLAR_ORDER if p in preds]
    exact = [(preds[f]["raw_abs_error"] == 0).mean() for f in fams]
    within = [(preds[f]["raw_abs_error"] <= 1).mean() for f in fams]
    x = np.arange(len(fams))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    b1 = ax.bar(x - w / 2, exact, w, label="Exact score match", color=SERIES_A)
    b2 = ax.bar(x + w / 2, within, w, label="Within 1 point", color=SERIES_B)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{b.get_height():.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([PILLAR_LABELS[f] for f in fams])
    ax.set_ylabel("Fraction of test answers")
    ax.set_ylim(0, 1.0)
    ax.set_title("Deployment metrics: integer-score accuracy (LLM leads)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_deployment.png")


def fig_confusion_all():
    preds = _load_preds()
    fams = [p for p in PILLAR_ORDER if p in preds]
    labels = [0, 1, 2, 3, 4]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.2))
    for ax, fam in zip(axes.flat, fams):
        d = preds[fam]
        m = np.zeros((5, 5), dtype=int)
        for t, p in zip(d["true_label"], d["pred_label"]):
            if 0 <= int(t) <= 4 and 0 <= int(p) <= 4:
                m[int(t), int(p)] += 1
        im = ax.imshow(m, cmap="Blues")
        ax.set_xticks(labels); ax.set_yticks(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(PILLAR_LABELS[fam], fontsize=10)
        thresh = m.max() / 2 if m.max() else 0.5
        for i in labels:
            for j in labels:
                if m[i, j]:
                    ax.text(j, i, str(m[i, j]), ha="center", va="center",
                            color="white" if m[i, j] > thresh else "black", fontsize=8)
    for ax in axes.flat[len(fams):]:
        ax.axis("off")
    fig.suptitle("Confusion matrices (true vs predicted grade)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "fig_confusion_all.png")


def fig_pred_vs_true():
    preds = _load_preds()
    fams = [p for p in PILLAR_ORDER if p in preds]
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.2))
    for ax, fam in zip(axes.flat, fams):
        d = preds[fam]
        t = d["true_raw"].astype(float) + rng.uniform(-0.15, 0.15, len(d))
        p = d["pred_raw"].astype(float) + rng.uniform(-0.15, 0.15, len(d))
        ax.scatter(t, p, s=10, alpha=0.4, color=PILLAR_COLORS[fam])
        hi = float(max(d["true_raw"].max(), d["pred_raw"].max()))
        ax.plot([0, hi], [0, hi], color="gray", ls="--", lw=1)
        ax.set_xlabel("True raw score"); ax.set_ylabel("Predicted raw score")
        ax.set_title(PILLAR_LABELS[fam], fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(fams):]:
        ax.axis("off")
    fig.suptitle("Predicted vs true raw score (identity line = perfect)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "fig_pred_vs_true.png")


def fig_score_dist():
    preds = _load_preds()
    if "llm" not in preds:
        print("  [skip] fig_score_dist: llm predictions missing")
        return
    d = preds["llm"]
    hi = int(max(d["true_raw"].max(), d["pred_raw"].max()))
    bins = np.arange(-0.5, hi + 1.5, 1)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.hist(d["true_raw"], bins=bins, alpha=0.6, label="Human (true)", color=SERIES_A)
    ax.hist(d["pred_raw"], bins=bins, alpha=0.6, label="LLM (predicted)", color=SERIES_B)
    ax.set_xlabel("Raw score (points)")
    ax.set_ylabel("Number of answers")
    ax.set_title("LLM score distribution vs human (test set)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_score_dist.png")


# ============================================================== Group F: LLM fine-tune gain
def _v08_qwk(dataset, model_key, zeroshot):
    suffix = f"v08z_llm_{model_key}_zeroshot" if zeroshot else f"v08_llm_{model_key}"
    # Prefer the collated copy; fall back to the raw per-experiment leaderboard so
    # the figure works directly off fresh exp08 output without a collation step.
    candidates = [
        os.path.join(_ROOT, "results", "leaderboards", f"{dataset}_{suffix}.csv"),
        os.path.join(_ROOT, f"results_{dataset}_{suffix}", "leaderboard.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="utf-8-sig")
            if len(df):
                return float(df.iloc[0]["test_qwk"])
    return None


def fig_llm_finetune_gain():
    """Base (zero-shot) vs QLoRA fine-tuned test QWK for the KhmerGrader family.
    Pairs each model on the first dataset variant where both numbers exist."""
    labels, base_v, ft_v = [], [], []
    for key, name in LLM_KEYS:
        chosen = next((ds for ds in LLM_DATASETS
                       if _v08_qwk(ds, key, True) is not None
                       and _v08_qwk(ds, key, False) is not None), None)
        if chosen is None:
            print(f"  [skip] llm gain: {key} missing base (zero-shot) or fine-tuned leaderboard")
            continue
        labels.append(name)
        base_v.append(_v08_qwk(chosen, key, True))
        ft_v.append(_v08_qwk(chosen, key, False))
    if not labels:
        print("  [skip] fig_llm_finetune_gain: need zero-shot + fine-tuned leaderboards (run exp08)")
        return
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - w / 2, base_v, w, label="Base (zero-shot)", color=SERIES_A)
    ax.bar(x + w / 2, ft_v, w, label="Fine-tuned (QLoRA)", color=SERIES_B)
    for i in range(len(labels)):
        ax.text(i - w / 2, base_v[i] + 0.01, f"{base_v[i]:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, ft_v[i] + 0.01, f"{ft_v[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test QWK")
    ax.set_ylim(0, 1.0)
    ax.set_title("QLoRA fine-tuning lift over the base LLM (KhmerGrader family)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_llm_finetune_gain.png")


# ============================================================== Group D: explainability
# SHAP plausibility is reported as a table (thesis/paper tab:xai), not a generated figure, and the
# heatmaps are browser-screenshotted separately. The faithfulness (ERASER) and question-leakage
# figures were removed in the SHAP-only / no-leakage pivot.


# ============================================================== Group E: ablations
def fig_cleaning_ablation():
    df = _rd("results_stats/cleaning_ablation.csv")
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(x - w / 2, df["old_qwk"].astype(float), w, label="Old cleaning", color=SERIES_A)
    ax.bar(x + w / 2, df["new_qwk"].astype(float), w, label="New cleaning", color=SERIES_B)
    for i, dlt in enumerate(df["delta"].astype(float)):
        ax.text(i, max(df.loc[i, "old_qwk"], df.loc[i, "new_qwk"]) + 0.01,
                f"{dlt:+.3f}", ha="center", fontsize=8, color="gray")
    ax.set_xticks(x); ax.set_xticklabels(df["dataset"], fontsize=9)
    ax.set_ylabel("Classical test QWK")
    ax.set_ylim(0, 1.0)
    ax.set_title("Cleaning refinement changes QWK negligibly")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_cleaning_ablation.png")


def fig_hparam_tuning():
    df = _rd("results_stats/hparam_tuning.csv")
    default_col = [c for c in df.columns if c.startswith("default_test_qwk")][0]
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(x - w / 2, df[default_col].astype(float), w, label="Default hparams", color=SERIES_A)
    ax.bar(x + w / 2, df["tuned_test_qwk"].astype(float), w, label="Tuned hparams", color=SERIES_B)
    ax.set_xticks(x); ax.set_xticklabels(df["dataset"], fontsize=9)
    ax.set_ylabel("Classical test QWK")
    ax.set_ylim(0, 1.0)
    ax.set_title("Hyperparameter tuning has little effect")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_hparam_tuning.png")


def fig_calibration():
    """Per-pillar uncalibrated vs calibrated test QWK, joined on champion run_id
    between each base leaderboard and its *_calibrated counterpart (no10c)."""
    pairs = {
        "classical": ("no10c_v01_tfidf.csv", "no10c_v02_calibrated.csv"),
        "rnn":       ("no10c_v05_bilstm.csv", "no10c_v05_bilstm_calibrated.csv"),
        "encoder":   ("no10c_v06_transformer.csv", "no10c_v06_transformer_calibrated.csv"),
    }
    fams, uncal, cal = [], [], []
    for fam, (base_f, cal_f) in pairs.items():
        base_p = os.path.join(_ROOT, "results", "leaderboards", base_f)
        cal_p = os.path.join(_ROOT, "results", "leaderboards", cal_f)
        if not (os.path.exists(base_p) and os.path.exists(cal_p)):
            continue
        base = pd.read_csv(base_p, encoding="utf-8-sig")
        cal_df = pd.read_csv(cal_p, encoding="utf-8-sig")
        pick = "val_qwk" if base["val_qwk"].notna().any() else "test_qwk"
        champ = base.sort_values(pick, ascending=False).iloc[0]
        rid = champ["run_id"]
        match = cal_df[cal_df["run_id"] == rid]
        if match.empty:
            continue
        fams.append(fam)
        uncal.append(float(champ["test_qwk"]))
        cal.append(float(match.iloc[0]["test_qwk"]))
    if not fams:
        print("  [skip] fig_calibration: no run_id joins found")
        return
    x = np.arange(len(fams))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar(x - w / 2, uncal, w, label="Uncalibrated", color=SERIES_A)
    ax.bar(x + w / 2, cal, w, label="Calibrated", color=SERIES_B)
    for i in range(len(fams)):
        ax.text(i, max(uncal[i], cal[i]) + 0.01, f"{cal[i] - uncal[i]:+.3f}",
                ha="center", fontsize=8, color="gray")
    ax.set_xticks(x); ax.set_xticklabels([PILLAR_LABELS[f] for f in fams], fontsize=9)
    ax.set_ylabel("Test QWK")
    ax.set_ylim(0, 1.0)
    ax.set_title("Calibration is fragile and model-dependent")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_calibration.png")


# ============================================================== index + sync
def write_index():
    galleries = [
        ("Classical SHAP heatmaps", "../../results_xai/no10c/heatmaps/classical/classical_gallery.html"),
        ("BiLSTM SHAP heatmaps", "../../results_xai/no10c/heatmaps/bilstm/bilstm_gallery.html"),
    ]
    parts = ["<!doctype html><meta charset='utf-8'>",
             "<title>Khmer ASAG figure suite</title>",
             "<style>body{font-family:sans-serif;margin:24px;background:#fafafa}"
             "h1{font-size:20px}figure{display:inline-block;margin:10px;vertical-align:top;"
             "background:#fff;padding:8px;border:1px solid #ddd;border-radius:6px}"
             "img{width:360px;display:block}figcaption{font-size:12px;color:#444;margin-top:4px}"
             "a{color:#2b6cb0}</style>",
             "<h1>Khmer ASAG figure suite</h1>",
             f"<p>{len(_written)} figures generated from result files.</p>"]
    for name in sorted(_written):
        parts.append(f"<figure><img src='{name}'><figcaption>{name}</figcaption></figure>")
    parts.append("<h1>Khmer SHAP heatmap galleries</h1><ul>")
    for label, href in galleries:
        parts.append(f"<li><a href='{href}'>{label}</a></li>")
    parts.append("</ul>")
    with open(os.path.join(FIG, "figures_index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("  wrote figures_index.html")


def copy_curves():
    """Copy each iteratively-trained champion's train_history.png -> <fam>_curve.png
    (BiLSTM, encoder, LLM). The classical pillar trains in a single fit and has no curve."""
    srcs = {
        "bilstm_curve.png":  "results/champions/rnn_clean_ra_bilstm_909/train_history.png",
        "encoder_curve.png": "results/champions/encoder_clean_qar_dual_gte_maxfeat_1184/train_history.png",
        "llm_curve.png":     "results/champions/llm_clean_qar_qwen35_4b_909/train_history.png",
    }
    for name, src in srcs.items():
        p = os.path.join(_ROOT, src)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(FIG, name)); _written.append(name)
        else:
            print(f"  [skip] {name}: {src} not found yet")


def sync_to_thesis():
    os.makedirs(THESIS_FIG, exist_ok=True)
    for name in _written:
        shutil.copy(os.path.join(FIG, name), os.path.join(THESIS_FIG, name))
    print(f"  synced {len(_written)} figures to thesis/figures/")


FIGURES = [
    fig_label_dist, fig_subject_dist, fig_maxscore_dist, fig_answer_length,
    write_dataset_descriptive,
    fig_qwk_pillars, fig_metrics_grouped, fig_prf_pillars,
    fig_deployment, fig_confusion_all, fig_pred_vs_true, fig_score_dist,
    fig_llm_finetune_gain,
    fig_cleaning_ablation, fig_hparam_tuning, fig_calibration,
    copy_curves,
]


if __name__ == "__main__":
    print("Generating figure suite ->", FIG)
    for fn in FIGURES:
        try:
            fn()
        except Exception as e:  # one bad input never aborts the rest
            print(f"  [error] {fn.__name__}: {e}")
    write_index()
    sync_to_thesis()
    print("\nDone.", len(_written), "figures in paper/figures/ (also thesis/figures/).")
