"""Render the full figure suite for the thesis/paper from the real result files.

No GPU, no re-training, no network. Reads only CSVs and champion prediction files
already on disk and writes PNGs into paper/figures/ (copied to thesis/figures/).

Groups:
  A dataset        data/dataset.csv                          (label/subject/max-score/length)
  B accuracy       results_stats/champion_metrics.csv        (QWK + multi-metric)
  C deployment     results/champions/*/predictions_test.csv  (exact/within-1, confusion, scatter)
  D explainability results_xai/no10c_no0/faithfulness_leaderboard.csv  (LOO occlusion only)
  E robustness     split_compare / cleaning_ablation / hparam_tuning / leaderboards

Every value traces to a result file; nothing is hardcoded. Encoder/LLM faithfulness
rows are HPC-pending, so Group D plots whatever occlusion rows exist and skips the rest.

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
    "classical": "results/champions/classical_segment_ra_tfidf_svr_cal_895/predictions_test.csv",
    "rnn":       "results/champions/rnn_clean_ra_bilstm_895/predictions_test.csv",
    "encoder":   "results/champions/encoder_clean_qar_dual_gte_maxfeat_1184/predictions_test.csv",
    "llm":       "results/champions/llm_clean_qar_qwen35_4b_909/predictions_test.csv",
}

# KhmerGrader LLM family: internal key -> released name. Used by fig_llm_finetune_gain.
LLM_KEYS = [("qwen35_4b", "Qwen-\nKhmerGrader-4B"),
            ("gemma4_e4b", "Gemma-\nKhmerGrader-4B"),
            ("sealion_v45_e2b", "SEA-LION-\nKhmerGrader-E2B")]
LLM_DATASETS = ["no10c_no0", "no10c", "full"]   # preference order for a fair pairing

# Faithfulness leaderboard family keys (note: it uses "bilstm", not "rnn").
FAITH_ORDER = ["classical", "bilstm", "encoder", "llm"]
FAITH_LABELS = {"classical": "Classical", "bilstm": "RNN\n(BiLSTM)",
                "encoder": "Transformer", "llm": "LLM"}

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
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([PILLAR_LABELS[p] for p in PILLAR_ORDER])
    ax.set_ylabel("Test QWK")
    ax.set_ylim(0, 1.0)
    ax.set_title("Primary metric: QWK comparable across pillars (narrow band)")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_qwk_pillars.png")


def fig_metrics_grouped():
    df = _champion_metrics()
    metrics = [("qwk", "QWK"), ("cohen_kappa", "Cohen kappa"),
               ("accuracy", "Accuracy"), ("f1_macro", "Macro-F1")]
    x = np.arange(len(metrics))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, p in enumerate(PILLAR_ORDER):
        vals = [float(df.loc[p, m]) for m, _ in metrics]
        ax.bar(x + (i - 1.5) * w, vals, w, label=PILLAR_LABELS[p], color=PILLAR_COLORS[p])
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Standard agreement metrics by pillar")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_metrics_grouped.png")


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


# ============================================================== Group D: explainability (LOO only)
def _occlusion_rows():
    df = _rd("results_xai/no10c_no0/faithfulness_leaderboard.csv")
    df = df[df["explainer"] == "occlusion"].copy()
    df["__ord"] = df["family"].apply(lambda f: FAITH_ORDER.index(f) if f in FAITH_ORDER else 99)
    return df.sort_values("__ord")


def fig_faithfulness():
    df = _occlusion_rows()
    if df.empty:
        print("  [skip] fig_faithfulness: no occlusion rows yet")
        return
    fams = df["family"].tolist()
    vals = df["aopc_comprehensiveness"].astype(float).tolist()
    rand = float(df["comprehensiveness_random"].iloc[0])
    pending = [f for f in FAITH_ORDER if f not in fams]
    if pending:
        print("  [skip] faithfulness families pending (HPC):", ", ".join(pending))
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(range(len(vals)), vals, color=GREEN, width=0.6)
    ax.axhline(rand, color="gray", ls="--", label=f"random baseline ({rand:.3f})")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([FAITH_LABELS.get(f, f) for f in fams], fontsize=9)
    ax.set_ylabel("AOPC comprehensiveness (higher = faithful)")
    ax.set_title("LOO word attribution is faithful across families")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_faithfulness.png")


def fig_faithfulness_gap():
    df = _occlusion_rows()
    if df.empty:
        print("  [skip] fig_faithfulness_gap: no occlusion rows yet")
        return
    fams = df["family"].tolist()
    gaps = df["faithfulness_gap"].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    bars = ax.bar(range(len(gaps)), gaps, color=GREEN, width=0.6)
    for b, v in zip(bars, gaps):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([FAITH_LABELS.get(f, f) for f in fams], fontsize=9)
    ax.set_ylabel("Faithfulness gap (LOO minus random)")
    ax.set_title("LOO beats random word removal (gap > 0 = faithful)")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_faithfulness_gap.png")


def fig_faith_vs_plaus():
    df = _occlusion_rows()
    if df.empty:
        print("  [skip] fig_faith_vs_plaus: no occlusion rows yet")
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for _, r in df.iterrows():
        fam = r["family"]
        x = float(r["aopc_comprehensiveness"]); y = float(r["plausibility"])
        ax.scatter(x, y, s=80, color=GREEN, zorder=3)
        ax.annotate(FAITH_LABELS.get(fam, fam).replace("\n", " "),
                    (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("AOPC comprehensiveness (faithfulness)")
    ax.set_ylabel("Plausibility (reference overlap)")
    ax.set_title("Faithful and plausible (upper right is best)")
    ax.grid(alpha=0.3)
    _save(fig, "fig_faith_vs_plaus.png")


# ============================================================== Group E: robustness
def fig_leakage():
    df = _rd("results_stats/split_compare.csv").set_index("split")
    cats = [("random", "Random split\n(seen questions)"),
            ("question", "Question-held-out\n(unseen questions)")]
    means = [float(df.loc[k, "mean_qwk"]) for k, _ in cats]
    stds = [float(df.loc[k, "std_qwk"]) for k, _ in cats]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    bars = ax.bar([0, 1], means, yerr=stds, capsize=6, color=[GREEN, RED], width=0.55)
    ax.set_xticks([0, 1]); ax.set_xticklabels([c[1] for c in cats], fontsize=9)
    ax.set_ylabel("Classical QWK (mean +/- std, 5 seeds)")
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Question leakage: QWK collapses on unseen questions")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.2f}", ha="center", fontsize=10)
    ax.annotate(f"-{means[0] - means[1]:.2f} QWK", xy=(0.5, max(means) - 0.1),
                ha="center", fontsize=11, color=RED, fontweight="bold")
    _save(fig, "fig_leakage.png")


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
    between each base leaderboard and its *_calibrated counterpart (no10c_no0)."""
    pairs = {
        "classical": ("no10c_no0_v01_tfidf.csv", "no10c_no0_v02_calibrated.csv"),
        "rnn":       ("no10c_no0_v05_bilstm.csv", "no10c_no0_v05_bilstm_calibrated.csv"),
        "encoder":   ("no10c_no0_v06_transformer.csv", "no10c_no0_v06_transformer_calibrated.csv"),
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
        ("Classical LOO heatmaps", "../../results_xai/no10c_no0/heatmaps/classical/classical_gallery.html"),
        ("BiLSTM LOO heatmaps", "../../results_xai/no10c_no0/heatmaps/bilstm/bilstm_gallery.html"),
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
    parts.append("<h1>Khmer LOO heatmap galleries</h1><ul>")
    for label, href in galleries:
        parts.append(f"<li><a href='{href}'>{label}</a></li>")
    parts.append("</ul>")
    with open(os.path.join(FIG, "figures_index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("  wrote figures_index.html")


def sync_to_thesis():
    os.makedirs(THESIS_FIG, exist_ok=True)
    for name in _written:
        shutil.copy(os.path.join(FIG, name), os.path.join(THESIS_FIG, name))
    print(f"  synced {len(_written)} figures to thesis/figures/")


FIGURES = [
    fig_label_dist, fig_subject_dist, fig_maxscore_dist, fig_answer_length,
    write_dataset_descriptive,
    fig_qwk_pillars, fig_metrics_grouped,
    fig_deployment, fig_confusion_all, fig_pred_vs_true, fig_score_dist,
    fig_llm_finetune_gain,
    fig_faithfulness, fig_faithfulness_gap, fig_faith_vs_plaus,
    fig_leakage, fig_cleaning_ablation, fig_hparam_tuning, fig_calibration,
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
