"""Render publication figures for the IEEE manuscript from the real result CSVs.

No GPU, no re-training — reads:
  results_stats/split_compare.csv                      (random vs unseen-question)
  results_xai/no10c_no0/faithfulness_leaderboard.csv   (AOPC faithfulness)

Outputs PNGs into paper/figures/. Run:  python paper/make_figures.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
FIG = os.path.join(_HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_leakage():
    rows = {r["split"]: r for r in _read(os.path.join(_ROOT, "results_stats", "split_compare.csv"))}
    cats = [("random", "Random split\n(seen questions)"), ("question", "Question-held-out\n(unseen questions)")]
    means = [float(rows[k]["mean_qwk"]) for k, _ in cats]
    stds = [float(rows[k]["std_qwk"]) for k, _ in cats]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    bars = ax.bar([0, 1], means, yerr=stds, capsize=6,
                  color=["#2f855a", "#c53030"], width=0.55)
    ax.set_xticks([0, 1]); ax.set_xticklabels([c[1] for c in cats], fontsize=9)
    ax.set_ylabel("Classical QWK (mean ± std, 5 seeds)")
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Question leakage: QWK collapses on unseen questions")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.2f}", ha="center", fontsize=10)
    ax.annotate(f"-{means[0]-means[1]:.2f} QWK", xy=(0.5, max(means) - 0.1),
                ha="center", fontsize=11, color="#c53030", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_leakage.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_faithfulness():
    rows = _read(os.path.join(_ROOT, "results_xai", "no10c_no0", "faithfulness_leaderboard.csv"))
    keep = [("classical", "occlusion", "Classical\nocclusion"),
            ("bilstm", "occlusion", "BiLSTM\nocclusion"),
            ("bilstm", "attention", "BiLSTM\nattention")]
    vals, rand = [], None
    for fam, exp, _ in keep:
        for r in rows:
            if r["family"] == fam and r["explainer"] == exp:
                vals.append(float(r["aopc_comprehensiveness"]))
                rand = float(r["comprehensiveness_random"])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    # occlusion = reliably faithful (green); attention = unreliable/config-dependent (amber)
    colors = ["#2f855a", "#2f855a", "#dd6b20"]
    ax.bar(range(len(vals)), vals, color=colors, width=0.6)
    if rand is not None:
        ax.axhline(rand, color="gray", ls="--", label=f"random baseline ({rand:.3f})")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(keep))); ax.set_xticklabels([k[2] for k in keep], fontsize=9)
    ax.set_ylabel("AOPC comprehensiveness (↑ = faithful)")
    ax.set_title("Occlusion reliably faithful; attention configuration-dependent")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_faithfulness.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_leakage()
    fig_faithfulness()
    print("wrote:", sorted(os.listdir(FIG)))
