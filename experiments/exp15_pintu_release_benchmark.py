"""exp15: benchmark every published Pintu variant against the others.

Collects the validation runs of each release variant (adapter, merged BF16 M1/M2,
text-only GGUF BF16 and quants, zero-shot base), recomputes every metric from the
saved predictions with the shared ``evaluate.metrics`` (so all variants are scored
by one function), measures fidelity against a reference variant, and writes:

    results_stats/pintu_release_benchmark.csv   one row per release x variant x device
    results_stats/pintu_release_fidelity.csv    pairwise prediction agreement
    results_stats/figures/pintu_release/<release>/*.png
    publish/benchmark/<release>/REPORT.md       (+ figures/, for the Hub repos)
    docs/pintu_release_report.md                same report for the local docs

Numbers come only from saved predictions; a variant that has not been run is
listed as ``[pending]``. ``--dry-run`` builds a SYNTHETIC fixture in a temporary
folder to exercise the whole pipeline locally; its report is stamped as synthetic
and nothing is written to results_stats/ or docs/.

    python experiments/exp15_pintu_release_benchmark.py --models qwen
    python experiments/exp15_pintu_release_benchmark.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluate import metrics as evaluate_metrics  # noqa: E402

# ── Variant catalogue: fixed order and short IDs used in every table and figure ──
VARIANTS = [
    # key                  id    label                                   runtime
    ("zeroshot_base",      "H",  "Base model, zero-shot",                "unsloth"),
    ("adapter_qlora4bit",  "A",  "QLoRA adapter on 4-bit base (report)", "unsloth"),
    ("merged_bf16",        "B1", "Merged BF16, original base (M1)",      "transformers"),
    ("merged_nf4dequant",  "B2", "Merged BF16, dequantized NF4 base (M2)", "transformers"),
    ("gguf_bf16",          "C",  "Text-only GGUF BF16",                  "llama.cpp"),
    ("gguf_q8_0",          "D",  "Text-only GGUF Q8_0",                  "llama.cpp"),
    ("gguf_q6_k",          "E",  "Text-only GGUF Q6_K",                  "llama.cpp"),
    ("gguf_q5_k_m",        "F",  "Text-only GGUF Q5_K_M",                "llama.cpp"),
    ("gguf_q4_k_m",        "G",  "Text-only GGUF Q4_K_M",                "llama.cpp"),
]
IMAT_IDS = {"gguf_q6_k_imat": "E*", "gguf_q5_k_m_imat": "F*", "gguf_q4_k_m_imat": "G*"}
for _key, _id in IMAT_IDS.items():
    VARIANTS.append((_key, _id, f"Text-only GGUF {_key[5:-5].upper()} + imatrix", "llama.cpp"))
ORDER = {key: i for i, (key, *_rest) in enumerate(VARIANTS)}
VARIANT_ID = {key: vid for key, vid, *_ in VARIANTS}
VARIANT_LABEL = {key: label for key, _vid, label, _rt in VARIANTS}
VARIANT_RUNTIME = {key: rt for key, _vid, _label, rt in VARIANTS}

RELEASES = {
    "qwen": ("Pintu-Qwen3.5-4B", "qwen35_4b"),
    "gemma": ("Pintu-Gemma4-E4B", "gemma4_e4b"),
    "sealion": ("Pintu-SEA-LION-v4.5-E2B", "sealion_v45_e2b"),
}

# Reference palette (dataviz skill, light mode): categorical slots 1-3 in fixed order.
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
KEY_COLS = ["Question", "Answer", "Max Score", "true_raw"]


@dataclass
class Run:
    variant: str
    device: str
    predictions: pd.DataFrame
    validation: dict
    source: str


# ── Loading ───────────────────────────────────────────────────────────────────
def keyed(df: pd.DataFrame) -> pd.DataFrame:
    """Add a row key robust to order and duplicates (text + gold + occurrence)."""
    out = df.copy()
    base = out[KEY_COLS].astype(str).agg("\x1f".join, axis=1)
    out["_key"] = base + "\x1f" + out.groupby(base).cumcount().astype(str)
    return out


def load_runs(release: str, model_suffix: str, bench_root: Path, results_root: Path) -> list[Run]:
    runs: list[Run] = []
    release_dir = bench_root / release
    if release_dir.is_dir():
        for run_dir in sorted(release_dir.iterdir()):
            pred = run_dir / "predictions_test.csv"
            if not run_dir.is_dir() or "__" not in run_dir.name or not pred.is_file():
                continue
            variant, device = run_dir.name.split("__", 1)
            if variant not in ORDER:
                print(f"[warn] unknown variant folder skipped: {run_dir}")
                continue
            val_path = run_dir / "validation.json"
            validation = json.loads(val_path.read_text(encoding="utf-8")) if val_path.is_file() else {}
            runs.append(Run(variant, device, pd.read_csv(pred), validation, str(run_dir)))

    have = {r.variant for r in runs}
    # The report's own adapter run and the zero-shot base come from the exp08 grid.
    fallbacks = {
        "adapter_qlora4bit": results_root / f"results_no10c_v08_llm_{model_suffix}"
        / "runs" / f"clean_qar_{model_suffix}",
        "zeroshot_base": results_root / f"results_no10c_v08z_llm_{model_suffix}_zeroshot"
        / "runs" / f"zeroshot_qar_{model_suffix}",
    }
    for variant, run_dir in fallbacks.items():
        pred = run_dir / "predictions_test.csv"
        if variant not in have and pred.is_file():
            runs.append(Run(variant, "gpu", pd.read_csv(pred), {}, str(run_dir)))
    return sorted(runs, key=lambda r: (ORDER[r.variant], r.device))


# ── Metrics ───────────────────────────────────────────────────────────────────
def score(df: pd.DataFrame) -> dict:
    m = evaluate_metrics(
        pred_scores=df["pred_score"].to_numpy(),
        true_labels=df["true_label"].to_numpy(),
        max_scores=df["Max Score"].to_numpy(),
        true_raw=df["true_raw"].to_numpy(),
    )
    m["n"] = len(df)
    return m


def fidelity(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Agreement of raw integer predictions between two runs on the same rows."""
    joined = keyed(a)[["_key", "pred_raw"]].merge(
        keyed(b)[["_key", "pred_raw"]], on="_key", suffixes=("_a", "_b"))
    if joined.empty:
        return {"rows": 0}
    diff = joined["pred_raw_a"].to_numpy(int) - joined["pred_raw_b"].to_numpy(int)
    return {
        "rows": len(joined),
        "match_rate": float(np.mean(diff == 0)),
        "changed": int(np.sum(diff != 0)),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": int(np.max(np.abs(diff))),
    }


def pick_device_runs(runs: list[Run]) -> dict[str, Run]:
    """One run per variant for cross-variant comparisons (GPU preferred)."""
    chosen: dict[str, Run] = {}
    for run in runs:
        current = chosen.get(run.variant)
        if current is None or (current.device != "gpu" and run.device == "gpu"):
            chosen[run.variant] = run
    return dict(sorted(chosen.items(), key=lambda kv: ORDER[kv[0]]))


def weights_gb(run: Run, release: str, gguf_root: Path, merged_roots: dict) -> float | None:
    cost = run.validation.get("cost", {})
    if cost.get("weights_gb"):
        return float(cost["weights_gb"])
    if run.variant.startswith("gguf_"):
        stem = run.variant[5:].upper().replace("_IMAT", "")
        suffix = "-imat" if run.variant.endswith("_imat") else ""
        path = gguf_root / release / f"{release}-{stem}{suffix}.gguf"
        return round(path.stat().st_size / 1e9, 3) if path.is_file() else None
    if run.variant in merged_roots:
        folder = merged_roots[run.variant] / release
        files = list(folder.glob("*.safetensors"))
        return round(sum(f.stat().st_size for f in files) / 1e9, 3) if files else None
    return None


def build_tables(release: str, runs: list[Run], reference: str, gguf_root: Path,
                 merged_roots: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = pick_device_runs(runs)
    ref_run = primary.get(reference)
    ref_metrics = score(ref_run.predictions) if ref_run else None

    rows = []
    for run in runs:
        m = score(run.predictions)
        cost = run.validation.get("cost", {})
        fid = fidelity(run.predictions, ref_run.predictions) if ref_run else {}
        rows.append({
            "release": release,
            "id": VARIANT_ID[run.variant],
            "variant": run.variant,
            "label": VARIANT_LABEL[run.variant],
            "device": run.device,
            "runtime": run.validation.get("runtime", VARIANT_RUNTIME[run.variant]),
            "n": m["n"],
            "qwk": m["qwk"],
            "cohen_kappa": m["cohen_kappa"],
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "adjacent_accuracy": m["adjacent_accuracy"],
            "raw_exact": m["raw_exact"],
            "raw_within1": m["raw_within1"],
            "raw_mae": m["raw_mae"],
            "reference": reference if ref_run else None,
            "qwk_delta_vs_ref": (m["qwk"] - ref_metrics["qwk"]) if ref_metrics else None,
            "match_rate_vs_ref": fid.get("match_rate"),
            "changed_vs_ref": fid.get("changed"),
            "mean_abs_diff_vs_ref": fid.get("mean_abs_diff"),
            "max_abs_diff_vs_ref": fid.get("max_abs_diff"),
            "weights_gb": weights_gb(run, release, gguf_root, merged_roots),
            "sec_per_answer_median": cost.get("seconds_per_answer_median"),
            "sec_per_answer_p90": cost.get("seconds_per_answer_p90"),
            "load_seconds": cost.get("load_seconds"),
            "peak_mem_gb": cost.get("peak_vram_gb") or cost.get("peak_rss_gb"),
            "threads": cost.get("threads"),
            "parse_failures": cost.get("parse_failures"),
            "source": run.source,
        })
    bench = pd.DataFrame(rows)

    pairs = []
    keys = list(primary)
    for i, a in enumerate(keys):
        for b in keys[i:]:
            f = fidelity(primary[a].predictions, primary[b].predictions)
            pairs.append({"release": release, "variant_a": a, "variant_b": b,
                          "id_a": VARIANT_ID[a], "id_b": VARIANT_ID[b], **f})
    # Same variant on CPU vs GPU: a runtime-determinism check.
    by_variant: dict[str, list[Run]] = {}
    for run in runs:
        by_variant.setdefault(run.variant, []).append(run)
    for variant, group in by_variant.items():
        devices = {r.device: r for r in group}
        if {"cpu", "gpu"} <= set(devices):
            f = fidelity(devices["cpu"].predictions, devices["gpu"].predictions)
            pairs.append({"release": release, "variant_a": f"{variant}__cpu",
                          "variant_b": f"{variant}__gpu", "id_a": VARIANT_ID[variant] + " cpu",
                          "id_b": VARIANT_ID[variant] + " gpu", **f})
    return bench, pd.DataFrame(pairs)


# ── Figures (matplotlib, static PNG for the report and the Hub) ───────────────
def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def make_figures(release: str, bench: pd.DataFrame, fid: pd.DataFrame, runs: list[Run],
                 reference: str, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    primary_rows = (bench.assign(_gpu=bench["device"].eq("gpu"))
                    .sort_values(["variant", "_gpu"]).groupby("variant").tail(1)
                    .assign(_o=lambda d: d["variant"].map(ORDER)).sort_values("_o"))

    # 1. Agreement with the human grader: QWK, exact, within-1 per variant.
    fig, ax = plt.subplots(figsize=(9, 3.8))
    _style(ax)
    x = np.arange(len(primary_rows))
    width = 0.26
    for i, (col, name) in enumerate([("qwk", "QWK"), ("raw_exact", "Exact (raw points)"),
                                     ("raw_within1", "Within 1 point")]):
        vals = primary_rows[col].to_numpy(float)
        bars = ax.bar(x + (i - 1) * width, vals, width - 0.03, color=SERIES[i], label=name)
        if col == "qwk":
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(x, primary_rows["id"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Agreement with human grader", color=INK_2, fontsize=9)
    ax.set_title(f"{release}: accuracy by variant (test split, n={int(primary_rows['n'].iloc[0])})",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    paths.append(_save(fig, out_dir / "accuracy_by_variant.png"))

    # 2. Size vs QWK (Pareto view).
    sized = primary_rows.dropna(subset=["weights_gb"])
    if len(sized):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        _style(ax, "both")
        ax.scatter(sized["weights_gb"], sized["qwk"], s=64, color=SERIES[0],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        for _, r in sized.iterrows():
            ax.annotate(r["id"], (r["weights_gb"], r["qwk"]), textcoords="offset points",
                        xytext=(6, 4), fontsize=9, color=INK)
        frontier = sized.sort_values("weights_gb")
        best, pts = -np.inf, []
        for _, r in frontier.iterrows():
            if r["qwk"] > best:
                best = r["qwk"]
                pts.append((r["weights_gb"], r["qwk"]))
        if len(pts) > 1:
            ax.plot(*zip(*pts), color=SERIES[0], linewidth=2, alpha=0.5, zorder=2)
        ax.set_xlabel("Weights on disk (GB)", color=INK_2, fontsize=9)
        ax.set_ylabel("QWK", color=INK_2, fontsize=9)
        ax.set_title(f"{release}: size vs QWK (line = Pareto frontier)", color=INK,
                     fontsize=11, loc="left")
        paths.append(_save(fig, out_dir / "size_vs_qwk.png"))

    # 3. Latency vs QWK, one panel per device (no dual axis).
    timed = bench.dropna(subset=["sec_per_answer_median"])
    devices = [d for d in ("gpu", "cpu") if d in set(timed["device"])]
    if devices:
        fig, axes = plt.subplots(1, len(devices), figsize=(5.2 * len(devices), 3.8), squeeze=False)
        for ax, device in zip(axes[0], devices):
            _style(ax, "both")
            sub = timed[timed["device"] == device]
            ax.scatter(sub["sec_per_answer_median"], sub["qwk"], s=64, color=SERIES[0],
                       edgecolor=SURFACE, linewidth=2, zorder=3)
            for _, r in sub.iterrows():
                ax.annotate(r["id"], (r["sec_per_answer_median"], r["qwk"]),
                            textcoords="offset points", xytext=(6, 4), fontsize=9, color=INK)
            ax.set_xlabel("Median seconds per answer", color=INK_2, fontsize=9)
            ax.set_ylabel("QWK", color=INK_2, fontsize=9)
            ax.set_title(f"{device.upper()}", color=INK, fontsize=10, loc="left")
        fig.suptitle(f"{release}: latency vs QWK", color=INK, fontsize=11, x=0.02, ha="left")
        paths.append(_save(fig, out_dir / "latency_vs_qwk.png"))

    # 4. Pairwise prediction-agreement heatmap (single-hue sequential).
    pair = fid[~fid["variant_a"].str.contains("__")]
    ids = [VARIANT_ID[k] for k in sorted(set(pair["variant_a"]) | set(pair["variant_b"]),
                                         key=ORDER.get)]
    if len(ids) > 1:
        mat = pd.DataFrame(np.nan, index=ids, columns=ids)
        for _, r in pair.iterrows():
            mat.loc[r["id_a"], r["id_b"]] = mat.loc[r["id_b"], r["id_a"]] = r["match_rate"]
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("blue_seq", ["#eaf2fc", "#2a78d6", "#0c3a70"])
        fig, ax = plt.subplots(figsize=(0.62 * len(ids) + 2, 0.55 * len(ids) + 1.4))
        vmin = float(np.nanmin(mat.to_numpy()) - 0.02)
        im = ax.imshow(mat.to_numpy(float), cmap=cmap, vmin=vmin, vmax=1.0)
        ax.set_xticks(range(len(ids)), ids)
        ax.set_yticks(range(len(ids)), ids)
        for i in range(len(ids)):
            for j in range(len(ids)):
                v = mat.iloc[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if (v - vmin) / (1.0 - vmin) > 0.45 else INK)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8, colors=INK_2)
        cbar.outline.set_visible(False)
        ax.set_title(f"{release}: identical raw scores between variants", color=INK,
                     fontsize=11, loc="left")
        paths.append(_save(fig, out_dir / "agreement_heatmap.png"))

    # 5. Confusion matrices vs the human grader: reference and smallest quant.
    primary = pick_device_runs(runs)
    small = next((k for k in ("gguf_q4_k_m", "gguf_q4_k_m_imat", "gguf_q5_k_m", "gguf_q8_0")
                  if k in primary), None)
    panels = [k for k in (reference, small) if k in primary]
    if panels:
        from sklearn.metrics import confusion_matrix
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f5f8fd", "#2a78d6", "#0c3a70"])
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.9), squeeze=False)
        for ax, key in zip(axes[0], panels):
            df = primary[key].predictions
            pred_label = np.round(df["pred_score"].to_numpy(float) * 4).clip(0, 4).astype(int)
            cm = confusion_matrix(df["true_label"].astype(int), pred_label, labels=range(5))
            ax.imshow(cm, cmap=cmap)
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                            color="white" if cm[i, j] > cm.max() * 0.55 else INK)
            ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.set_xlabel("Predicted grade (0-4)", color=INK_2, fontsize=9)
            ax.set_ylabel("Human grade (0-4)", color=INK_2, fontsize=9)
            ax.set_title(f"{VARIANT_ID[key]}: {VARIANT_LABEL[key]}", color=INK, fontsize=9, loc="left")
            ax.tick_params(colors=INK_2, labelsize=9, length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        paths.append(_save(fig, out_dir / "confusion_matrices.png"))

    # 6. Raw-score changes vs the reference, small multiples per quant.
    quants = [k for k in primary if k.startswith("gguf_") or k == "merged_nf4dequant"
              or k == "merged_bf16" or k == "adapter_qlora4bit"]
    quants = [k for k in quants if k != reference]
    if reference in primary and quants:
        ref_df = keyed(primary[reference].predictions)[["_key", "pred_raw"]]
        cols = min(4, len(quants))
        rows_n = int(np.ceil(len(quants) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(3.1 * cols, 2.7 * rows_n),
                                 squeeze=False, sharey=True,
                                 gridspec_kw={"hspace": 0.55, "wspace": 0.15})
        bins = np.arange(-3, 4)
        for ax, key in zip(axes.flat, quants):
            _style(ax)
            j = keyed(primary[key].predictions)[["_key", "pred_raw"]].merge(
                ref_df, on="_key", suffixes=("", "_ref"))
            d = (j["pred_raw"] - j["pred_raw_ref"]).clip(-3, 3)
            counts = [int((d == b).sum()) for b in bins]
            ax.bar(bins, counts, 0.8, color=SERIES[0])
            ax.set_xticks(bins, ["<=-3", "-2", "-1", "0", "+1", "+2", ">=+3"], fontsize=7)
            ax.set_title(f"{VARIANT_ID[key]} vs {VARIANT_ID[reference]}: "
                         f"{int((d != 0).sum())} changed", color=INK, fontsize=9, loc="left")
        for ax in list(axes.flat)[len(quants):]:
            ax.axis("off")
        fig.supxlabel("Raw-score difference (points)", color=INK_2, fontsize=9)
        fig.supylabel("Answers", color=INK_2, fontsize=9)
        paths.append(_save(fig, out_dir / "score_changes_vs_reference.png"))

    # 7. Pipeline diagram.
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis("off")
    steps = ["QLoRA adapter\n(4-bit NF4 base)", "Merge in BF16\n(M1 / M2)", "Validate\n(test split)",
             "Text-only GGUF\nBF16", "Quantize\nQ8_0 ... Q4_K_M", "Validate\nCPU + GPU", "Upload\nHF Hub"]
    for i, text in enumerate(steps):
        xc = 0.07 + i * 0.143
        ax.add_patch(plt.Rectangle((xc - 0.06, 0.3), 0.12, 0.42, facecolor="#eaf2fc",
                                   edgecolor=SERIES[0], linewidth=1.2, transform=ax.transAxes))
        ax.text(xc, 0.51, text, ha="center", va="center", fontsize=8, color=INK,
                transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xc + 0.083, 0.51), xytext=(xc + 0.061, 0.51),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.2))
    ax.text(0.01, 0.92, "Pintu release pipeline (every stage writes a provenance manifest)",
            fontsize=10, color=INK, transform=ax.transAxes)
    paths.append(_save(fig, out_dir / "pipeline.png"))
    return paths


# ── Report ────────────────────────────────────────────────────────────────────
def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "[pending]"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.{digits}f}"


def write_report(release: str, bench: pd.DataFrame, fid: pd.DataFrame, reference: str,
                 fig_paths: list[Path], fig_prefix: str, out_path: Path, synthetic: bool):
    present = set(bench["variant"])
    lines: list[str] = []
    if synthetic:
        lines += ["> **SYNTHETIC DRY RUN.** Every number below comes from a generated fixture used "
                  "to test the pipeline. None of it is a result.", ""]
    lines += [
        f"# {release}: release benchmark",
        "",
        "## Method",
        "",
        "The Pintu adapter was trained with QLoRA (Dettmers et al., 2023): a frozen 4-bit NF4 base "
        "plus 16-bit LoRA weights. Following standard practice (merge in 16-bit, quantize last), "
        "the adapter is merged into a BF16 model and every quantized build is derived from that "
        "merged model. Because the adapter learned corrections to the dequantized NF4 weights, "
        "not the original BF16 weights, merging is not exactly lossless (the gap motivated "
        "QA-LoRA and LoftQ). Two merge targets are therefore compared: M1 merges into the "
        "original BF16 base, M2 into the dequantized NF4 base the adapter was trained against.",
        "",
        "The quantized builds are text-only GGUF files (llama.cpp): the vision and audio towers "
        "are dropped because grading uses text only. Following Kurtic et al. (2024), each quant "
        "is evaluated against its BF16 reference on the same task, reporting accuracy, fidelity "
        "(identical predictions), size and latency.",
        "",
        "All variants score the same curated test rows with the same chat-template prompt "
        "(thinking disabled), greedy decoding (32 new tokens), the same integer parser and the "
        "same metric function. Agreement is with a single human grader. The test split is small, "
        "so the number of changed answers is reported next to every QWK; a difference of one or "
        "two answers should be read as comparable.",
        "",
        f"Fidelity reference: **{VARIANT_ID.get(reference, reference)}** ({VARIANT_LABEL.get(reference, reference)}).",
        "",
        "## Variants",
        "",
        "| ID | Variant | Runtime | Status |",
        "|---|---|---|---|",
    ]
    for key, vid, label, runtime in VARIANTS:
        if key.endswith("_imat") and key not in present:
            continue
        lines.append(f"| {vid} | {label} | {runtime} | {'measured' if key in present else '[pending]'} |")

    lines += ["", "## Accuracy (agreement with the human grader)", "",
              "| ID | Device | n | QWK | Cohen k | Exact | Within 1 | Raw MAE (pt) | Macro-F1 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in bench.iterrows():
        lines.append(f"| {r['id']} | {r['device']} | {r['n']} | {_fmt(r['qwk'])} | "
                     f"{_fmt(r['cohen_kappa'])} | {_fmt(r['raw_exact'])} | {_fmt(r['raw_within1'])} | "
                     f"{_fmt(r['raw_mae'], 2)} | {_fmt(r['f1_macro'])} |")

    lines += ["", f"## Fidelity vs {VARIANT_ID.get(reference, reference)}", "",
              "| ID | Device | QWK change | Identical scores | Changed answers | Mean abs diff (pt) | Max abs diff (pt) |",
              "|---|---|---|---|---|---|---|"]
    for _, r in bench.iterrows():
        lines.append(f"| {r['id']} | {r['device']} | {_fmt(r['qwk_delta_vs_ref'])} | "
                     f"{_fmt(r['match_rate_vs_ref'])} | {_fmt(r['changed_vs_ref'])} | "
                     f"{_fmt(r['mean_abs_diff_vs_ref'], 2)} | {_fmt(r['max_abs_diff_vs_ref'])} |")

    lines += ["", "## Cost", "",
              "| ID | Device | Weights (GB) | Median s/answer | p90 s/answer | Load (s) | Peak memory (GB) | Parse failures |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in bench.iterrows():
        lines.append(f"| {r['id']} | {r['device']} | {_fmt(r['weights_gb'], 2)} | "
                     f"{_fmt(r['sec_per_answer_median'], 2)} | {_fmt(r['sec_per_answer_p90'], 2)} | "
                     f"{_fmt(r['load_seconds'], 1)} | {_fmt(r['peak_mem_gb'], 2)} | "
                     f"{_fmt(r['parse_failures'])} |")

    determinism = fid[fid["variant_a"].str.contains("__")]
    if len(determinism):
        lines += ["", "## CPU vs GPU determinism", "",
                  "| Variant | Identical scores | Changed answers |", "|---|---|---|"]
        for _, r in determinism.iterrows():
            lines.append(f"| {r['id_a'].split()[0]} | {_fmt(r['match_rate'])} | {_fmt(r['changed'])} |")

    lines += ["", "## Figures", ""]
    for path in fig_paths:
        lines.append(f"![{path.stem}]({fig_prefix}{path.name})")
        lines.append("")

    lines += [
        "## Limitations",
        "",
        "- One test split scored against one human grader; small differences between variants "
        "are within a handful of answers.",
        "- Latency depends on hardware and thread count (logged in the manifests); compare "
        "variants within one device, not across machines.",
        "- The text-only GGUF builds cannot process images or audio; the full BF16 repository "
        "keeps the original multimodal architecture.",
        "",
        "## References",
        "",
        "- Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.",
        "- Xu, Y. et al. (2023). QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models. arXiv:2309.14717.",
        "- Li, Y. et al. (2023). LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models. arXiv:2310.08659.",
        "- Kurtic, E. et al. (2024). \"Give Me BF16 or Give Me Death\"? Accuracy-Performance Trade-Offs in LLM Quantization. arXiv:2411.02355.",
        "",
        "Provenance for every run (code commit, library versions, hardware, file SHA-256) is in "
        "`publish/manifests/`.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ── Synthetic fixture for --dry-run ───────────────────────────────────────────
def make_fixture(root: Path, release: str, n: int = 137, seed: int = 0) -> None:
    """Write fake predictions for every variant. Used only to test the pipeline."""
    rng = np.random.default_rng(seed)
    maxes = rng.choice([5, 6, 7, 8, 10, 12, 15, 20], size=n)
    true_raw = np.array([rng.integers(0, m + 1) for m in maxes])
    base = pd.DataFrame({
        "Question": [f"q{i % 41}" for i in range(n)],
        "Reference": "r",
        "Answer": [f"answer {i}" for i in range(n)],
        "Max Score": maxes,
        "true_raw": true_raw,
    })
    base["true_score"] = base["true_raw"] / base["Max Score"]
    base["true_label"] = np.round(base["true_score"] * 4).clip(0, 4).astype(int)
    ref_raw = np.clip(true_raw + rng.integers(-2, 3, size=n), 0, maxes)
    noise = {"zeroshot_base": 0.6, "adapter_qlora4bit": 0.0, "merged_bf16": 0.05,
             "merged_nf4dequant": 0.02, "gguf_bf16": 0.06, "gguf_q8_0": 0.07,
             "gguf_q6_k": 0.09, "gguf_q5_k_m": 0.12, "gguf_q4_k_m": 0.18}
    sizes = {"merged_bf16": 9.3, "merged_nf4dequant": 9.3, "gguf_bf16": 8.4, "gguf_q8_0": 4.5,
             "gguf_q6_k": 3.5, "gguf_q5_k_m": 3.1, "gguf_q4_k_m": 2.7, "adapter_qlora4bit": 3.4}
    for variant, p in noise.items():
        flip = rng.random(n) < p
        raw = np.where(flip, np.clip(ref_raw + rng.choice([-2, -1, 1, 2], size=n), 0, maxes), ref_raw)
        df = base.copy()
        df["pred_raw"] = raw
        df["pred_score"] = raw / maxes
        devices = ["gpu", "cpu"] if variant.startswith("gguf_") else ["gpu"]
        for device in devices:
            d = root / release / f"{variant}__{device}"
            d.mkdir(parents=True, exist_ok=True)
            df.to_csv(d / "predictions_test.csv", index=False)
            speed = sizes.get(variant, 9) * (0.9 if device == "cpu" else 0.03)
            cost = {"device": device, "weights_gb": sizes.get(variant),
                    "seconds_per_answer_median": speed, "seconds_per_answer_p90": speed * 1.3,
                    "load_seconds": 5.0, "parse_failures": 0}
            (d / "validation.json").write_text(json.dumps({"cost": cost}), encoding="utf-8")


def resolve_reference(requested: str, release_bench: Path) -> str:
    """'auto' -> the merge run_release.sh chose for this release (CHOSEN_MERGE)."""
    if requested != "auto":
        return requested
    chosen = release_bench / "CHOSEN_MERGE"
    if chosen.is_file() and chosen.read_text().strip() == "nf4-dequant":
        return "merged_nf4dequant"
    return "merged_bf16"


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=tuple(RELEASES), default=["qwen"])
    parser.add_argument("--reference", default="auto",
                        help="Fidelity reference variant. 'auto' reads publish/benchmark/<release>/"
                             "CHOSEN_MERGE (written by run_release.sh), else merged_bf16.")
    parser.add_argument("--bench-root", type=Path, default=Path("publish/benchmark"))
    parser.add_argument("--gguf-root", type=Path, default=Path("publish/gguf"))
    parser.add_argument("--stats-dir", type=Path, default=Path("results_stats"))
    parser.add_argument("--docs-report", type=Path, default=Path("docs/pintu_release_report.md"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Run on a synthetic fixture in a temp folder (writes nothing real).")
    parser.add_argument("--dry-run-out", type=Path, default=None,
                        help="Keep dry-run outputs here instead of a deleted temp folder.")
    args = parser.parse_args()
    if args.reference != "auto" and args.reference not in ORDER:
        raise SystemExit(f"--reference must be 'auto' or one of {list(ORDER)}")

    merged_roots = {"merged_bf16": Path("publish/full_models"),
                    "merged_nf4dequant": Path("publish/full_models_nf4dequant")}
    tmp = None
    results_root = ROOT
    if args.dry_run:
        tmp = Path(tempfile.mkdtemp(prefix="exp15_dryrun_"))
        base = args.dry_run_out or tmp
        args.bench_root = base / "benchmark"
        args.stats_dir = base / "results_stats"
        args.docs_report = base / "docs" / "pintu_release_report.md"
        results_root = base  # no fallback runs in the fixture
        for key in args.models:
            make_fixture(args.bench_root, RELEASES[key][0])

    all_bench, all_fid = [], []
    for key in args.models:
        release, suffix = RELEASES[key]
        runs = load_runs(release, suffix, args.bench_root, results_root)
        if not runs:
            print(f"[{release}] no runs found under {args.bench_root / release}; skipped")
            continue
        reference = resolve_reference(args.reference, args.bench_root / release)
        print(f"[{release}] fidelity reference: {reference}")
        bench, fid = build_tables(release, runs, reference, args.gguf_root, merged_roots)
        if reference not in set(bench["variant"]):
            print(f"[{release}] reference {reference} not measured yet: fidelity is [pending]")
        fig_dir = args.stats_dir / "figures" / "pintu_release" / release
        figs = make_figures(release, bench, fid, runs, reference, fig_dir)

        hub_dir = args.bench_root / release
        hub_figs = hub_dir / "figures"
        hub_figs.mkdir(parents=True, exist_ok=True)
        for f in figs:
            shutil.copy2(f, hub_figs / f.name)
        bench.to_csv(hub_dir / "benchmark.csv", index=False)
        fid.to_csv(hub_dir / "fidelity.csv", index=False)
        write_report(release, bench, fid, reference, figs, "figures/",
                     hub_dir / "REPORT.md", args.dry_run)
        if len(args.models) == 1:
            # docs/ and results_stats/ are siblings in both real and dry runs.
            rel = f"../{args.stats_dir.name}/figures/pintu_release/{release}/"
            write_report(release, bench, fid, reference, figs, rel,
                         args.docs_report, args.dry_run)
        all_bench.append(bench)
        all_fid.append(fid)
        print(bench[["id", "device", "n", "qwk", "raw_exact", "raw_within1",
                     "match_rate_vs_ref", "changed_vs_ref", "weights_gb"]].to_string(index=False))

    if all_bench:
        args.stats_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(all_bench).to_csv(args.stats_dir / "pintu_release_benchmark.csv", index=False)
        pd.concat(all_fid).to_csv(args.stats_dir / "pintu_release_fidelity.csv", index=False)
        print(f"\nWrote {args.stats_dir / 'pintu_release_benchmark.csv'}")
    if args.dry_run:
        print(f"[dry-run] outputs in {args.dry_run_out or tmp}")
        if args.dry_run_out is None:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
