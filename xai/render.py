"""Visualization of explanations: Khmer word-importance heatmaps + rationale cards.

Renders the answer as a strip of word tiles colored white→red by importance, with
the true and predicted score in the title. Adapted from the original token-saliency
renderer (``xai.py``) but operating on word units so the picture is readable to a
Khmer-speaking teacher.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as _fm


def _set_khmer_font():
    """Pick an installed Khmer-capable font so heatmap glyphs render correctly."""
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in ("Khmer OS", "Khmer OS System", "Khmer UI", "Noto Sans Khmer",
                 "Khmer OS Content", "Leelawadee UI"):
        if name in available:
            plt.rcParams["font.family"] = name
            return name
    return None


_KHMER_FONT = _set_khmer_font()


def render_word_heatmap(
    words: List[str],
    importance: np.ndarray,
    out_path: str,
    title: str,
    cols: int = 12,
):
    """Save a word-tile heatmap PNG colored by (positive) importance magnitude."""
    if len(words) == 0:
        return
    imp = np.asarray(importance, dtype=np.float64)
    imp = np.clip(imp, 0, None)  # show supporting evidence (positive contributions)
    imp = imp / (imp.max() + 1e-9)

    cmap = LinearSegmentedColormap.from_list("imp", ["#ffffff", "#ff5555"])
    n = len(words)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(cols * 1.1, rows * 0.6 + 0.7))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows + 0.5)
    ax.axis("off")
    for i, (tok, s) in enumerate(zip(words, imp)):
        r = rows - 1 - (i // cols)
        c = i % cols
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color=cmap(float(s)),
                                   ec="#cccccc", lw=0.5))
        ax.text(c + 0.5, r + 0.5, tok, ha="center", va="center", fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def write_rationale_card(out_path: str, record: dict):
    """Persist an LLM rationale (or any text explanation) as a small JSON card."""
    import json
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
