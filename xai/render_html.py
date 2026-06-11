"""Render Khmer word-importance heatmaps as self-contained HTML.

Why HTML: matplotlib (and Pillow without libraqm) cannot *shape* Khmer — it draws
each codepoint left-to-right without stacking subscripts (COENG) or reordering
pre-base vowels, so correct Unicode looks broken in a PNG. Web browsers shape Khmer
correctly, so an HTML heatmap renders the script properly with no extra dependencies.
Open the file in any browser; print-to-PDF or screenshot for a figure.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

_KHMER_FONT_STACK = ("'Khmer OS', 'Khmer OS System', 'Noto Sans Khmer', "
                     "'Leelawadee UI', 'Khmer UI', sans-serif")


def _tile(word: str, s: float) -> str:
    # white -> red interpolation (s in [0,1]); matches the PNG colour scheme
    g = int(round(255 - (255 - 85) * s))
    bg = f"rgb(255,{g},{g})"
    return (f'<span style="display:inline-block;margin:2px;padding:6px 9px;'
            f'border:1px solid #ccc;border-radius:3px;background:{bg};'
            f'font-size:20px;line-height:1.6">{_esc(word)}</span>')


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def heatmap_html_fragment(words: List[str], importance, caption: str = "") -> str:
    """Return a self-contained HTML *string* (browser-shaped Khmer word heatmap).

    Unlike :func:`render_word_heatmap_html` this writes nothing to disk; it is meant
    to be embedded directly in a web component (e.g. Gradio ``gr.HTML``).
    """
    imp = np.clip(np.asarray(importance, dtype=np.float64), 0, None)
    imp = imp / (imp.max() + 1e-9) if len(imp) else imp
    tiles = "\n".join(_tile(w, float(s)) for w, s in zip(words, imp))
    head = f'<div style="font-family:sans-serif;font-size:13px;color:#2c5282;margin-bottom:4px">{_esc(caption)}</div>' if caption else ""
    return (f'<div style="font-family:{_KHMER_FONT_STACK}">{head}<div>{tiles}</div>'
            f'<div style="font-family:sans-serif;font-size:12px;color:#555;margin-top:8px">'
            f'low <span style="display:inline-block;width:90px;height:11px;vertical-align:middle;'
            f'background:linear-gradient(to right,#fff,rgb(255,85,85));border:1px solid #ccc"></span> high '
            f'&nbsp;—&nbsp; darker = removing this word changes the score more.</div></div>')


def render_word_heatmap_html(words: List[str], importance, out_path: str, title: str):
    """Write a standalone HTML heatmap (browser-shaped Khmer)."""
    imp = np.clip(np.asarray(importance, dtype=np.float64), 0, None)
    imp = imp / (imp.max() + 1e-9) if len(imp) else imp
    tiles = "\n".join(_tile(w, float(s)) for w, s in zip(words, imp))
    html = f"""<!doctype html><html lang="km"><head><meta charset="utf-8">
<title>{_esc(title)}</title>
<style>body{{font-family:{_KHMER_FONT_STACK};margin:24px;background:#fff;color:#1a202c}}
h3{{font-size:16px;color:#2c5282;font-family:sans-serif}}
.legend{{font-family:sans-serif;font-size:12px;color:#555;margin-top:14px}}
.bar{{display:inline-block;width:120px;height:12px;vertical-align:middle;
background:linear-gradient(to right,#fff,rgb(255,85,85));border:1px solid #ccc}}</style>
</head><body>
<h3>{_esc(title)}</h3>
<div>{tiles}</div>
<div class="legend">word importance (occlusion): low <span class="bar"></span> high &nbsp;|&nbsp;
darker red = removing this Khmer word changes the predicted score more.</div>
</body></html>"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def render_gallery_html(items, out_path: str, title: str):
    """Combine several heatmaps (each a dict: words, importance, caption) into one page."""
    blocks = []
    for it in items:
        imp = np.clip(np.asarray(it["importance"], dtype=np.float64), 0, None)
        imp = imp / (imp.max() + 1e-9) if len(imp) else imp
        tiles = "\n".join(_tile(w, float(s)) for w, s in zip(it["words"], imp))
        blocks.append(f'<h3>{_esc(it["caption"])}</h3><div>{tiles}</div>')
    body = "\n<hr>\n".join(blocks)
    html = f"""<!doctype html><html lang="km"><head><meta charset="utf-8">
<title>{_esc(title)}</title>
<style>body{{font-family:{_KHMER_FONT_STACK};margin:24px;background:#fff;color:#1a202c}}
h2,h3{{font-family:sans-serif;color:#2c5282}} hr{{margin:18px 0;border:none;border-top:1px solid #eee}}
span.tile{{}}</style></head><body>
<h2>{_esc(title)}</h2>
{body}
<p style="font-family:sans-serif;font-size:12px;color:#555">Browser-shaped Khmer
(occlusion importance on the original answer). Print-to-PDF or screenshot for a figure.</p>
</body></html>"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
