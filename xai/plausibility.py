"""Plausibility proxy (reference overlap).

Following the faithfulness-vs-plausibility distinction (Jacovi & Goldberg, 2020),
*plausibility* asks whether an explanation looks reasonable to a human. A teacher
grades a short answer by checking whether it contains the key content of the
reference answer, so a plausible explanation should highlight answer words that also
appear in the reference. We measure the overlap between the explainer's top-k
important answer words and the reference-answer content words.

This is a cheap, fully-automatic proxy, not a substitute for a human-rationale study,
but a defensible quantitative signal computable for every model family. An optional
human spot-check can be layered on top.
"""

from __future__ import annotations

from typing import List

import numpy as np


def _reference_content_words(reference_proc: str, preprocess_mode: str) -> set:
    """Content-word set of the reference answer (same segmenter as the answer side)."""
    if not reference_proc:
        return set()
    if preprocess_mode == "segment":
        toks = reference_proc.split(" ")
    else:
        from .explainers import tokenize_answer
        toks = tokenize_answer(reference_proc, preprocess_mode)
    return {t.strip() for t in toks if len(t.strip()) > 1}  # drop 1-char tokens


def plausibility(
    words: List[str],
    importance: np.ndarray,
    reference_proc: str,
    preprocess_mode: str,
    fraction: float = 0.2,
) -> float:
    """Reference-overlap plausibility: fraction of the explainer's top-k answer words
    that occur in the reference answer.

    Selects max(1, round(n * fraction)) tokens by signed attribution, with ties
    resolved in answer order. One-character reference tokens are excluded.
    Returns overlap in [0, 1], not faithfulness or educational validity.
    """
    importance = np.asarray(importance, dtype=float)
    if importance.shape != (len(words),) or not np.all(np.isfinite(importance)):
        raise ValueError("importance must be finite and aligned with words")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if not words:
        return 0.0
    ref = _reference_content_words(reference_proc, preprocess_mode)
    if not ref:
        return 0.0
    k = max(1, int(round(len(words) * fraction)))
    top = np.argsort(-importance, kind="stable")[:k]
    hits = sum(1 for i in top if words[i].strip() in ref)
    return hits / float(len(top))
