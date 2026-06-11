"""Leave-One-Out (LOO) occlusion attribution over Khmer word units.

Returns a parallel pair ``(words, importance)`` where ``words`` is a list of
human-readable Khmer word tokens of the *answer* and ``importance[i]`` is the
LOO attribution of ``words[i]``: score(full) - score(answer without word i).

Word units:
  * ``segment`` preprocess → words are already space-delimited (split on space).
  * ``raw`` / ``clean``     → words come from ``khmernltk.word_tokenize`` (the same
                             segmenter the pipeline uses), since clean Khmer text
                             has no whitespace word boundaries.

The reference side is held fixed throughout — we only attribute the answer, which
is what a teacher grades.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Tokenization helpers (answer-side word units)
# ────────────────────────────────────────────────────────────────────────────


def tokenize_answer(answer_proc: str, preprocess_mode: str) -> List[str]:
    """Split a preprocessed answer string into Khmer word units."""
    if not answer_proc:
        return []
    if preprocess_mode == "segment":
        return [w for w in answer_proc.split(" ") if w]
    # raw / clean: no whitespace boundaries → use the pipeline's segmenter
    try:
        import khmernltk
        return [w for w in khmernltk.word_tokenize(answer_proc) if w.strip()]
    except Exception:
        # Fallback: character units (still valid, just finer-grained)
        return list(answer_proc)


def detokenize(words: List[str], preprocess_mode: str) -> str:
    """Inverse of :func:`tokenize_answer` for the model's input format."""
    if preprocess_mode == "segment":
        return " ".join(words)
    return "".join(words)


# ────────────────────────────────────────────────────────────────────────────
# Occlusion importance — model-agnostic, the unifying explainer
# ────────────────────────────────────────────────────────────────────────────


def occlusion_importance(
    predict_fn: Callable[[str, str], float],
    answer_proc: str,
    reference_proc: str,
    preprocess_mode: str,
) -> Tuple[List[str], np.ndarray]:
    """Leave-one-word-out occlusion attribution.

    ``importance[i] = score(full) - score(answer without word i)``. A positive
    value means removing the word *lowered* the score, i.e. the word supported the
    grade. Works for any model exposing ``predict_fn(answer, reference) -> score``.
    """
    words = tokenize_answer(answer_proc, preprocess_mode)
    if not words:
        return [], np.zeros(0, dtype=np.float64)
    full = float(predict_fn(answer_proc, reference_proc))
    imp = np.zeros(len(words), dtype=np.float64)
    for i in range(len(words)):
        masked = detokenize(words[:i] + words[i + 1:], preprocess_mode)
        imp[i] = full - float(predict_fn(masked, reference_proc))
    return words, imp
