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
        raise RuntimeError("Khmer word segmentation is required for word attribution")


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


# ────────────────────────────────────────────────────────────────────────────
# SHAP importance — Shapley values over the answer word units. This is the headline
# attribution method for the project; occlusion above is kept as a fast special case.
# Returns the same (words, imp) shape and sign convention (positive = the word
# supported the grade), so it plugs straight into plausibility.py.
# ────────────────────────────────────────────────────────────────────────────


def shap_importance(
    predict_fn: Callable[[str, str], float],
    answer_proc: str,
    reference_proc: str,
    preprocess_mode: str,
    max_evals: int | None = None,
    n_perm: int = 32,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray]:
    """Permutation-sampled SHAP values over answer words, empty-answer baseline.

    Uses a seeded Monte Carlo permutation estimator with no dependency-dependent
    backend switch. Each complete permutation telescopes to full minus empty.
    ``max_evals`` is a hard bound on model calls, including the empty baseline.
    A budget smaller than n_words + 1 cannot cover one complete permutation and
    raises before scoring. Model failures propagate; they are never replaced by
    occlusion or a second estimator.
    """
    words = tokenize_answer(answer_proc, preprocess_mode)
    n = len(words)
    if n == 0:
        return [], np.zeros(0, dtype=np.float64)
    # Preserve actual separators so the full coalition is the graded string.
    # Repeated words are aligned by position, not by membership in a word set.
    pieces = []
    cursor = 0
    for word in words:
        start = answer_proc.find(word, cursor)
        if start < 0 or answer_proc[cursor:start].strip():
            raise ValueError("word tokens do not align with the graded answer")
        pieces.append(answer_proc[cursor:start] + word)
        cursor = start + len(word)
    tail = answer_proc[cursor:]
    if tail.strip():
        raise ValueError("word tokens omit answer content")
    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    if max_evals is not None and max_evals < n + 1:
        raise ValueError(f"SHAP needs at least {n + 1} evaluations for {n} words")
    n_perm_eff = n_perm if max_evals is None else min(n_perm, (int(max_evals) - 1) // n)
    rng = np.random.default_rng(seed)
    phi = np.zeros(n, dtype=np.float64)

    def score(mask):
        text = "".join(piece for piece, keep in zip(pieces, mask) if keep)
        if any(mask):
            text += tail
        value = float(predict_fn(text, reference_proc))
        if not np.isfinite(value):
            raise ValueError("SHAP predictor returned a non-finite score")
        return value

    base = score(np.zeros(n, dtype=bool))
    for _ in range(n_perm_eff):
        order = rng.permutation(n)
        present = np.zeros(n, dtype=bool)
        prev = base
        for idx in order:
            present[idx] = True
            value = score(present)
            phi[idx] += value - prev
            prev = value
    phi /= n_perm_eff
    return words, phi
