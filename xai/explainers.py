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
    """Shapley-value attribution of each answer word, as a comparison to LOO.

    The baseline coalition is the empty answer (all words removed); the Shapley
    value of word ``i`` is its average marginal contribution to the predicted
    score across coalitions, which uses the same remove-a-word mechanism as LOO
    and so is directly comparable. Uses the ``shap`` library if available
    (Permutation explainer, background = all words removed) and otherwise a
    self-contained Monte-Carlo permutation estimator, so it always runs.
    """
    words = tokenize_answer(answer_proc, preprocess_mode)
    n = len(words)
    if n == 0:
        return [], np.zeros(0, dtype=np.float64)
    if n == 1:
        full = float(predict_fn(answer_proc, reference_proc))
        empty = float(predict_fn(detokenize([], preprocess_mode), reference_proc))
        return words, np.array([full - empty], dtype=np.float64)

    def f(masks):  # masks: (B, n) of 0/1; keep word where >= 0.5
        masks = np.asarray(masks)
        out = np.empty(masks.shape[0], dtype=np.float64)
        for r in range(masks.shape[0]):
            kept = [w for w, keep in zip(words, masks[r]) if keep >= 0.5]
            out[r] = float(predict_fn(detokenize(kept, preprocess_mode), reference_proc))
        return out

    # Preferred: the SHAP library (Permutation explainer, all-removed background).
    try:
        import shap  # noqa: F401
        np.random.seed(seed)
        background = np.zeros((1, n), dtype=np.float64)
        explainer = shap.Explainer(f, background)
        me = max_evals if max_evals is not None else max(2 * n + 1, 100)
        sv = explainer(np.ones((1, n), dtype=np.float64), max_evals=me)
        return words, np.asarray(sv.values[0], dtype=np.float64)
    except Exception:
        pass

    # Fallback: Monte-Carlo permutation Shapley (same maths, no extra deps).
    rng = np.random.default_rng(seed)
    phi = np.zeros(n, dtype=np.float64)
    base = float(predict_fn(detokenize([], preprocess_mode), reference_proc))
    for _ in range(n_perm):
        order = rng.permutation(n)
        present = np.zeros(n, dtype=bool)
        prev = base
        for idx in order:
            present[idx] = True
            kept = [w for w, p in zip(words, present) if p]
            s = float(predict_fn(detokenize(kept, preprocess_mode), reference_proc))
            phi[idx] += s - prev
            prev = s
    phi /= n_perm
    return words, phi
