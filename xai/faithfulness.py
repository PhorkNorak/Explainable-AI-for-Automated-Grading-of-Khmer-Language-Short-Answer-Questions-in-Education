"""ERASER-style faithfulness metrics (DeYoung et al., 2020).

Given a per-word ``importance`` vector and a ``predict_fn(answer, reference) ->
score``, we measure whether the words the explainer calls important actually drive
the model's score:

  * **comprehensiveness** = score(full) - score(answer with top-k words REMOVED).
    If the important words matter, removing them should drop the score a lot →
    *higher is more faithful*.

  * **sufficiency**       = score(full) - score(answer with ONLY top-k words kept).
    If the important words are enough, keeping only them should preserve the score →
    *lower (closer to 0) is more faithful*.

Both are model-agnostic, so the same numbers are directly comparable across the
classical, RNN, encoder, and LLM families. The occlusion explainer and these
metrics share the same perturbation, so the explanation and its evaluation are
self-consistent.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

from .explainers import detokenize


def _topk_idx(importance: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(importance)[::-1][:k]


def _k_from_fraction(n: int, fraction: float) -> int:
    return max(1, int(round(n * fraction)))


def comprehensiveness(
    predict_fn: Callable[[str, str], float],
    words: List[str],
    importance: np.ndarray,
    reference_proc: str,
    preprocess_mode: str,
    fraction: float = 0.2,
) -> float:
    if not words:
        return 0.0
    full = float(predict_fn(detokenize(words, preprocess_mode), reference_proc))
    k = _k_from_fraction(len(words), fraction)
    drop = set(_topk_idx(importance, k).tolist())
    kept = [w for i, w in enumerate(words) if i not in drop]
    reduced = float(predict_fn(detokenize(kept, preprocess_mode), reference_proc))
    return full - reduced


def sufficiency(
    predict_fn: Callable[[str, str], float],
    words: List[str],
    importance: np.ndarray,
    reference_proc: str,
    preprocess_mode: str,
    fraction: float = 0.2,
) -> float:
    if not words:
        return 0.0
    full = float(predict_fn(detokenize(words, preprocess_mode), reference_proc))
    k = _k_from_fraction(len(words), fraction)
    keep = _topk_idx(importance, k)
    kept = [words[i] for i in sorted(keep.tolist())]
    only = float(predict_fn(detokenize(kept, preprocess_mode), reference_proc))
    return full - only


def comprehensiveness_random(
    predict_fn: Callable[[str, str], float],
    words: List[str],
    reference_proc: str,
    preprocess_mode: str,
    fraction: float = 0.2,
    seed: int = 42,
) -> float:
    """Comprehensiveness when removing *random* words — the sanity baseline.

    A faithful explainer must beat this: removing its top words should hurt the
    score more than removing random words.
    """
    if not words:
        return 0.0
    rng = np.random.default_rng(seed)
    full = float(predict_fn(detokenize(words, preprocess_mode), reference_proc))
    k = _k_from_fraction(len(words), fraction)
    drop = set(rng.choice(len(words), size=k, replace=False).tolist())
    kept = [w for i, w in enumerate(words) if i not in drop]
    reduced = float(predict_fn(detokenize(kept, preprocess_mode), reference_proc))
    return full - reduced


DEFAULT_KGRID = (0.1, 0.2, 0.3, 0.4, 0.5)


def faithfulness_report(
    predict_fn: Callable[[str, str], float],
    per_instance,
    preprocess_mode: str,
    fraction: float = 0.2,
    kgrid=DEFAULT_KGRID,
) -> dict:
    """Aggregate comprehensiveness / sufficiency (+ random baseline) over a set.

    ``per_instance`` is an iterable of ``(words, importance, reference_proc)``.
    Reports the single-``fraction`` numbers AND the **AOPC** (area over the
    perturbation curve) — comprehensiveness/sufficiency averaged across ``kgrid``
    (ERASER-style), which is more robust than a single top-k cutoff.
    """
    comp, suff, comp_rand = [], [], []
    aopc_comp_rows, aopc_suff_rows = [], []
    for words, importance, ref in per_instance:
        if not words:
            continue
        comp.append(comprehensiveness(predict_fn, words, importance, ref, preprocess_mode, fraction))
        suff.append(sufficiency(predict_fn, words, importance, ref, preprocess_mode, fraction))
        comp_rand.append(comprehensiveness_random(predict_fn, words, ref, preprocess_mode, fraction))
        aopc_comp_rows.append(np.mean([
            comprehensiveness(predict_fn, words, importance, ref, preprocess_mode, k) for k in kgrid]))
        aopc_suff_rows.append(np.mean([
            sufficiency(predict_fn, words, importance, ref, preprocess_mode, k) for k in kgrid]))
    if not comp:
        return {"n": 0}
    comp_m = float(np.mean(comp))
    rand_m = float(np.mean(comp_rand))
    return {
        "n": len(comp),
        "fraction": fraction,
        "comprehensiveness": comp_m,
        "sufficiency": float(np.mean(suff)),
        "comprehensiveness_random": rand_m,
        "faithfulness_gap": comp_m - rand_m,  # >0 means explainer beats random
        "aopc_comprehensiveness": float(np.mean(aopc_comp_rows)),
        "aopc_sufficiency": float(np.mean(aopc_suff_rows)),
    }
