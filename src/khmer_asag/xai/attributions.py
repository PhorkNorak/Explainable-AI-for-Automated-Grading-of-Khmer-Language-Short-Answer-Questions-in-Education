"""Occlusion (leave-one-out) word attribution — legacy thin wrapper.

Kept for call-site compatibility. The headline explanation method for the project is
SHAP (see ``xai.explainers.shap_importance``); this occlusion wrapper remains as a fast,
deterministic, model-agnostic alternative.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .explainers import occlusion_importance


def word_importance(
    predict_one: Callable[[str, str], float],
    answer_proc: str,
    reference_proc: str,
    preprocess_mode: str,
) -> Tuple[List[str], np.ndarray]:
    """Leave-One-Out word attribution for any grading model.

    ``predict_one(answer, reference) -> float`` is the sole requirement — no
    gradients, no internal access, works for SVR, BiLSTM, Transformer, and LLM.
    Returns ``(words, importance)`` where ``importance[i] = score(full) -
    score(answer without words[i])``.
    """
    return occlusion_importance(predict_one, answer_proc, reference_proc, preprocess_mode)
