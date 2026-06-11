"""LOO (Leave-One-Out) word attribution — the unified explanation method.

Thin dispatch wrapper kept for call-site compatibility. All explanation in this
project uses occlusion (LOO): drop a word, measure the score change. It is
model-agnostic, deterministic, and directly aligned with the ERASER faithfulness
metrics used to evaluate it.
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
