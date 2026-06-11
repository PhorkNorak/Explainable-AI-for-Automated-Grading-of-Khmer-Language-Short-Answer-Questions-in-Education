"""Explainable-AI module for Khmer ASAG.

A model-agnostic explanation + faithfulness toolkit shared across all four model
families (classical, RNN, encoder, LLM). The explanation method is Leave-One-Out
(LOO) occlusion word attribution — the single unified method used throughout.

  * explainers.py    — LOO occlusion importance over Khmer word units.
  * attributions.py  — thin dispatch wrapper; calls occlusion_importance.
  * faithfulness.py  — ERASER-style comprehensiveness & sufficiency, computed from
                       any `predict_fn(answer, reference) -> score`. The same
                       perturbation defines both the explanation and the metric, so
                       explainer and evaluator are self-consistent.
  * plausibility.py  — reference-overlap plausibility proxy (overlap of important
                       answer words with reference-answer content words).
  * render.py        — token-heatmap PNGs and rationale cards.

Everything operates on the (answer, reference) text pair already used by the
grading pipeline, so explanations are directly comparable across families.
"""

from .explainers import (
    tokenize_answer,
    detokenize,
    occlusion_importance,
)
from .faithfulness import comprehensiveness, sufficiency, faithfulness_report
from .plausibility import plausibility
from .attributions import word_importance

__all__ = [
    "tokenize_answer",
    "detokenize",
    "occlusion_importance",
    "comprehensiveness",
    "sufficiency",
    "faithfulness_report",
    "plausibility",
    "word_importance",
]
