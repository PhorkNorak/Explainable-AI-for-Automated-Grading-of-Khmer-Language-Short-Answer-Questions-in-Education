"""Explainable-AI module for Khmer ASAG.

A model-agnostic explanation toolkit shared across all four model families
(classical, RNN, encoder, LLM). The explanation method is SHAP word attribution,
the single unified method used throughout (following Kumar & Boulanger, 2020).

  * explainers.py    — SHAP word attribution (and occlusion, a fast special case)
                       over Khmer word units.
  * attributions.py  — thin dispatch wrapper over the occlusion explainer.
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
    shap_importance,
)
from .plausibility import plausibility
from .attributions import word_importance

__all__ = [
    "tokenize_answer",
    "detokenize",
    "occlusion_importance",
    "shap_importance",
    "plausibility",
    "word_importance",
]
