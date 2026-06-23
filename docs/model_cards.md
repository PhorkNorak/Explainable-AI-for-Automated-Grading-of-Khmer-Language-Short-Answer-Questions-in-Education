# KhmerGrader: fine-tuned LLMs for Khmer short-answer grading

KhmerGrader is a small family of open-source LLMs **fine-tuned by this thesis** (QLoRA) to grade
Khmer secondary-school short answers, predicting an integer score from a question, a reference
answer, and a student answer. The family exists to test one question across different starting
points: does the choice of base model (general multilingual vs region-adapted) matter for Khmer
grading once you fine-tune?

**What we claim.** The fine-tuned adapters, the training recipe, the Khmer grading data, and the
benchmark results are our contribution. We do **not** claim the base models; each KhmerGrader model
is a derivative of an existing open model and is named to disclose its lineage, following the
SEA-LION convention (`Gemma-SEA-LION-v3`, `Qwen-SEA-LION-v4`) and prior domain fine-tunes
(SeaLLMs-v3 on Qwen2, ChatDoctor on LLaMA).

## The family

| Released name | Base model (HF id) | Base license | Params | Notes |
|---|---|---|---|---|
| **Qwen-KhmerGrader-4B** | `Qwen/Qwen3.5-4B` | Apache 2.0 | 4B | general multilingual base |
| **Gemma-KhmerGrader-4B** | `google/gemma-4-E4B-it` | Apache 2.0 | 4B effective | general multilingual instruction-tuned base |
| **SEA-LION-KhmerGrader-E2B** | `aisingapore/Gemma-SEA-LION-v4.5-E2B-IT` | MIT | 2.3B effective | Southeast-Asian-adapted base (Gemma 4 E2B underneath) |

Lineage for SEA-LION-KhmerGrader-E2B is two layers deep: our fine-tune ← Gemma-SEA-LION-v4.5-E2B-IT
(AI Singapore) ← Gemma 4 E2B (Google). Credit both upstream providers. Note that SEA-LION v4.5-E2B
was adapted on Burmese, Indonesian, Tagalog, Malay, Tamil, Thai, and Vietnamese but **not Khmer**
specifically, so whether its regional adaptation transfers to Khmer is an empirical question this
family is meant to answer, not an assumption.

## Method (identical across the family)

- **Adaptation:** QLoRA, 4-bit base, LoRA rank 16, alpha 16, dropout 0, target modules
  `q,k,v,o,gate,up,down` projections (see `experiments/exp08_llm_finetune.py`).
- **Objective:** completion-only supervised fine-tuning. The prompt is rendered with the model's
  own chat template with reasoning disabled, and only the integer-score completion is supervised.
- **Inference:** greedy decode, reasoning disabled, parse the integer, clip to `[0, max_score]`.
- **Data:** the Khmer ASAG corpus, `clean` preprocessing, `qar` input (question + answer vs
  reference). Training/validation/test use the project's seed-42 stratified split.

## Results

Each model is compared against its **own untuned base evaluated zero-shot** on the same test set,
so the reported lift isolates the effect of fine-tuning (the standard, honest baseline).

| Model | Base zero-shot QWK | Fine-tuned QWK | Lift | Exact |
|---|---|---|---|---|
| Qwen-KhmerGrader-4B | 0.500 | **0.843** | +0.34 | 0.657 |
| Gemma-KhmerGrader-4B | [pending] | 0.763 | [pending] | 0.657 |
| SEA-LION-KhmerGrader-E2B | 0.541 | 0.802 | +0.26 | **0.693** |

All values are the `no10c` (909) test set. Sourced from
`results_no10c_v08z_llm_<model>_zeroshot/leaderboard.csv` (zero-shot baseline) and
`results_no10c_v08_llm_<model>/leaderboard.csv` (fine-tuned); `Exact` is `test_raw_exact`. The
figure `paper/figures/fig_llm_finetune_gain.png` visualizes this table. Do not hand-edit these
numbers; they trace to the result files above.

## Intended use and limitations

- **Intended use:** assistive grading of Khmer secondary-school short answers against a reference,
  with a human in the loop. Research and educational use.
- **Not for:** high-stakes automated grading without human review, or languages/domains outside the
  training corpus.
- **Known limitation:** all labels reflect a single grader, and the corpus is small (41 questions,
  4 subjects) and skewed toward full credit, so deployment should keep a teacher in the loop.

## Attribution and license notices

- **Qwen** base under Apache 2.0; retain the upstream copyright and license notice.
- **Gemma** base: per the model card, Apache 2.0; retain Google's copyright, license, and NOTICE,
  and mark modifications. Verify the license on the live model card before redistributing weights.
- **SEA-LION** under MIT (AI Singapore); the underlying Gemma notice still applies. Credit AI
  Singapore and Google.
- Disclose the `base_model:` in any released model card (HuggingFace convention).
