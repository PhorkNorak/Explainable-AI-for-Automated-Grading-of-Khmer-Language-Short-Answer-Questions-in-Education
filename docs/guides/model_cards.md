# KhmerGrader: fine-tuned LLMs for Khmer short-answer grading

**KhmerGrader** is the QLoRA-fine-tuned family studied in the [final report](<../reports/Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education.pdf>). This page covers that report version only. The contribution is adaptation for Khmer short-answer grading, not the upstream base models. Student data is private; this page does not claim that model weights have been publicly released.

## Models and lineage

| Final-report name | Base model | Reported size |
|---|---|---|
| Qwen-KhmerGrader-4B | Qwen 3.5 4B | 4B |
| Gemma-KhmerGrader-4B | Gemma 4 E4B | 4B |
| SEA-LION-KhmerGrader-E2B | Gemma-SEA-LION v4.5 E2B | 2.3B |

SEA-LION lineage includes AI Singapore's adaptation of Google's Gemma; credit both upstream providers. Configure the actual model name offered by a compatible server. A research model name is not a verified repository ID or download location.

## Configured recipe (historical execution unverified)

- **Adaptation:** QLoRA, 4-bit base, LoRA rank 16, alpha 16, dropout 0, target modules
  `q,k,v,o,gate,up,down` projections (see `experiments/exp08_llm_finetune.py`).
- **Objective:** completion-only supervised fine-tuning. The prompt is rendered with the model's
  own chat template with reasoning disabled, and only the integer-score completion is supervised.
- **Inference:** greedy decode, reasoning disabled, parse the integer, clip to `[0, max_score]`.
- **Data:** the Khmer ASAG corpus, `clean` preprocessing, `qar` input (question + answer vs
  reference). Training/validation/test use the project's seed-42 stratified split.

Official model cards: [Qwen](https://huggingface.co/Qwen/Qwen3.5-4B), [Gemma](https://huggingface.co/google/gemma-4-E4B-it), [SEA-LION](https://huggingface.co/aisingapore/Gemma-SEA-LION-v4.5-E2B-IT). IDs and displayed licences checked 2026-09-20; this does not verify the local adapters or publication state.

## Final-report results

Report Table 12 states:

| Model | Fine-tuned QWK |
|---|---:|
| Qwen-KhmerGrader-4B | 0.850 |
| Gemma-KhmerGrader-4B | 0.844 |
| SEA-LION-KhmerGrader-E2B | 0.713 |

Qwen is the reported QWK champion. Table 10 gives its zero-shot reference QWK as 0.529 and its fine-tuned exact agreement as 0.67. The latest family table does not report exact-match rankings or pending zero-shot cells for Gemma and SEA-LION. These are report-stated figures, not a new reproduction of the three adapters. See [report-to-evidence notes](report_evidence.md) for the summary match and unresolved prediction provenance.

New training requires the unresolved source-score decision and the [isolated HPC workflow](hpc.md). Preserve existing predictions and summaries.

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
