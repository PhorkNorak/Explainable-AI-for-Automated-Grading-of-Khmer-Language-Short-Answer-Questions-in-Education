# Defense questions for the final report

Use the [June 2026 final report](<../reports/Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education Report final.pdf>) as the reference. [Evidence notes](report_evidence.md) distinguish report findings from unresolved reproduction issues.

**What is the contribution?** A Khmer short-answer grading study covering four model families, the QLoRA-fine-tuned KhmerGrader family, SHAP word attribution and a prototype.

**Which model has the highest QWK?** Report Table 10 identifies Qwen-KhmerGrader-4B at 0.850, versus Classical 0.802, BiLSTM 0.789 and GTE 0.777. These figures match the champion summary. The encoder's different test set makes its comparison indicative.

**What is the exact-mark agreement?** Latest Table 10 gives Classical 0.28, BiLSTM 0.34, GTE 0.49 and Qwen-KhmerGrader 0.67. Qwen within-one agreement is 0.80. The latest family table contains fine-tuned QWK only, so do not carry over an older family exact-match ranking.

**Does fine-tuning help?** The report gives Qwen zero-shot QWK 0.529 and fine-tuned QWK 0.850. Table 12 gives fine-tuned QWK 0.844 for Gemma and 0.713 for SEA-LION; it has no pending baseline columns.

**Does the specialist beat frontier models?** The report says it does on this benchmark: 0.850 versus the best reported frontier QWK of 0.734. The saved summary matches those frontier figures, but parsed sample counts differ and no fresh provider/API verification was performed. This does not establish general superiority.

**Are the champion results independently reproduced?** Not as one coherent champion set. The summary matches the report, but the historical audit recomputed saved predictions and identified conflicting values and unresolved checkpoint/preprocessing provenance. That audit remains available; it is not a replacement final-report experiment.

**What is SHAP plausibility?** The fraction of the top 20% SHAP words appearing in the reference answer. Latest Table 14 gives 0.63 / 0.60 / 0.68 / 0.61 for Classical / BiLSTM / Encoder / LLM, matching the saved summary after rounding. Sample sizes are 66 / 66 / 18 / 18, so the highest displayed encoder value is not a common-sample ranking.

**What does the efficiency table measure?** Table 11 reports score-plus-SHAP latency of approximately 2 / 13 / 10 / 76 seconds per answer. These are capped-explanation timings, not score-only measurements.

**Why not use attention as the explanation?** Attention is part of the BiLSTM architecture. SHAP is the shared model-agnostic explanation method, including for the non-differentiable SVR.

**Does overlap prove educational value?** No. It is a lexical proxy, not teacher judgement or a classroom study. Agreement with the grader also does not establish independent correctness.

**What are the data limits?** One grader, 1,184 answers, 41 questions, 203 students, four subjects and two schools. The no10c variant has 909 answers and retains grade 0. The recorded 19/15 score requires source confirmation before affected results are regenerated. Private data access and consent remain separate from source-code licensing.

**Can teachers use the prototype?** It demonstrates scoring, optional SHAP highlights and feedback, with a teacher retaining the decision. Feedback uses an open-source model endpoint with rule-based fallback. No public demo URL or classroom field study is verified.

**What remains unresolved?** The final report has no pending-result labels. Compatible checkpoint/XAI provenance, source-score confirmation, consent/signatures and Khmer proofreading remain separate evidence issues. Appendix G includes the prototype screenshot from the final presentation; a classroom field study remains uncompleted. Programme, June 2026 date and committee names are already printed in the final PDF. Manuscript sources stay local, and the supplied presentation PDF has not been regenerated from the Markdown decks.
