# Speaker notes for the final-report presentation

These notes follow the 11 slides in [the active deck](slide_final.md). The [final report](<../reports/Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education.pdf>) is the reference; [evidence notes](report_evidence.md) record unresolved differences.

## Slide 1: Study and final report

Introduce the undergraduate report and its teacher-assist purpose. This presentation follows the designated June 2026 final PDF. It does not include subsequent experiments. The report's findings and unresolved evidence differences are kept distinct.

## Slide 2: Three research questions

1. **RQ1: Which model family gives the best agreement with the human teacher?**
2. **RQ2: Which model is most useful for classroom deployment based on exact mark and within-one-point agreement?**
3. **RQ3: Can SHAP provide understandable word-level explanations across all model families?**

Explain why QWK, exact marks and explanations answer different questions. QWK describes ordinal agreement. Raw-mark metrics describe how close the output is to the teacher's mark. SHAP provides words the teacher can inspect, while reference overlap is only a proxy for explanation plausibility.

## Slide 3: Khmer corpus

Describe the manually digitised examination answers, reference answers and teacher marks. Question maxima vary, so marks are normalised. Distinguish the 909-answer dataset from its 137-answer test split. The encoder headline uses the 178-answer full test set. Do not claim the corpus is publicly downloadable.

## Slide 4: Models and scoring

Explain the common output scale without claiming identical training objectives: SVR uses its regression objective, neural regression heads use MSE, and LLM fine-tuning supervises integer-score completions. Preprocessing strips invisible characters, applies NFC and strips punctuation; segmentation uses khmernltk. There is no custom KCC reorder.

## Slide 5: Reported agreement

The final report identifies Qwen-KhmerGrader as the highest QWK performer. Explain that the table matches the summary CSV, but the audit did not reproduce a coherent champion set from saved continuous predictions. Do not call the summary independently reproduced or describe the difference as statistically significant. The encoder is evaluated on a different dataset variant.

## Slide 6: Raw-mark agreement

Describe exact match as reproducing the teacher's integer mark and within-one as an absolute error of at most one raw point. The report presents the LLM as the deployment leader. The latest Table 10 reports exact agreement 0.28 / 0.34 / 0.49 / 0.67 for the four pillars. Qwen within-one agreement is 0.80 at report precision. Do not claim a field study demonstrated classroom benefit.

## Slide 7: KhmerGrader family

Keep the family comparison separate from the four-pillar comparison, whose LLM representative is Qwen. The latest Table 12 contains fine-tuned QWK only: 0.850 / 0.844 / 0.713. Qwen's zero-shot reference is 0.529 in Table 10, a gain of approximately 0.32. Do not restore the older PDF's family exact-match or pending baseline columns.

## Slide 8: Frontier comparison

The report's specialist-versus-frontier finding is limited to this benchmark. The best reported frontier QWK is 0.734. Saved reasoning results match the report's frontier QWK values, but different parse rates change the evaluated sample counts. The latest report supplies the full breakdown and states an evaluation date of 24 June 2026. This is report-stated, not newly verified against providers. Do not infer general superiority beyond this study.

## Slide 9: SHAP word attribution

The latest Table 14 reports 0.63 / 0.60 / 0.68 / 0.61, matching the saved values 0.6285 / 0.5985 / 0.6842 / 0.6053 after rounding. Sample sizes are 66 / 66 / 18 / 18, so the encoder's highest displayed value does not establish a common-sample ranking. All four values are populated. The historical audit's estimator/checkpoint concerns remain separate from the numerical match. Reference overlap is not a human evaluation of pedagogical usefulness.

## Slide 10: Teacher-facing prototype

Present the prototype as a demonstration of teacher assistance. The teacher retains the grading decision. SHAP adds inference cost, particularly for generative models, so explanation is optional and budgeted. Table 11 reports approximately 2 / 13 / 10 / 76 seconds per answer with SHAP. These are explanation-mode timings, not score-only latencies. Do not present generated written feedback as SHAP itself or claim that a public deployment has been verified.

## Slide 11: Limitations and conclusion

Close with the report's contribution: a Khmer grading study across four families, a fine-tuned KhmerGrader and teacher-inspectable word highlights. Keep the source-score and provenance issues visible without substituting later experiments for the final report. A second grader and teacher evaluation are future work, not completed validation.
