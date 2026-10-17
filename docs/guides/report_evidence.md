# Latest final report and saved evidence

The [approved final report](<../reports/Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education.pdf>) is the 67-page version with the author's review corrections and clearer wording. It retains the June 2026 report date and benchmark values. It supersedes the earlier report PDFs as the active documentation reference. No pending-result labels occur in this final version.

The comparison with saved results was made locally. The CSV paths and detailed audit records named below are local provenance references and are excluded from Git; they are not downloads included in this repository. No new training, external API verification or result-file editing was performed.

## Authority and versions

Use the final PDF for report-facing names and findings. Later experiments are outside this documentation scope. Its editable source was recovered and revised, rather than located as an exact original Overleaf export. All manuscript sources remain local and are excluded from Git. The local `local/manuscripts/thesis/` tree is a separate historical audit revision. See [report versions](../reports/README.md).

The cover states Bachelor of Engineering in Data Science and Engineering, June 2026. The supervision statement names the Honor Program of Data Science and Engineering. The committee lists Dr. Khim Chamroeun (Chairperson), Ms. Chea Daly, Mr. Ky Soklay, Mr. Chap Chanpiseth and Mr. Mao Makara. These are report-transcribed details.

## Headline results

Report Table 10 (PDF page 42, printed page 30) gives QWK **0.802 / 0.789 / 0.777 / 0.850** for Classical / BiLSTM / GTE / Qwen-KhmerGrader, matching the local `local/results/results_stats/champion_metrics.csv`. Its test sizes are 137 / 137 / 178 / 137. The encoder uses the full dataset variant, so cross-pillar comparisons are indicative.

The latest table gives exact raw-mark matches **0.28 / 0.34 / 0.49 / 0.67**. Its caption attributes exact matches to per-configuration leaderboards; the champion summary itself has no exact-match column. Do not carry forward the earlier PDF's Qwen exact value of 0.693. Latest report headline: **67% exact, 80% within one point**. The summary's Qwen within-one value is 0.796, which rounds to 0.80.

Table 12 (PDF page 48, printed page 36) gives fine-tuned family QWK: Qwen **0.850**, Gemma **0.844**, SEA-LION **0.713**. Table 10 gives the Qwen zero-shot reference **0.529**. The latest family table no longer includes the older PDF's family exact-match or pending baseline columns; do not reinstate them.

The historical prediction audit (`local/archive/audit/data_metrics_audit.md`, local) recomputed 628 saved test files. Its named configurations yielded QWK 0.782 / 0.666 / 0.820 / 0.843, not the report's coherent champion set. The audit records that the claimed encoder and LLM headline values were not reproduced from the corresponding continuous predictions using uncalibrated rounding; stored-label and calibrated metrics are distinct conditions. Expected no10c Classical/RNN champion artifacts and no-KCC training provenance remain unresolved. Preserve these findings, and do not relabel the named audit configurations as newly selected champions.

## SHAP plausibility

Report Table 14 (PDF page 52, printed page 40) agrees with the saved XAI summary (`local/results/results_xai/no10c/faithfulness_leaderboard.csv`, local) after rounding:

| Family | Latest report | Saved plausibility | Answers explained |
|---|---:|---:|---:|
| Classical | 0.63 | 0.6285 | 66 |
| BiLSTM | 0.60 | 0.5985 | 66 |
| Encoder | 0.68 | 0.6842 | 18 |
| LLM | 0.61 | 0.6053 | 18 |

All four values are populated. Encoder has the highest reported value, but unequal samples limit the ranking. The filename is historical; the relevant measure is SHAP reference-overlap plausibility. The historical XAI audit (`local/archive/audit/xai_audit.md`, local) records estimator and checkpoint provenance concerns. Matching a saved summary confirms transcription, not independent reproduction or a human evaluation of educational value.

## Efficiency

Table 11 (PDF page 48, printed page 36) reports:

| Pillar | Training time | Per-answer score + SHAP |
|---|---|---:|
| Classical | approximately 30 s, CPU | approximately 2 s |
| BiLSTM | approximately 150 s, CPU | approximately 13 s |
| Encoder | approximately 140 s, GPU | approximately 10 s |
| LLM | approximately 3.8 h, GPU | approximately 76 s |

The XAI summary's total seconds divided by explained-answer count are approximately 2.12 / 12.88 / 10.23 / 76.37, numerically consistent with the rounded explanation timings. This arithmetic does not establish a separately instrumented per-request latency benchmark. Training times are report-stated; the current saved runs are not confirmed as the exact report runs. The latest table does not contain score-only latency columns.

## Frontier comparison

Report Tables 13 and 16 (PDF pages 50 and 65) contain both bare and reasoning results, including exact matches and unparsed counts. The report states an evaluation date of **2026-06-24**. Its frontier QWK values match the local `local/results/results_frontier/no10c/frontier_metrics.csv`:

| Saved model version | Setting | Parsed n | Unparsed n | QWK |
|---|---|---:|---:|---:|
| openai/gpt-5.5-20260423 | bare | 126 | 11 | 0.684 |
| openai/gpt-5.5-20260423 | reasoning | 136 | 1 | 0.708 |
| anthropic/claude-opus-4.8 | bare | 137 | 0 | 0.662 |
| anthropic/claude-opus-4.8 | reasoning | 137 | 0 | 0.734 |
| google/gemini-3.5-flash-20260519 | bare | 72 | 65 | 0.555 |
| google/gemini-3.5-flash-20260519 | reasoning | 117 | 20 | 0.585 |
| deepseek/deepseek-v4-flash-20260423 | bare | 134 | 3 | 0.553 |
| deepseek/deepseek-v4-flash-20260423 | reasoning | 136 | 1 | 0.579 |

Unparsed responses are excluded, so denominators differ. The best reported frontier QWK is 0.734 versus fine-tuned Qwen's 0.850. The CSV has no evaluation-date column; the date above is report-stated. No live API verification or new evaluation was performed. This benchmark does not authorise proprietary models for prototype feedback.

## Remaining evidence and release boundaries

- The historical audit recorded a score of 19 with maximum 15. The author's current local CSVs change that score to 9 in both dataset variants. Public dataset version 1.0.1 records exactly one changed cell per CSV relative to the archived public exports; counts and row order are unchanged. See the [dataset correction record](dataset.md). This supersedes the current-data blocker, not the dated audit record. No report metrics, predictions or checkpoints were regenerated; a fresh stratified split may differ after the correction.
- Public pseudonymous dataset copies are available through [Hugging Face](https://huggingface.co/datasets/phorknorak/khmer-asag-benchmark) and [Kaggle](https://www.kaggle.com/datasets/phorknorak/khmer-asag-benchmark), separately from Git. The dataset has a CC BY-NC 4.0 licence. Original local records and predictions remain excluded. Public availability does not establish consent or ethics approval; see [ethics](ethics.md).
- The final report includes the prototype screenshot from the final presentation in Appendix G. It does not establish a public demo URL or completed classroom field study.
- Programme, date and committee names are printed, but signatures, consent records and Khmer proofreading are not independently verified here.
- The supplied presentation PDF is copied unchanged into `docs/reports/Final_Presentation.pdf`. Active Markdown decks and notes describe the final report; they are not a regenerated binary slide export.

Historical files under `local/archive/audit/`, including archived writing, remain unchanged. Their descriptions of prior report versions should be read in their dated context.
