# Ethics and data access

This project uses real Khmer examination answers from two Cambodian high schools, written by 203 students and graded by one teacher. Public dataset copies are hosted on [Hugging Face](https://huggingface.co/datasets/phorknorak/khmer-asag-benchmark) and [Kaggle](https://www.kaggle.com/datasets/phorknorak/khmer-asag-benchmark). See the [dataset card](dataset.md) for the schema, correction history, licence and links to the report and code.

## Public copies and private working records

Public exports contain replacement school codes (`SCH001` and `SCH002`), pseudonymous class and student codes, question identifiers, Khmer question/reference/answer text, marks and collection year. They exclude one blank-answer record and do not include an identifier mapping. The reference is the highest-scoring student response for each question because an official answer key was unavailable.

Original local records, predictions, checkpoints, manuscript sources and historical audit files remain excluded from Git. Downloaded CSVs also stay under ignored `local/data/`; public distribution through the dataset hosts does not make the whole working directory public.

Pseudonymous codes and public hosting do not establish complete anonymity. Free-text answers may contain context that an automated check cannot detect. Do not attempt to identify or contact students, teachers or schools, or republish examples that disclose personal details.

## Consent and governance evidence

Public hosting was verified, but supporting consent records, school authorisation, student/guardian permission and an ethics approval or exemption have not been independently verified in this repository review. Their status must not be inferred from the report's release wording or the dataset's licence. Preserve these evidence limits when describing the corpus.

## Dataset version and research claims

The author's current local CSVs correct one mark from 19/15 to 9/15, recorded in dataset version 1.0.1. Public counts remain 1,184 full and 909 curated answers. The final report and saved benchmark results have not been rerun on the corrected version. Historical audits remain unchanged; see the [report-to-evidence notes](report_evidence.md).

All labels come from a single grader and measure agreement with that grader, not independent correctness or inter-grader agreement. Data covers only two schools and four subjects, is unevenly distributed, and uses student-derived reference answers. No completed classroom field study establishes educational benefit or fairness.

## Intended use and licensing

The dataset retains its [CC BY-NC 4.0 licence](https://creativecommons.org/licenses/by-nc/4.0/), requiring attribution and excluding commercial use. The source code has a separate MIT licence. Neither licence proves consent or replaces privacy and institutional responsibilities.

Use the system for research and teacher assistance with human review. It is not established as an autonomous high-stakes grader. SHAP word attribution shows model contributions to a score; lexical reference overlap is not proof that a response or explanation is educationally correct. Prototype feedback uses an open-source LLM with rule-based fallback.

## Compute evidence

Runtime figures are report-stated or derived from saved summaries as described in the evidence notes. No new training or benchmark API evaluation was performed for this dataset and documentation update.
