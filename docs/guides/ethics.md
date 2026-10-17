# Ethics & Data-Governance Statement

This project uses **real student answers** collected in Cambodian schools, so it carries
human-subjects and data-protection obligations. This statement records what has been done and
**what the author must confirm before any public data release** (items marked ⚠️).

The [designated final report](<../reports/Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education Report final.pdf>) is dated June 2026. Its public-release wording does not establish data availability or consent. The [report-to-evidence reference](report_evidence.md) records this difference; the restrictions below remain in force.

## Data collection & consent
- The corpus contains short-answer responses written by school students (grades/classes across
  2 schools, 203 students) and graded by one teacher.
- ⚠️ **Confirm and state**: that informed consent (from the school and from students/guardians,
  as minors are involved) was obtained for research use and publication of anonymized data, and
  that data collection followed the schools' policies.
- ⚠️ **IRB / ethics approval**: state the approving body and protocol number, or, if the work
  qualified for exemption, the basis for exemption.

## Anonymization & privacy
- The local research CSVs contain `SchoolID`, `ClassID`, `StudentID`, and `QuestionID`. These are
  **pseudonymous codes**, not names. No student names, contact details, or free-text personal
  identifiers should be present.
- ⚠️ **Verify before release**: that no answer text contains personally identifying information
  (names, IDs) and that the school/student codes cannot be re-identified outside the research
  team. If any PII exists in answer text, scrub it.
- Only aggregate metrics and a small number of de-identified example answers (heatmaps) are
  shown in the undergraduate report and slides.

## Fairness, bias & intended use
- The dataset is **single-grader**: all labels reflect one teacher's judgement. Reported scores
  are agreement with *that* grader, not ground-truth correctness. This is disclosed as a primary
  limitation (no inter-annotator agreement was obtainable).
- The system is intended as a **teacher-assist / formative-feedback** tool, **not** an
  autonomous high-stakes grader. Any deployment should keep a human in the loop and surface the
  model's explanation (the **SHAP word attribution**; see the XAI results).
- Subject/topic coverage is narrow (Biology, History, Geography, Earth Science; grades from 2
  schools), so the model should not be assumed to generalize to other subjects, dialects, or
  grade levels without re-validation.

## Reproducibility & data availability
- Code, configuration, seeds, and experiment scripts are included in this repository. The student
  CSVs, saved predictions, and generated results are excluded from Git.
- ⚠️ Confirm that the data licence and consent terms permit public release before publishing any
  corpus files. Otherwise, provide a synthetic sample or an access-controlled request process.

## Compute evidence

Total compute and model-specific runtime claims remain UNVERIFIED; no full training was performed during the audit.
