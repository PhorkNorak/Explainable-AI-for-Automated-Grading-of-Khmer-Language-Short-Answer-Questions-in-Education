# Ethics & Data-Governance Statement

This project uses **real student answers** collected in Cambodian schools, so it carries
human-subjects and data-protection obligations. This statement records what has been done and
**what the author must confirm before journal submission** (items marked ⚠️).

## Data collection & consent
- The corpus contains short-answer responses written by school students (grades/classes across
  2 schools, 203 students) and graded by one teacher.
- ⚠️ **Confirm and state**: that informed consent (from the school and from students/guardians,
  as minors are involved) was obtained for research use and publication of anonymized data, and
  that data collection followed the schools' policies.
- ⚠️ **IRB / ethics approval**: state the approving body and protocol number, or, if the work
  qualified for exemption, the basis for exemption.

## Anonymization & privacy
- The released CSVs contain `SchoolID`, `ClassID`, `StudentID`, `QuestionID` — these are
  **pseudonymous codes**, not names. No student names, contact details, or free-text personal
  identifiers should be present.
- ⚠️ **Verify before release**: that no answer text contains personally identifying information
  (names, IDs) and that the school/student codes cannot be re-identified outside the research
  team. If any PII exists in answer text, scrub it.
- Only aggregate metrics and a small number of de-identified example answers (heatmaps) are
  shown in the thesis/slides.

## Fairness, bias & intended use
- The dataset is **single-grader**: all labels reflect one teacher's judgement. Reported scores
  are agreement with *that* grader, not ground-truth correctness. This is disclosed as a primary
  limitation (no inter-annotator agreement was obtainable).
- The system is intended as a **teacher-assist / formative-feedback** tool, **not** an
  autonomous high-stakes grader. Any deployment should keep a human in the loop and surface the
  model's explanation (and its **faithfulness caveats** — see the XAI results).
- Subject/topic coverage is narrow (Biology, History, Geography, Earth Science; grades from 2
  schools), so the model should not be assumed to generalize to other subjects, dialects, or
  grade levels without re-validation.

## Reproducibility & data availability
- Code, configuration, seeds, leaderboards, and the curated dataset variants are in this
  repository. ⚠️ Confirm the data licence/usage terms permit public release before publishing
  the CSVs; otherwise release a synthetic or access-controlled sample.

## Environmental note
- Compute was modest (~40 GPU-hours total for the full grid; the headline classical model
  trains in ~30 s on CPU). The cheapest, most accountable model is also competitive.
