# Khmer Automatic Short-Answer Grading Benchmark

Dataset version **1.0.1**, prepared **17 October 2026**. Developed by **Phork Norak**, supervised by **Dr. Khim Chamroeun**, Royal University of Phnom Penh.

## Project resources

| Resource | Link |
|---|---|
| Code, setup and project README | [GitHub](https://github.com/PhorkNorak/Explainable-AI-for-Automated-Grading-of-Khmer-Language-Short-Answer-Questions-in-Education) |
| Dataset and dataset viewer | [Hugging Face](https://huggingface.co/datasets/phorknorak/khmer-asag-benchmark) |
| Dataset mirror | [Kaggle](https://www.kaggle.com/datasets/phorknorak/khmer-asag-benchmark) |
| Report publication page | [ResearchGate](https://www.researchgate.net/publication/416039997_Explainable_AI_for_Automated_Grading_of_Khmer_Language_Short-Answer_Questions_in_Education) |
| Final report and presentation downloads | [GitHub release v1.1.0](https://github.com/PhorkNorak/Explainable-AI-for-Automated-Grading-of-Khmer-Language-Short-Answer-Questions-in-Education/releases/tag/v1.1.0) |

The dataset version and software release version are separate. Check each dataset host's revision or version history when citing or downloading data.

## Real examination data

The corpus contains real Khmer short answers manually transcribed from handwritten semester 1 examination papers from two Cambodian high schools, one in a province and one in the capital city. It contains no generated or crowd-sourced answers. Scores were assigned by one teacher.

An official answer key was unavailable. The highest-scoring student response for each question serves as the `Reference`. It is a student-derived proxy and may be incomplete; it is not a teacher-written answer key or a verified marking rubric.

| Property | Full corpus | Curated corpus |
|---|---:|---:|
| Complete answers | 1,184 | 909 |
| Questions | 41 | 36 |
| Students | 203 | 203 |
| Schools | 2 | 2 |
| Classes | 8 | 8 |
| Biology answers | 435 | 160 |
| History answers | 433 | 433 |
| Geography answers | 307 | 307 |
| Earth Science answers | 9 | 9 |

The collection year is 2026. Recorded maximum marks are 5, 6, 7, 8, 10, 12, 15 and 20.

## Files and configurations

| File | Hugging Face configuration | Purpose |
|---|---|---|
| `dataset.csv` | `full` | All 1,184 complete answers |
| `dataset_no_10c_biology.csv` | `curated-no10c` | 909 answers after excluding the five-question class-10C Biology subset with inconsistent grading |

The curated file is a subset of the full file. Do not concatenate them as independent records. Both retain grade-0 answers. Each public file excludes the same one blank-answer source record. Original local source files contain 1,185 and 910 rows, respectively.

Public exports replace school identifiers with `SCH001` and `SCH002`, preserve pseudonymous student and class codes, normalise surrounding whitespace and column headers, and omit the blank-answer record. No private identifier mapping is distributed.

## Data fields

| Column | Meaning |
|---|---|
| `SchoolID` | Replacement school code, `SCH001` or `SCH002` |
| `ClassID` | Pseudonymous class code |
| `Subject` | Biology, History, Geography or Earth Science |
| `StudentID` | Pseudonymous student code |
| `QuestionID` | Question identifier |
| `Question` | Transcribed Khmer examination question |
| `Reference` | Highest-scoring student response used as the reference |
| `Answer` | Transcribed student answer |
| `Student Score` | Teacher-assigned raw mark, including the documented correction below |
| `Max Score` | Recorded maximum mark for the question |
| `Year` | Collection year |

CSV files use UTF-8 with a byte-order mark. Question, reference and answer text are retained in Khmer; model preprocessing is applied separately by the code.

## Version 1.0.1 correction

The author's edited local data changes one `Student Score` from **19 to 9**, with `Max Score` remaining **15**. The same answer occurs in both configurations. Comparing the new public exports with the archived 1.0.0 exports finds exactly one changed cell in each CSV and no added, removed or reordered public rows.

All scores in the corrected exports are numeric and satisfy `0 <= Student Score <= Max Score`, with positive maxima. No question, reference, answer or identifier value changed relative to those archived public exports.

The final report, saved predictions, checkpoints and historical audit records are preserved. Its benchmark metrics were **not rerun on version 1.0.1**. A label-stratified split can change after a corrected score changes the derived label, even with the same seed. Do not describe a new run on this version as an exact reproduction of the report without checking dataset hashes and split membership.

## Download and use

To use the published CSVs with the GitHub project, download both files from either dataset host and place them at:

```text
local/data/dataset.csv
local/data/dataset_no_10c_biology.csv
```

These local copies remain ignored by Git. Do not overwrite an existing research copy without preserving it and its hashes.

Using Hugging Face's `datasets` package:

```python
from datasets import load_dataset

corpus = load_dataset(
    "phorknorak/khmer-asag-benchmark",
    "curated-no10c",
    split="train",
)
```

The hosted split named `train` contains the **entire selected configuration**; it is a loading convention, not the report's training partition. The project code creates its label-stratified 70/15/15 answer-level split with seed 42.

For a local CSV:

```python
import pandas as pd

df = pd.read_csv("local/data/dataset_no_10c_biology.csv", encoding="utf-8-sig")
df["normalized_score"] = df["Student Score"] / df["Max Score"]
df["score_label"] = (4 * df["normalized_score"]).round().clip(0, 4).astype(int)
```

Quadratic Weighted Kappa (QWK) is the primary report metric. Accuracy, macro-F1, exact raw-mark agreement and within-one-point agreement support it. The report's Classical, BiLSTM and LLM rows use a 137-answer no10c test set; the encoder uses a different, 178-answer full test set. Frontier rows also have different parsed sample counts.

## Limitations and responsible use

Labels measure agreement with one teacher, not independent correctness or inter-grader agreement. The corpus covers two schools and four subjects, contains uneven score distributions and only nine Earth Science answers, and uses student-derived references. Identical answers can have different teacher marks. The reported evaluation is answer-level, so a student can occur in multiple partitions.

School, class and student codes are pseudonymous, not proof of complete anonymity. Do not attempt re-identification or contact participants, and avoid redistributing examples that reveal personal details. Use human review for educational decisions; this corpus does not establish the safety or fairness of autonomous high-stakes grading. Public availability does not independently establish consent or ethics approval.

## Licence and citation

The dataset retains its **CC BY-NC 4.0** licence: attribution is required and commercial use is not permitted. See the [official licence](https://creativecommons.org/licenses/by-nc/4.0/). The project's MIT licence applies to its software, not the student corpus. No licence change is made in this update.

```bibtex
@dataset{phork_khmer_asag_2026,
  author  = {Phork, Norak},
  title   = {Khmer Automatic Short-Answer Grading Benchmark},
  year    = {2026},
  version = {1.0.1},
  url     = {https://huggingface.co/datasets/phorknorak/khmer-asag-benchmark},
  note    = {Undergraduate research supervised by Dr. Khim Chamroeun; also available on Kaggle}
}
```

Please cite the dataset version and the report when using this work. Use the linked ResearchGate page for the report record and the GitHub release for its final PDF. Report errors and suggested corrections through [GitHub issues](https://github.com/PhorkNorak/Explainable-AI-for-Automated-Grading-of-Khmer-Language-Short-Answer-Questions-in-Education/issues).
