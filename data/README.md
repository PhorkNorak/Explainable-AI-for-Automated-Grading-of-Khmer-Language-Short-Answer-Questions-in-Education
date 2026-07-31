# Dataset setup

The dataset used by this project contains real answers written by school students. It is intentionally
excluded from Git until its public release and reuse terms have been confirmed. Do not commit or share
the CSV files without the required consent, anonymization review, and data-owner approval.

## Expected files

Place approved local copies at:

```text
data/dataset.csv
data/dataset_no_10c_biology.csv
```

`dataset.csv` is the complete corpus. `dataset_no_10c_biology.csv` is the curated experimental corpus
with the inconsistently graded 10C Biology subset removed.

## Required columns

| Column | Description |
|---|---|
| `SchoolID` | Pseudonymous school code |
| `ClassID` | Pseudonymous class code |
| `Subject` | Subject name |
| `StudentID` | Pseudonymous student code |
| `QuestionID` | Question identifier |
| `Question` | Question text |
| `Reference` | Teacher reference answer |
| `Answer` | Student answer |
| `Student Score` | Score assigned by the teacher |
| `Max Score` | Maximum possible score for the question |
| `Year` | Collection year |

Leading and trailing spaces in column names are removed by the loader. UTF-8 with an optional byte-order
mark is supported.

## Privacy checklist

Before using or releasing a dataset:

1. Confirm permission for research use and publication.
2. Confirm the applicable ethics approval or exemption.
3. Remove names, contact information, and other identifying text from answers.
4. Confirm that school, class, and student codes cannot be linked back to individuals by recipients.
5. Document the dataset licence separately from the repository's MIT software licence.

For more information, see [`../docs/ethics.md`](../docs/ethics.md).
