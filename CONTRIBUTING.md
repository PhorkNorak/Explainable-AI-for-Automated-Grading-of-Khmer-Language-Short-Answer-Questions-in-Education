# Contributing

Thank you for helping improve the Khmer ASAG project. Useful contributions include reproducibility findings, dataset corrections, documentation improvements and fixes to Khmer text handling.

## Before opening an issue

Read the [README](README.md), [dataset card](docs/guides/dataset.md) and [report-to-evidence notes](docs/guides/report_evidence.md). The final report records historical results; dataset v1.0.1 includes a later score correction. Different data versions, splits or checkpoints can produce different metrics.

Use the issue form that matches your finding:

- **Bug or reproducibility issue:** include the command, observed behaviour, expected behaviour, software revision and relevant environment details.
- **Dataset correction:** include the dataset host and version, CSV filename and row reference, plus a description of the possible error. A local row number alone is not enough without a version.

Do not post access tokens, original school records, identifier mappings, personal details, private predictions or model checkpoints. Use a short, non-identifying example when one is needed. A public dataset does not make all source records public.

## Local development

Create an isolated environment, install the dependencies for your task, then install the package from the repository root:

```bash
python -m pip install -e .
```

Use the [prototype guide](prototype/README.md) for its separate environment. Encoder and LLM training belong on a prepared HPC/GPU system; follow the [HPC guide](docs/guides/hpc.md) rather than starting a full run for a small fix.

For code changes, run the relevant checks. The existing unit suite can be run with its dependencies installed:

```bash
python -m unittest discover -s tests
git diff --check
```

For documentation changes, check relative links, image paths and commands. Trace numerical claims to the final report or a named saved result; keep report-stated results separate from newly reproduced results. Do not add unsupported claims about model quality or classroom effectiveness.

## Pull requests

Keep the change focused and explain the problem, the resulting behaviour and how it was checked. Include data/model revisions when they affect the result. Add a meaningful test for behaviour changes where appropriate; a wording change does not need a new test.

Keep source CSVs, predictions, weights, environments, generated outputs and all editable LaTeX/bibliography files out of Git. Do not rewrite historical audits or replace result files to make them match the report. Store new research outputs separately with their provenance.

Preserve the project's SHAP word-attribution framing and readable Khmer tokens. Prototype feedback must use an open-source LLM with rule-based fallback. Keep the source-code MIT licence separate from the dataset's CC BY-NC 4.0 terms and upstream model licences.
