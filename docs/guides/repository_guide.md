# Repository layout and GitHub policy

## Public folders

| Folder | Contents |
|---|---|
| `src/khmer_asag/` | Importable Python package: configuration, data, preprocessing, evaluation, training, models and SHAP |
| `experiments/` | Experiment definitions and their helper modules |
| `prototype/` | Gradio application, requirements and setup guide |
| `scripts/` | Pipeline runners, analysis tools and the isolated HPC helper |
| `tests/` | Focused automated checks, with no training or API calls |
| `docs/` | Guides, reviewed README images, final report and presentation PDFs |

The root keeps the README, requirements, `pyproject.toml`, citation, licence, project instructions and ignore rules. Install dependencies, then run `python -m pip install -e .` from the root. The core package is imported as `khmer_asag`.

## Private local folders

The entire `local/` folder is ignored by Git:

- `data/`: student CSVs and approved private data.
- `results/`: all existing `results_*` and `xai_visuals_*` folders, with their names and contents preserved.
- `checkpoints/results/`: the existing champion checkpoints and associated leaderboards.
- `logs/`: run logs and isolated-run receipts.
- `builds/`: exports, generated files, previews, temporary work and local release builds.
- `manuscripts/`: the final editable report and historical thesis project; no LaTeX source is public.
- `archive/`: old documents, historical audits, reference papers, local metadata and the migration manifest.
- `environments/` and `cache/`: local Python environments and model download caches.

Source and artifact locations are separated in `src/khmer_asag/paths.py`. It resolves older relative run names to their new private locations without rewriting saved CSVs, JSON records or checkpoint bytes. Paths recorded inside historical evidence may still name their old locations; use the migration manifest under `local/archive/restructure/` to trace them. Local historical helper scripts were archived intact and may require path updates before reuse.

Only the final report PDF, presentation PDF and document README are public under [reports](../reports/README.md). All LaTeX/bibliography files are ignored everywhere, as well as data, weights, caches, secrets and generated outputs. A public checkout can read the PDFs but cannot rebuild the private manuscript.

The local Kaggle inspection metadata was previously removed from tracking and remains on disk in the local archive. That removal is staged. Other reorganization changes are left for the author's review and staging. No commit or push was created.

## Review commands

```bash
git status --short
git diff --check
git diff --stat
git ls-files --others --exclude-standard
git ls-files -ci --exclude-standard
python -m unittest discover -s tests
```

Review renames, new paths and staged changes before a manual commit. Do not force-add the local archive or manuscript files. Ignoring a file does not remove it from earlier Git history.
