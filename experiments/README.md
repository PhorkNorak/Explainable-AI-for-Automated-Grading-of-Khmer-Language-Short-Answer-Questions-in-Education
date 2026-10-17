# Experiments registry

The registry implements the four-family study described in the [final report](<../docs/reports/Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education Report final.pdf>). The LLM family is KhmerGrader. This registry documents the final-report study; subsequent release experiments are excluded. [Report-to-evidence notes](../docs/guides/report_evidence.md) explain why existing summaries and historical predictions are not a newly reproduced champion set.

Before any new training, resolve the source score of 19/15 and follow the [isolated HPC handoff](../docs/guides/hpc.md). The commands below describe the scripts; they are not a request to overwrite existing evidence.

The v01-v07 grid uses the shared schema from `scripts/run_all.py`; later studies have their own outputs:

```
train_*  (qwk, accuracy, adjacent_accuracy, mae,
          raw_exact, raw_within1, raw_mae, pct_mae)
test_*   (same eight)
val_qwk, best_epoch, seconds
```

## Versioning scheme

| Version | Script | Output dir per dataset | Effort | What it does |
|---|---|---|---|---|
| **v01** | `exp01_tfidf_baseline.py` | `local/results/results_<ds>/` | low | TF-IDF Cosine + SVR baseline (12 cells per dataset) |
| **v02** | `exp02_threshold_calibration.py` | `local/results/results_<ds>_v02_calibrated/` | very low | Re-calibrates the 4 bin boundaries on val to maximize val QWK, then applies to test. Only the 5-class metrics change; raw-score metrics are unchanged. |
| **v03** | `exp03_maxscore_feature.py` | `local/results/results_<ds>_v03_maxfeat/` | low | TF-IDF+SVR with `max_score` concatenated to the feature vector. SVR sees the per-question scoring scale explicitly. |
| **v03b** | `exp03b_maxfeat_neural.py` | `local/results/results_<ds>_v03b_maxfeat_neural/` | low/medium | Neural variant of v03; BiLSTM + transformers with `max_score` concatenated to the head input. BiLSTM CPU-feasible; transformer cells HPC only. |
| **v04** | `exp04_bucket_svr.py` | `local/results/results_<ds>_v04_bucket/` | medium | Trains a separate TF-IDF+SVR per `max_score` bucket. Routes test answers by max_score to the matching model. |
| **v05** | `exp05_bilstm.py` | `local/results/results_<ds>_v05_bilstm/` | low | Full BiLSTM+Attention grid (12 cells per dataset). Full grid scheduled on HPC; individual BiLSTM runs are CPU-feasible. |
| **v06** | `exp06_transformer.py` | `local/results/results_<ds>_v06_transformer/` | medium | Full transformer grid (36 cells per dataset). **HPC ONLY**; needs model access and GPU resources. |
| **v07** | `exp07_ensemble.py` | `local/results/results_<ds>_v07_ensemble/` | low | Reads the top-3 cells across v01-v06 by validation QWK, takes weighted mean of `pred_score`, recomputes metrics. |

## Dataset variants (in `experiments/_common.py::DATASETS`)

| run_name | dataset CSV | DROP_SCORE_ZERO | rows |
|---|---|---|---|
| `full`      | `dataset.csv`                 | False | 1,184 |
| `no10c`     | `dataset_no_10c_biology.csv`  | False |   909 |

The shared v01-v07 grid runs on **both** datasets by default (grade-0 kept; full 5-class). Use the
`--datasets` CLI flag on these grid scripts to target a subset.

```powershell
python experiments/exp01_tfidf_baseline.py                    # both
python experiments/exp01_tfidf_baseline.py --datasets full    # 1184 only
python experiments/exp01_tfidf_baseline.py --datasets no10c   # 909 only
```

## Conventions

- Grid experiments write a per-cell `runs/<run_id>/` directory with:
  - `config.json`         (what was trained)
  - `metrics.json`        ({train, val, test, best_epoch})
  - `predictions_train.csv`, `predictions_val.csv`, `predictions_test.csv`
- Grid experiments write a `leaderboard.csv` with the shared grid schema.
- Experiments do **not** modify each other's output directories.
- Re-running an experiment overwrites its own output cleanly (deletes leaderboard before re-running).

## How to run

The shared grid scripts iterate over **both datasets** sequentially by default, patching
`config.RAW_CSV`, `DROP_SCORE_ZERO`, and the output paths between datasets.

```powershell
# 1. Baseline (needed before anything else)
python experiments/exp01_tfidf_baseline.py

# 2-4. Classical enhancements (fast, CPU)
python experiments/exp02_threshold_calibration.py
python experiments/exp03_maxscore_feature.py
python experiments/exp04_bucket_svr.py

# 5-6. Neural (HPC only)
python experiments/exp05_bilstm.py
python experiments/exp06_transformer.py
# Neural variant of v03 (max_score as numeric input to BiLSTM + transformer heads):
python experiments/exp03b_maxfeat_neural.py
# Optionally calibrate any of the above too:
python experiments/exp02_threshold_calibration.py --source v05_bilstm
python experiments/exp02_threshold_calibration.py --source v06_transformer
python experiments/exp02_threshold_calibration.py --source v03b_maxfeat_neural

# 7. Ensemble (run AFTER any upstream you want included in the pool)
python experiments/exp07_ensemble.py

# Compare everything
python experiments/compare_all.py --topk 10
```

Within this grid sequence, `exp07_ensemble.py` depends on upstream prediction files; it reads the leaderboards of
v01-v06 to pick the top-3 cells, so run all upstream experiments first.

## LLMs and follow-up studies

| Script | Role | Execution and evidence |
|---|---|---|
| `exp08_llm_finetune.py` | KhmerGrader QLoRA adaptation and `--zeroshot` bases | GPU/HPC |
| `exp09_xai.py` | SHAP word attribution and reference-overlap plausibility | `--families`, `--dataset`, `--output-dir`, `--sample`, `--shap-max-evals`; heavier families on HPC |
| `exp10_significance.py` | Point metrics from configured champion predictions | CPU; requires a fresh `--output-dir`; no confidence intervals or significance tests |
| `exp11_cleaning_ablation.py` | Old versus refined format-noise cleaning for classical models | CPU; not a new KCC reorder |
| `exp12_hparam_tuning.py` | Classical C/max-features search selected by validation QWK | CPU |
| `exp14_frontier_baselines.py` | Bare/reasoning zero-shot external reference | Network/API experiment; no calls made by documentation work |

The point-metric script requires all configured prediction files, including its zero-shot entry, and refuses to overwrite an existing output CSV. Once compatible artifacts have been reviewed, use a new directory:

```bash
python experiments/exp10_significance.py --output-dir output/reviewed_champion_metrics
```

This invocation is not currently guaranteed to complete: the audit records missing expected champion artifacts. Do not substitute old grade-0-excluded runs. A matching summary CSV does not resolve that issue.

Report Table 14 matches the saved XAI plausibility summary after rounding, including the unequal sample counts. Tables 13 and 16 provide the frontier breakdown. These results are populated in the latest PDF; do not retain the older PDF's pending labels. Historical provenance issues remain documented in the [evidence reference](../docs/guides/report_evidence.md).
