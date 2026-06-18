# Khmer ASAG — A Reproducible Multi-Pillar Benchmark

Automatic Short Answer Grading (ASAG) for **Khmer**: grading a student's free-text answer
against a reference answer on a numeric scale. This repository accompanies the thesis and
contains the dataset, the full experiment pipeline, the result leaderboards, and the defense
slides.

To our knowledge this is the **first reproducible Khmer ASAG benchmark**. It compares four
model families ("pillars") under one consistent recipe:

| Pillar | Representative model | Headline test QWK (uncalibrated) | Deployment |
|---|---|---|---|
| **Classical** | TF-IDF + RBF-SVR | QWK 0.795 (895) | exact 0.20 |
| **RNN** | BiLSTM + Attention (char-level) | **QWK 0.845** (895) | exact 0.54 |
| **Transformer** | GTE-multilingual dual-encoder (+ max-score feature) | QWK 0.820 (1184) | raw-exact 0.573 |
| **LLM** | Qwen 3.5 4B, QLoRA fine-tune (KhmerGrader family) | QWK 0.843 (909) | **66% exact**, **83% within ±1 pt** |

All four QWKs fall in a narrow 0.05 band (0.795–0.845; see `results_stats/champion_metrics.csv`),
so **no pillar is a clear research winner on QWK**; the LLM's lead on the deployment/accuracy metrics
is the one clear ranking.

**Key finding — no free lunch:** the four pillars are comparable on the research metric
(QWK), while a fine-tuned LLM wins every classroom-deployment metric (exact-score match, within
±1 point). Data cleaning moves results **as much as or more than swapping architectures** on this
small corpus.

> **Caveat (read before citing numbers):** headline QWKs are **uncalibrated**, reported consistently
> across pillars. Threshold calibration is a *validation-selected, fragile, model-dependent ablation*
> — it lifts the classical model (test 0.795→0.847) but **lowers** the BiLSTM on test, so it is not
> folded into the headline. Pillars are also evaluated on different dataset variants (895/909/1184),
> so cross-pillar QWK differences are indicative rather than head-to-head (only classical vs BiLSTM
> share a test set).

---

## Dataset

`data/dataset.csv` — **1,184** graded answers (after dropping 1 incomplete row), from
2 schools, 8 classes, 203 students, **41 unique questions** across 4 subjects (Biology,
History, Geography, Earth Science). Each question has its own max score ∈ {5,6,7,8,10,12,15,20}.
A single trained teacher graded every answer.

Ordinal label = `round(4 · StudentScore / MaxScore) ∈ {0..4}` (heavy skew: ~42% full credit).

**Three curated variants** (selected via `DROP_SCORE_ZERO` and the CSV in `config.py`):

| Variant | Rows | Definition |
|---|---:|---|
| `full` | 1,184 | original |
| `no10c` | 909 | drop the noisy "10C Biology" subset (`data/dataset_no_10c_biology.csv`) |
| `no10c_no0` | 895 | also drop the 14 `score=0` outliers (`DROP_SCORE_ZERO=True`) |

---

## Repository structure

```
final_kxs/
├── config.py          single source of truth: paths, model registry, hyperparameters
├── data.py            CSV load, 70/15/15 stratified split (seed 42), dataset classes
├── preprocess.py      raw / clean (strip invisibles+KCC+punct) / segment (khmernltk); digits kept
├── train.py           train_classical · train_bilstm · train_transformer
├── evaluate.py        QWK · accuracy · adjacent-acc · MAE + raw (deployment) metrics
├── run_all.py         grid orchestrator (legacy single-dataset entry point)
├── analyze_run.py     per-run error inspection
├── xai.py             gradient × input saliency for the top transformer cell
├── models/            classical.py · bilstm.py · dual.py · cross.py
├── xai/               explainable-AI toolkit shared across all four families
│   ├── explainers.py  LOO occlusion — the sole unified word-attribution method (all 4 pillars)
│   ├── attributions.py popular attribution methods: occlusion · LIME · SHAP (one dispatcher)
│   ├── faithfulness.py ERASER comprehensiveness & sufficiency + random baseline
│   ├── plausibility.py reference-overlap plausibility proxy (answer↔reference word overlap)
│   ├── render.py      Khmer word-importance heatmaps (PNG) + rationale cards
│   └── render_html.py browser-shaped Khmer heatmaps (PNG can't shape Khmer)
├── prototype/         live teacher-facing web app (Gradio) — see prototype/README.md
│   ├── app.py         score + word-attribution heatmap + AI/rule-based feedback; 4-pillar selector
│   └── requirements.txt
├── experiments/       exp01–exp09 (one enhancement each) + orchestration
│   ├── exp01_tfidf_baseline.py          v01  TF-IDF cosine + SVR
│   ├── exp02_threshold_calibration.py   v02  post-hoc cut-point calibration
│   ├── exp03_maxscore_feature.py        v03  max-score feature → SVR
│   ├── exp03b_maxfeat_neural.py         v03b max-score feature → neural heads
│   ├── exp04_bucket_svr.py              v04  per-max-score bucket SVR
│   ├── exp05_bilstm.py                  v05  BiLSTM + Attention grid
│   ├── exp06_transformer.py             v06  Dual/Cross × mBERT/XLM-R/GTE
│   ├── exp07_ensemble.py                v07  weighted top-3 by val QWK
│   ├── exp08_llm_finetune.py            v08  Qwen 3.5 4B QLoRA fine-tune
│   ├── exp09_xai.py                     XAI: faithfulness + plausibility across families
│   ├── check_progress.py                resumable orchestrator + status table
│   ├── compare_all.py                   cross-experiment ranking
│   └── plot_history.py                  per-cell training curves
├── data/              dataset.csv · dataset_no_10c_biology.csv
├── results/
│   ├── leaderboards/  one CSV per (dataset × version), 24-column schema
│   └── champions/     full run dir (config + metrics + predictions) for each cited cell
└── docs/
    ├── slides.md          thesis-defense deck (Marp)
    ├── script.md          speaker notes
    ├── references.md      verified citation list
    └── reference_papers/  Arabic ASAG anchor (Alaoui et al. 2024)
```

The leaderboard schema (24 columns) reports **train + test** for 8 metrics per cell:
`qwk, accuracy, adjacent_accuracy, mae, raw_exact, raw_within1, raw_mae, pct_mae`, plus
`val_qwk, best_epoch, seconds`.

---

## Reproducing the results

Install dependencies (Python 3.10+; a GPU is needed only for v05/v06/v03b/v08):

```bash
pip install -r requirements.txt
```

Each experiment runs over **all three datasets** by default and writes its own
`results_<dataset>_<version>/` directory. Order only matters for the ensemble (v07), which
reads the upstream leaderboards.

```bash
# Classical (fast, CPU)
python experiments/exp01_tfidf_baseline.py
python experiments/exp02_threshold_calibration.py
python experiments/exp03_maxscore_feature.py
python experiments/exp04_bucket_svr.py

# Neural / LLM (GPU)
python experiments/exp05_bilstm.py
python experiments/exp06_transformer.py
python experiments/exp03b_maxfeat_neural.py
python experiments/exp08_llm_finetune.py

# Ensemble + comparison
python experiments/exp07_ensemble.py
python experiments/compare_all.py --topk 20

# Resumable status / what's left to run
python experiments/check_progress.py
```

Run a single cell for a smoke test:

```bash
python experiments/exp01_tfidf_baseline.py --datasets full
```

> **Note on GTE.** `Alibaba-NLP/gte-multilingual-base` ships a NaN-corrupted RoPE cache under
> fp16; `models/dual.py` rebuilds the rotary cache in fp32 on load. If you see NaN loss from
> step 1 or a CUDA index assert in RoPE, that patch has regressed.

The pre-computed leaderboards for every cell are in `results/leaderboards/`, and the full
prediction files for the thesis-cited champion cells are in `results/champions/`.

---

## Running the full pipeline on an HPC (step by step, with tmux)

The classical/RNN baselines run on CPU; the Transformer encoders (v06) and the LLM (v08) need a
CUDA GPU. Run the whole thing inside **tmux** so it survives an SSH disconnect. Budget ~40 GPU-hours
for the full grid + LLM on one A40-class GPU.

**Step 1 — get the code onto the HPC and open a tmux session.**

```bash
cd final_kxs                 # the project folder
tmux new -s kxs              # detach with Ctrl-b then d; reattach with: tmux attach -t kxs
```

**Step 2 — create a virtual environment and install dependencies.**
Install the CUDA build of PyTorch *first* (match your driver — check `nvidia-smi`; `cu124` is the
most compatible, use `cu121`/`cu118` if the driver is older), then the rest.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU build of torch
pip install -r requirements.txt
# extras needed by the encoders (v06) and the LLM (v08) — not in requirements.txt:
pip install accelerate sentencepiece einops "datasets>=2.18" peft bitsandbytes
pip install unsloth          # LLM QLoRA (v08); if its build fails, exp08 falls back to peft+bnb
```

**Step 3 — sanity-check GPU + Khmer segmenter (≈10 s).**

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import khmernltk; print('khmernltk OK')"
export HF_HOME=$PWD/.hfcache   # keep downloaded models (mBERT/XLM-R/GTE/Qwen) inside the project
export TOKENIZERS_PARALLELISM=false
```

**Step 4 — run the whole pipeline (one resumable, logged command).**
Training scripts take `--resume`, so a crash/preemption is recovered by re-pasting the same block.

```bash
mkdir -p logs && ( \
  python -u experiments/exp01_tfidf_baseline.py   --resume && \
  python -u experiments/exp03_maxscore_feature.py --resume && \
  python -u experiments/exp04_bucket_svr.py       --resume && \
  python -u experiments/exp05_bilstm.py           --resume && \
  python -u experiments/exp06_transformer.py      --resume && \
  python -u experiments/exp03b_maxfeat_neural.py  --resume && \
  python -u experiments/exp02_threshold_calibration.py && \
  python -u experiments/exp02_threshold_calibration.py --source v05_bilstm && \
  python -u experiments/exp02_threshold_calibration.py --source v06_transformer && \
  python -u experiments/exp02_threshold_calibration.py --source v03b_maxfeat_neural && \
  python -u experiments/exp07_ensemble.py && \
  python -u experiments/exp08_llm_finetune.py --models qwen35_4b --epochs 7 && \
  python -u experiments/exp09_xai.py --families classical bilstm encoder llm --dataset no10c_no0 && \
  python -u experiments/exp10_significance.py && \
  python -u experiments/exp11_cleaning_ablation.py && \
  python -u experiments/exp12_hparam_tuning.py && \
  python -u experiments/compare_all.py --topk 20 && \
  python -u experiments/check_progress.py \
) 2>&1 | tee "logs/runall_$(date +%Y%m%d_%H%M%S).log"
```

**Step 5 — monitor / detach.** Detach with `Ctrl-b` then `d` (the run keeps going); reattach with
`tmux attach -t kxs`. From any other shell: `tail -f logs/runall_*.log`.

**Step 6 — resume after a crash.** Re-paste the Step-4 block (`--resume` skips finished cells), or
ask the orchestrator what is left and run exactly that:

```bash
python experiments/check_progress.py     # prints a tailored resume command for MISSING/PARTIAL cells
```

**Step 7 — (optional) unseen-question runs** for the neural/LLM leakage numbers (thesis future work):

```bash
KXS_SPLIT_MODE=question python -u experiments/exp05_bilstm.py
KXS_SPLIT_MODE=question python -u experiments/exp06_transformer.py
```

Tips: if `exp06`/`exp08` hit GPU out-of-memory, lower the batch size (`exp08 ... --batch_size 2`;
encoder batch is `TXFMR_BATCH` in `config.py`). The first run downloads the HuggingFace models, so
the node needs internet (or pre-populate `$HF_HOME`).

---

## Explainability (XAI)

`experiments/exp09_xai.py` runs the explainability study: it anchors each family's champion on
one common dataset, produces **Leave-One-Out (LOO) word attribution** (text highlighting) — the
single model-agnostic explanation method used across all four families — and scores it with
**faithfulness** (ERASER comprehensiveness & sufficiency, vs a random-removal baseline) and a
**plausibility** proxy (reference-overlap). Output: `results_xai/<dataset>/`
(`faithfulness_leaderboard.csv` + Khmer word-attribution heatmaps).

```bash
# Local (CPU, no extra deps) — classical + RNN pillars
python experiments/exp09_xai.py --families classical bilstm --dataset no10c_no0

# HPC (GPU) — encoder needs transformers+GPU; LLM needs unsloth+peft + the fine-tuned adapter
python experiments/exp09_xai.py --families encoder llm --dataset no10c_no0
```

Faithfulness reading: **higher comprehensiveness** and **lower sufficiency** = more faithful;
**faithfulness_gap > 0** means the explanation beats removing random words; **AOPC** averages
over k∈{10..50}%. Heatmaps show the **original Khmer answer** (word-attribution / occlusion
importance) and need an installed Khmer font (e.g. *Khmer OS*); `xai/render.py` auto-selects one.

Occlusion is the headline word-attribution method; `xai/attributions.py`
(`word_importance(method, ...)`) unifies it with the **LIME** and **SHAP** baselines used for the
method-robustness cross-check (`exp13`). The prototype uses occlusion word attribution.

---

## Live prototype (web app)

A teacher-facing Gradio app — paste **question + reference + student answer**, pick a model
pillar, get a **score + Khmer word-attribution heatmap + feedback**:

```bash
pip install -r prototype/requirements.txt
python prototype/app.py            # http://127.0.0.1:7860
```

The **Classical** pillar trains on startup and the **RNN** pillar loads the shipped champion
checkpoint — both run on CPU; the **Transformer/LLM** pillars are wired into the selector and
activate where a GPU + their weights are available. Explanation = **word attribution** (occlusion
text-highlighting). Deployment to a free Hugging Face Space is documented in `prototype/README.md`.

---

## Statistical rigor & robustness

```bash
# Champion point metrics for all four champions + the random-vs-unseen-question
# leakage comparison (classical, multi-seed). CPU-only.
python experiments/exp10_significance.py     # -> results_stats/{champion_metrics,split_compare}.csv

# Cleaning-refinement ablation: old vs new preprocessing on the classical champion,
# showing the removed zero-width/bullet noise changes QWK by <=0.004 (negligible). CPU-only.
python experiments/exp11_cleaning_ablation.py  # -> results_stats/cleaning_ablation.csv

# Hyperparameter tuning (validation-selected): classical SVR C + TF-IDF max-features. CPU-only.
python experiments/exp12_hparam_tuning.py      # -> results_stats/hparam_tuning.csv

# Attribution-method comparison (LOO vs LIME vs SHAP) + cross-method agreement + faithfulness.
# Mirrors ExASAG (SHAP+IG) and Pinto 2025 (LIME/IG/HEDGE/LOO). NEEDS: pip install lime shap captum
pip install lime shap captum                   # (machine with internet)
python experiments/exp13_attribution_comparison.py            # classical: LOO/LIME/SHAP, CPU
python experiments/exp13_attribution_comparison.py --families transformer   # Integrated Gradients, GPU/HPC
```

**Unseen-question (question-held-out) evaluation** — set the split mode so train/val/test share
no `QuestionID` (a `GroupShuffleSplit`). Driven by `config.SPLIT_MODE` or the env var:

```bash
# one cell under the stricter unseen-question split
KXS_SPLIT_MODE=question python experiments/exp01_tfidf_baseline.py --datasets no10c_no0
# full grid + v08 under the unseen-question split (HPC): set KXS_SPLIT_MODE=question and re-run exp01–exp09
```

All metrics are computed from any run's `predictions_test.csv`, so they need no re-training.
**Reproducibility note:** the classical champion's headline QWK is the *uncalibrated* continuous
score (0.795); calibration lifts the point estimate (0.795 → 0.847) but is a model-dependent
ablation (it lowers the BiLSTM's test QWK), so
headline numbers are uncalibrated. See `docs/ethics.md` for the data-governance statement.

---

## Citing

If you use this dataset or pipeline, please cite the thesis (see `docs/`). The closest prior
study, used as the methodological anchor for honest train/test-gap reporting, is Soulimani /
Alaoui et al. (2024), *Deep learning based Arabic short answer grading in serious games*,
IJECE — see `docs/references.md`.
