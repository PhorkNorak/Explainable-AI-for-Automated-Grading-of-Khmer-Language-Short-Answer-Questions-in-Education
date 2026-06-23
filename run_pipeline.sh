#!/usr/bin/env bash
# ============================================================================
# Full Khmer ASAG pipeline, end to end, in one go (designed for an HPC tmux run).
#
#   tmux new -s kxs                 # start a session that survives SSH disconnect
#   bash run_pipeline.sh            # run everything (logs to logs/pipeline_*.log)
#   # detach: Ctrl-b then d   ·   reattach: tmux attach -t kxs
#   # watch from another shell: tail -f logs/pipeline_*.log
#
# Every TRAINING step takes --resume, so a crash/preemption is recovered by just
# re-running this script: finished cells are skipped, only missing ones re-run.
#
# GPU note: exp06 (encoder), exp03b (neural max-score), and the two exp08 lines (LLM zero-shot +
# QLoRA fine-tune) need a CUDA GPU. On a CPU-only box, comment those lines and drop "encoder llm"
# from the exp09 --families list; classical + BiLSTM + all analyses still run on CPU.
# The LLM family (3 bases x {ra, qar} x {zero-shot, fine-tune}) on no10c is the heaviest part
# (~15-17 GPU-hours wall-clock; pure training is only ~5 GPU-hr, the rest is per-epoch generation/eval).
# Whole pipeline ~24-28 GPU-hr (~1 day on one A40). exp14 (frontier APIs, separate) ~ $5-11 total.
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# --- environment ---
export HF_HOME="$PWD/.hfcache"          # cache mBERT / XLM-R / GTE / Qwen inside the project
export TOKENIZERS_PARALLELISM=false
export PYTHONUTF8=1                      # Khmer-safe stdout

# activate the project venv if present (edit the path if yours lives elsewhere)
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

mkdir -p logs
LOG="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import khmernltk; print('khmernltk OK')"

{
  echo "=================== 1. MODEL GRID (3 datasets x text axes) ==================="
  python -u experiments/exp01_tfidf_baseline.py        --resume
  python -u experiments/exp03_maxscore_feature.py      --resume
  python -u experiments/exp04_bucket_svr.py            --resume
  python -u experiments/exp05_bilstm.py                --resume
  python -u experiments/exp06_transformer.py           --resume     # GPU/HPC
  python -u experiments/exp03b_maxfeat_neural.py       --resume     # GPU/HPC
  python -u experiments/exp02_threshold_calibration.py
  python -u experiments/exp02_threshold_calibration.py --source v05_bilstm
  python -u experiments/exp02_threshold_calibration.py --source v06_transformer
  python -u experiments/exp02_threshold_calibration.py --source v03b_maxfeat_neural
  python -u experiments/exp07_ensemble.py
  # LLM family (KhmerGrader) on no10c only, both input formats (qar, ra), all 3 bases
  # (qwen35_4b, gemma4_e4b, sealion_v45_e2b). Zero-shot baselines THEN QLoRA fine-tune.
  python -u experiments/exp08_llm_finetune.py --models both --zeroshot --datasets no10c --input qar  # GPU/HPC
  python -u experiments/exp08_llm_finetune.py --models both --zeroshot --datasets no10c --input ra   # GPU/HPC
  python -u experiments/exp08_llm_finetune.py --models both --epochs 10 --datasets no10c --input qar # GPU/HPC
  python -u experiments/exp08_llm_finetune.py --models both --epochs 10 --datasets no10c --input ra  # GPU/HPC

  echo "=================== 2. ANALYSES (SHAP, metrics, ablations) ==================="
  # SHAP: full on the cheap pillars; capped on the LLM (each SHAP eval is a full generation).
  python -u experiments/exp09_xai.py --families classical bilstm encoder --dataset no10c
  python -u experiments/exp09_xai.py --families llm --dataset no10c --shap-max-evals 50   # GPU/HPC
  python -u experiments/exp10_significance.py       # champion point metrics
  python -u experiments/exp11_cleaning_ablation.py  # format-noise robustness
  python -u experiments/exp12_hparam_tuning.py      # classical hyperparameter sweep

  echo "=================== 2b. FRONTIER BASELINES (paid API, ~\$5-11, cached) ==================="
  # Frontier LLMs via one OpenRouter key (zero-shot, bare + reasoning, no10c). Responses cache
  # to results_frontier/, so a re-run is free; skipped automatically if no key is exported.
  if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    python -u experiments/exp14_frontier_baselines.py --gateway openrouter --dataset no10c
  else
    echo "[skip] OPENROUTER_API_KEY not set; skipping exp14 frontier baselines."
  fi

  echo "=================== 3. AGGREGATE + FIGURES ==================="
  python -u experiments/compare_all.py --topk 20
  python -u paper/make_figures.py                   # regenerate the figure suite from the CSVs
  python -u experiments/check_progress.py
} 2>&1 | tee "$LOG"

echo
echo "Pipeline finished. Manual follow-up (needs a browser, for correct Khmer shaping):"
echo "  1) Open the SHAP heatmap HTML galleries and screenshot each to a PNG:"
echo "       results_xai/no10c/heatmaps/{classical,bilstm,encoder,llm}/*_gallery.html"
echo "     Save as thesis/figures/heatmap_{classical,bilstm,encoder,llm}.png"
echo "  2) Then propagate refreshed numbers from results_stats/*.csv into the thesis/paper tables."
