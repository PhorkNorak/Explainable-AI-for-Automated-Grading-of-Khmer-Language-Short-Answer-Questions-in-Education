#!/usr/bin/env bash
# ============================================================================
# Parallel Khmer ASAG pipeline: CPU, GPU, and API streams run AT THE SAME TIME,
# then a serial aggregation stage once all three finish.
#
#   tmux new -s kxs
#   export OPENROUTER_API_KEY="sk-or-..."     # enables the API stream
#   bash run_parallel.sh
#   # watch each stream from other panes:
#   #   tail -f logs/cpu_*.log    tail -f logs/gpu_*.log    tail -f logs/api_*.log
#
# Why three streams: they use different resources and do not contend.
#   CPU  = classical (exp01/03/04) + BiLSTM (exp05) + ablations (exp11/exp12)
#   GPU  = transformer (exp06) -> neural max-feat (exp03b) -> LLM family (exp08)
#          (serial WITHIN the stream so two jobs never share one A40)
#   API  = exp14 frontier baselines (network only)
# The aggregation stage (calibration, ensemble, SHAP, champions, figures) needs
# the trained models, so it runs only after the barrier (`wait`).
#
# IMPORTANT: run on a CLEAN tree. If you just `git reset --hard`, the committed
# result dirs come back and --resume will skip fresh training. Wipe first:
#   rm -rf results results_full* results_no10c* results_stats results_xai
# (keep results_frontier — the paid DeepSeek cache).
# ============================================================================

set -uo pipefail
cd "$(dirname "$0")"

export HF_HOME="$PWD/.hfcache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUTF8=1
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
CPU_LOG="logs/cpu_$TS.log"; GPU_LOG="logs/gpu_$TS.log"; API_LOG="logs/api_$TS.log"
AGG_LOG="logs/agg_$TS.log"

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

cpu_stream() {
  echo "[cpu] start $(date)"
  python -u experiments/exp01_tfidf_baseline.py        --resume
  python -u experiments/exp03_maxscore_feature.py      --resume
  python -u experiments/exp04_bucket_svr.py            --resume
  python -u experiments/exp05_bilstm.py                --resume
  python -u experiments/exp11_cleaning_ablation.py     # CPU, independent of the grid
  python -u experiments/exp12_hparam_tuning.py         # CPU, independent of the grid
  echo "[cpu] done $(date)"
}

ft() {  # one fine-tune cell: ft <model> <input>
  python -u experiments/exp08_llm_finetune.py --models "$1" --epochs 10 --datasets no10c --input "$2"
}

zs() {  # one zero-shot cell: zs <model> <input>
  python -u experiments/exp08_llm_finetune.py --models "$1" --zeroshot --datasets no10c --input "$2"
}

gpu_stream() {
  echo "[gpu] start $(date)"
  python -u experiments/exp06_transformer.py           --resume   # skipped if already on disk
  python -u experiments/exp03b_maxfeat_neural.py       --resume   # skipped if already on disk

  # PHASE 1: all 3 zero-shots in parallel (3 bases x ~10GB ~= 30GB, safe). Each lane does
  # its model's qar then ra. Zero-shot is inference, so this is the cheap, safe parallelism.
  echo "[gpu] phase 1: parallel zero-shot $(date)"
  ( zs qwen35_4b qar;       zs qwen35_4b ra )       &  Z1=$!
  ( zs gemma4_e4b qar;      zs gemma4_e4b ra )      &  Z2=$!
  ( zs sealion_v45_e2b qar; zs sealion_v45_e2b ra ) &  Z3=$!
  wait "$Z1" "$Z2" "$Z3"
  echo "[gpu] phase 1 done $(date)"

  # PHASE 2: all 3 fine-tunes in parallel (~38-44GB on a 46GB A40 - TIGHT). Memory is
  # allocated in the first steps and stays flat, so if it does not OOM in the first ~10 min
  # it will not OOM at all. If logs/gpu_*.log shows CUDA OOM, fall back to 2-wide.
  echo "[gpu] phase 2: parallel fine-tune $(date)"
  ( ft qwen35_4b qar;       ft qwen35_4b ra )       &  F1=$!
  ( ft gemma4_e4b qar;      ft gemma4_e4b ra )      &  F2=$!
  ( ft sealion_v45_e2b qar; ft sealion_v45_e2b ra ) &  F3=$!
  wait "$F1"; echo "[gpu] qwen lane exit $?"
  wait "$F2"; echo "[gpu] gemma lane exit $?"
  wait "$F3"; echo "[gpu] sealion lane exit $?"
  echo "[gpu] done $(date)"
}

api_stream() {
  echo "[api] start $(date)"
  if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    python -u experiments/exp14_frontier_baselines.py --gateway openrouter --dataset no10c
  else
    echo "[api] OPENROUTER_API_KEY not set; skipping exp14 frontier baselines."
  fi
  echo "[api] done $(date)"
}

echo "Logs: $CPU_LOG | $GPU_LOG | $API_LOG  (aggregation -> $AGG_LOG)"
cpu_stream > "$CPU_LOG" 2>&1 &  CPU_PID=$!
gpu_stream > "$GPU_LOG" 2>&1 &  GPU_PID=$!
api_stream > "$API_LOG" 2>&1 &  API_PID=$!
echo "Launched in parallel: CPU=$CPU_PID  GPU=$GPU_PID  API=$API_PID"

# Barrier: wait for all three, capture exit codes (do not abort on a single failure).
wait "$CPU_PID"; CPU_RC=$?; echo "[cpu] exit $CPU_RC"
wait "$GPU_PID"; GPU_RC=$?; echo "[gpu] exit $GPU_RC"
wait "$API_PID"; API_RC=$?; echo "[api] exit $API_RC"
echo "Streams finished (cpu=$CPU_RC gpu=$GPU_RC api=$API_RC)."

echo "=================== AGGREGATION (needs all trained models) ==================="
{
  python -u experiments/exp02_threshold_calibration.py
  python -u experiments/exp02_threshold_calibration.py --source v05_bilstm
  python -u experiments/exp02_threshold_calibration.py --source v06_transformer
  python -u experiments/exp02_threshold_calibration.py --source v03b_maxfeat_neural
  python -u experiments/exp07_ensemble.py
  # SHAP: full on the cheap pillars; capped on the LLM (each eval is a full generation).
  python -u experiments/exp09_xai.py --families classical bilstm encoder --dataset no10c
  python -u experiments/exp09_xai.py --families llm --dataset no10c --shap-max-evals 50
  python -u experiments/exp10_significance.py
  python -u experiments/compare_all.py --topk 20
  python -u paper/make_figures.py
  python -u experiments/check_progress.py
} 2>&1 | tee "$AGG_LOG"

echo
echo "Parallel pipeline finished. Manual follow-up (needs a browser for Khmer shaping):"
echo "  1) Screenshot the SHAP galleries:"
echo "       results_xai/no10c/heatmaps/{classical,bilstm,encoder,llm}/*_gallery.html"
echo "     Save as thesis/figures/heatmap_{classical,bilstm,encoder,llm}.png"
echo "  2) Propagate refreshed numbers from results_stats/*.csv into the thesis/paper tables."
