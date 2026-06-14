#!/usr/bin/env bash
set -u
export HF_HOME=$PWD/.hfcache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
LOG="logs/runfinal2_$(date +%Y%m%d_%H%M%S).log"
run () { echo -e "\n======== $1 ======== $(date)" | tee -a "$LOG"; shift
         "$@" >>"$LOG" 2>&1 && echo "[OK] $(date)" | tee -a "$LOG" \
         || echo "[FAIL rc=$?] $(date)" | tee -a "$LOG"; }

# 0. guard: OOM-fixed exp08 must be present
if ! grep -q "auto-halve\|VALIDATION-ONLY" experiments/exp08_llm_finetune.py; then
  echo "ERROR: exp08 not updated. scp the new exp08_llm_finetune.py first." | tee -a "$LOG"; exit 1
fi

# 1. optional perf kernels (wheel-only + non-fatal; skipped silently if no wheel for torch 2.10)
echo -e "\n======== optional perf installs ======== $(date)" | tee -a "$LOG"
pip install -q flash-linear-attention            >>"$LOG" 2>&1 || echo "fla skipped" | tee -a "$LOG"
pip install -q --only-binary :all: causal-conv1d >>"$LOG" 2>&1 || echo "causal-conv1d skipped" | tee -a "$LOG"
pip install -q --only-binary :all: flash-attn    >>"$LOG" 2>&1 || echo "flash-attn skipped (xformers fallback fine)" | tee -a "$LOG"

# 2. fix the corrupt v03b no10c: re-run FRESH (no --resume) -> clean leaderboard + val predictions
run "exp03b no10c (fresh)"   python -u experiments/exp03b_maxfeat_neural.py --datasets no10c
run "calib v03b no10c"       python -u experiments/exp02_threshold_calibration.py --source v03b_maxfeat_neural --datasets no10c

# 3. ensemble (all datasets) + comparison
run "exp07 ensemble" python -u experiments/exp07_ensemble.py
run "compare_all"    python -u experiments/compare_all.py --topk 15

# 4. LLM family (fresh, OOM-safe batch 4 -> auto 2/1, best epoch saved for HF)
rm -rf results_no10c_v08_llm_qwen35_4b results_no10c_v08_llm_gemma4_e4b \
       results_no10c_v08_llm_sealion_v45_e2b results_no10c_no0_v08_llm_qwen35_4b
run "llm ft qwen no10c"      python -u experiments/exp08_llm_finetune.py --models qwen35_4b       --epochs 7 --datasets no10c     --resume
run "llm ft gemma no10c"     python -u experiments/exp08_llm_finetune.py --models gemma4_e4b      --epochs 7 --datasets no10c     --resume
run "llm ft sealion no10c"   python -u experiments/exp08_llm_finetune.py --models sealion_v45_e2b --epochs 7 --datasets no10c     --resume
run "llm ft qwen no10c_no0"  python -u experiments/exp08_llm_finetune.py --models qwen35_4b       --epochs 7 --datasets no10c_no0 --resume
run "llm zeroshot no10c (3)" python -u experiments/exp08_llm_finetune.py --models both --zeroshot             --datasets no10c     --resume

# 5. publishable champions
run "export classical+bilstm" python -u experiments/export_models.py
run "export encoder"          python -u experiments/export_models.py --only encoder

# 6. unified 4-family LOO faithfulness
run "exp09 smoke enc+llm" python -u experiments/exp09_xai.py --families encoder llm --dataset no10c_no0 --sample 8
run "exp09 full 4-family" python -u experiments/exp09_xai.py --families classical bilstm encoder llm --dataset no10c_no0 --sample 135

# 7. all figures
run "make_figures" python -u paper/make_figures.py

echo -e "\n######## DONE $(date) ########" | tee -a "$LOG"
python experiments/check_progress.py | tee -a "$LOG"
