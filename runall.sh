#!/usr/bin/env bash
set -u
export HF_HOME=$PWD/.hfcache
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
LOG="logs/runall_$(date +%Y%m%d_%H%M%S).log"
run () { echo -e "\n======== $1 ======== $(date)" | tee -a "$LOG"; shift
         "$@" >>"$LOG" 2>&1 && echo "[OK] $(date)" | tee -a "$LOG" \
         || echo "[FAIL rc=$?] $(date)" | tee -a "$LOG"; }

# 1. finish the grid (v03b remaining, calibrations, ensemble)
run "exp03b neural"  python -u experiments/exp03b_maxfeat_neural.py --datasets no10c no10c_no0 --resume
run "calib v05"      python -u experiments/exp02_threshold_calibration.py --source v05_bilstm
run "calib v06"      python -u experiments/exp02_threshold_calibration.py --source v06_transformer
run "calib v03b"     python -u experiments/exp02_threshold_calibration.py --source v03b_maxfeat_neural
run "exp07 ensemble" python -u experiments/exp07_ensemble.py

# 2. stats for the figures
run "exp10 significance" python -u experiments/exp10_significance.py
run "exp11 cleaning"     python -u experiments/exp11_cleaning_ablation.py
run "exp12 hparam"       python -u experiments/exp12_hparam_tuning.py
run "compare_all"        python -u experiments/compare_all.py --topk 15

# 3. reclaim space BEFORE the LLMs/exports write weights (excludes champions, publish, venv)
echo -e "\n======== cleanup per-cell weights ======== $(date)" | tee -a "$LOG"
find . -path ./results/champions -prune -o -path ./.venv -prune -o -path ./publish -prune -o \
     -name "*.pt" -print -o -name "*.bin" -print -o -name "*.safetensors" -print | xargs -r rm -f
df -h / | tee -a "$LOG"

# 4. LLM family (fresh chat-template runs; best epoch saved for HF)
rm -rf results_no10c_v08_llm_qwen35_4b results_no10c_v08_llm_gemma4_e4b
run "llm ft qwen no10c"      python -u experiments/exp08_llm_finetune.py --models qwen35_4b       --epochs 7 --datasets no10c     --resume
run "llm ft gemma no10c"     python -u experiments/exp08_llm_finetune.py --models gemma4_e4b      --epochs 7 --datasets no10c     --resume
run "llm ft sealion no10c"   python -u experiments/exp08_llm_finetune.py --models sealion_v45_e2b --epochs 7 --datasets no10c     --resume
run "llm ft qwen no10c_no0"  python -u experiments/exp08_llm_finetune.py --models qwen35_4b       --epochs 7 --datasets no10c_no0 --resume
run "llm zeroshot no10c (3)" python -u experiments/exp08_llm_finetune.py --models both --zeroshot             --datasets no10c     --resume

# 5. export publishable champions (HuggingFace-ready, with sanity QWK)
run "export classical+bilstm" python -u experiments/export_models.py
run "export encoder"          python -u experiments/export_models.py --only encoder

# 6. unified 4-family LOO faithfulness (smoke, then full)
run "exp09 smoke enc+llm" python -u experiments/exp09_xai.py --families encoder llm --dataset no10c_no0 --sample 8
run "exp09 full 4-family" python -u experiments/exp09_xai.py --families classical bilstm encoder llm --dataset no10c_no0 --sample 135

# 7. regenerate every figure
run "make_figures" python -u paper/make_figures.py

echo -e "\n######## ALL DONE $(date) ########" | tee -a "$LOG"
python experiments/check_progress.py | tee -a "$LOG"
