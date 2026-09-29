#!/usr/bin/env bash
# ============================================================================
# Pintu release, end to end, in one go (HPC tmux):
#   train -> zero-shot -> validate adapter -> merge M1 + M2 -> validate both
#   -> pick merge -> text-only GGUF quantize -> validate GGUF (GPU, CPU) -> benchmark
#
#   tmux new -s pintu
#   bash run_release.sh                         # all three: qwen, gemma, sealion
#   MODELS="qwen sealion" bash run_release.sh   # a subset, in this order
#   MODEL=gemma bash run_release.sh             # exactly one model
#   # detach: Ctrl-b then d   ·   reattach: tmux attach -t pintu
#   # watch:  tail -f logs/release_*.log
#
# Re-running is safe: every step whose output already exists is skipped, so a
# crash is recovered by running the same command again.
#
# Merge choice (automatic, by fidelity): the merge whose predictions match the
# trained adapter's most often is published (QWK breaks a tie). QLoRA trains
# against the 4-bit base, so this is usually M2 (nf4-dequant). Override with
# MERGE=bf16 or MERGE=nf4-dequant.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Several models: run each in its own pass (one failing model does not stop
# the others), then print a summary.
if [ -z "${MODEL:-}" ]; then
  MODELS="${MODELS:-qwen gemma sealion}"
  declare -A STATUS=()
  for m in $MODELS; do
    if MODEL="$m" bash "$0"; then STATUS[$m]="ok"; else STATUS[$m]="FAILED"; fi
  done
  echo; echo "=================== SUMMARY ==================="
  for m in $MODELS; do echo "  $m: ${STATUS[$m]}"; done
  # One combined benchmark over every model that finished (each per-model pass
  # rewrites results_stats/pintu_release_*.csv with only its own model).
  OK=""
  for m in $MODELS; do [ "${STATUS[$m]}" = "ok" ] && OK="$OK $m"; done
  if [ -n "$OK" ]; then
    [ -f .venv/bin/activate ] && source .venv/bin/activate
    # shellcheck disable=SC2086
    PYTHONUTF8=1 python -u experiments/exp15_pintu_release_benchmark.py --models $OK --reference auto \
      2>&1 | tee "logs/release_combined_$(date +%Y%m%d_%H%M%S).log"
  fi
  for m in $MODELS; do [ "${STATUS[$m]}" = "ok" ] || exit 1; done
  exit 0
fi
case "$MODEL" in
  qwen)    KEY="qwen35_4b";       RELEASE="Pintu-Qwen3.5-4B" ;;
  gemma)   KEY="gemma4_e4b";      RELEASE="Pintu-Gemma4-E4B" ;;
  sealion) KEY="sealion_v45_e2b"; RELEASE="Pintu-SEA-LION-v4.5-E2B" ;;
  *) echo "MODEL must be qwen, gemma or sealion" >&2; exit 2 ;;
esac
THREADS="${THREADS:-8}"

export HF_HOME="$PWD/.hfcache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUTF8=1
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

mkdir -p logs
LOG="logs/release_${MODEL}_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

BENCH="publish/benchmark/$RELEASE"
RUN_DIR="results_no10c_v08_llm_${KEY}/runs/clean_qar_${KEY}"

step() { echo; echo "=================== $* ==================="; date; }
have() { [ -e "$1" ] && echo "[skip] $1 exists"; }

{
  python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
  [ -f data/dataset_no_10c_biology.csv ] || { echo "Missing data/dataset_no_10c_biology.csv" >&2; exit 1; }

  step "1. QLoRA fine-tune ($KEY, qar, no10c)"
  have "$RUN_DIR/lora_adapter/adapter_config.json" || \
    python -u experiments/exp08_llm_finetune.py --models "$KEY" --epochs 10 --datasets no10c --input qar --resume
  [ -f "$RUN_DIR/lora_adapter/adapter_config.json" ] || { echo "Training produced no adapter" >&2; exit 1; }

  step "2. Zero-shot base (row H)"
  have "results_no10c_v08z_llm_${KEY}_zeroshot/runs/zeroshot_qar_${KEY}/predictions_test.csv" || \
    python -u experiments/exp08_llm_finetune.py --models "$KEY" --zeroshot --datasets no10c --input qar --resume

  step "3. Validate adapter (row A)"
  have "$BENCH/adapter_qlora4bit__gpu/validation.json" || \
    python -u experiments/validate_pintu_models.py --models "$MODEL" --model-source adapter

  step "4. Merge M1 (bf16) and M2 (nf4-dequant)"
  have "publish/full_models/$RELEASE/config.json" || \
    python -u experiments/merge_pintu_models.py --models "$MODEL" --merge-target bf16
  M2_OK=1
  have "publish/full_models_nf4dequant/$RELEASE/config.json" || \
    python -u experiments/merge_pintu_models.py --models "$MODEL" --merge-target nf4-dequant || {
      echo "[warn] M2 (nf4-dequant) merge failed for $RELEASE; continuing with M1 only"; M2_OK=0; }

  step "5. Validate merged models (rows B1, B2)"
  have "$BENCH/merged_bf16__gpu/validation.json" || \
    python -u experiments/validate_pintu_models.py --models "$MODEL" --merge-target bf16
  if [ "$M2_OK" = "1" ]; then
    have "$BENCH/merged_nf4dequant__gpu/validation.json" || \
      python -u experiments/validate_pintu_models.py --models "$MODEL" --merge-target nf4-dequant
  fi

  step "6. Choose merge"
  if [ "$M2_OK" != "1" ] && [ -z "${MERGE:-}" ]; then
    MERGE="bf16"
  fi
  if [ -z "${MERGE:-}" ]; then
    MERGE="$(python - "$BENCH" <<'PY'
import json, sys
from pathlib import Path
bench = Path(sys.argv[1])
def load(v):
    d = json.loads((bench / f"{v}__gpu" / "validation.json").read_text(encoding="utf-8"))
    return d["test_metrics"]["qwk"], d["adapter_comparison"].get("prediction_match_rate") or 0.0
q1, m1 = load("merged_bf16")
q2, m2 = load("merged_nf4dequant")
print(f"M1 bf16:        QWK {q1:.4f}  match vs adapter {m1:.3f}", file=sys.stderr)
print(f"M2 nf4-dequant: QWK {q2:.4f}  match vs adapter {m2:.3f}", file=sys.stderr)
# Publish the merge that reproduces the trained adapter most faithfully; QWK
# only breaks a tie (a QWK gap of 1-2 answers is noise on a ~137-answer split).
print("nf4-dequant" if (m2, q2) > (m1, q1) else "bf16")
PY
)"
  fi
  if [ "$MERGE" = "nf4-dequant" ]; then
    MERGE_ROOT="publish/full_models_nf4dequant"; REFERENCE="merged_nf4dequant"
  else
    MERGE_ROOT="publish/full_models"; REFERENCE="merged_bf16"
  fi
  echo "Chosen merge: $MERGE  ($MERGE_ROOT)"
  echo "$MERGE" > "$BENCH/CHOSEN_MERGE"

  step "7. llama-cpp-python check"
  if ! python -c "import llama_cpp" 2>/dev/null; then
    echo "llama-cpp-python missing; building it (CUDA if nvcc is available)"
    if command -v nvcc >/dev/null 2>&1; then
      CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
    else
      pip install llama-cpp-python
    fi
  fi
  GPU_GGUF="$(python -c "import llama_cpp; print(int(llama_cpp.llama_supports_gpu_offload()))")"
  echo "llama.cpp GPU offload available: $GPU_GGUF"

  step "8. Text-only GGUF quantization"
  bash experiments/quantize_pintu_gguf.sh --model "$MODEL" --merge-root "$MERGE_ROOT"

  step "9. Validate GGUF builds"
  if [ "$GPU_GGUF" = "1" ]; then
    python -u experiments/validate_pintu_gguf.py --models "$MODEL" --device gpu --processor-root "$MERGE_ROOT" \
      || echo "[warn] GPU GGUF validation failed; continuing with CPU"
  else
    echo "[skip] GPU GGUF validation (llama-cpp-python built without GPU offload)"
  fi
  python -u experiments/validate_pintu_gguf.py --models "$MODEL" --device cpu --threads "$THREADS" \
    --processor-root "$MERGE_ROOT"

  step "10. Benchmark tables, figures, report"
  python -u experiments/exp15_pintu_release_benchmark.py --models "$MODEL" --reference "$REFERENCE"

  echo
  echo "DONE. Results:"
  echo "  $BENCH/REPORT.md"
  echo "  results_stats/pintu_release_benchmark.csv"
  echo "  results_stats/figures/pintu_release/$RELEASE/"
  echo "  publish/manifests/"
} 2>&1 | tee "$LOG"
