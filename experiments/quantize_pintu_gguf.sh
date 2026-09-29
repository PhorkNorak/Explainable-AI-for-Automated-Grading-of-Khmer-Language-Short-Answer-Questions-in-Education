#!/usr/bin/env bash
# ============================================================================
# Text-only GGUF quantization of a merged Pintu model with llama.cpp (HPC).
#
#   bash experiments/quantize_pintu_gguf.sh --model qwen
#   bash experiments/quantize_pintu_gguf.sh --model qwen --imatrix          # + importance matrix
#   bash experiments/quantize_pintu_gguf.sh --model qwen --qtypes "Q8_0 Q4_K_M"
#
# Input : publish/full_models/<release>/          (merge_pintu_models.py output, multimodal BF16)
# Output: publish/gguf/<release>/<release>-BF16.gguf          text-only reference
#         publish/gguf/<release>/<release>-<QTYPE>[-imat].gguf  quantized builds
#         publish/manifests/quantize_<release>_<variant>.json  provenance per file
#
# Text-only: convert_hf_to_gguf.py without --mmproj exports only the language
# model; the vision/audio towers are dropped. Every output is then checked with
# validate_pintu_gguf.py --check-text-only, which fails on any non-text tensor.
#
# Order: merge in 16-bit first, quantize last (BF16 GGUF -> each QTYPE), so every
# quant derives from the same reference file.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="qwen"
MERGE_ROOT="publish/full_models"
OUT_ROOT="publish/gguf"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-.build/llama.cpp}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-}"          # pin a tag/commit for reproducibility, e.g. b6500
QTYPES="Q8_0 Q6_K Q5_K_M Q4_K_M"
USE_IMATRIX=0

while [ $# -gt 0 ]; do
  case "$1" in
    --model)        MODEL="$2"; shift 2 ;;
    --merge-root)   MERGE_ROOT="$2"; shift 2 ;;
    --out-root)     OUT_ROOT="$2"; shift 2 ;;
    --llama-cpp)    LLAMA_CPP_DIR="$2"; shift 2 ;;
    --ref)          LLAMA_CPP_REF="$2"; shift 2 ;;
    --qtypes)       QTYPES="$2"; shift 2 ;;
    --imatrix)      USE_IMATRIX=1; shift ;;
    -h|--help)      sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$MODEL" in
  qwen)    RELEASE="Pintu-Qwen3.5-4B" ;;
  gemma)   RELEASE="Pintu-Gemma4-E4B" ;;
  sealion) RELEASE="Pintu-SEA-LION-v4.5-E2B" ;;
  *) echo "--model must be qwen, gemma or sealion" >&2; exit 2 ;;
esac

SRC="$MERGE_ROOT/$RELEASE"
OUT="$OUT_ROOT/$RELEASE"
[ -f "$SRC/config.json" ] || { echo "Missing merged model: $SRC (run merge_pintu_models.py first)" >&2; exit 1; }
mkdir -p "$OUT"

record() {  # record <variant> <seconds> <inputs...> -- <outputs...> [extra key=value...]
  local variant="$1" secs="$2"; shift 2
  python experiments/_release_manifest.py record --stage quantize --release "$RELEASE" \
    --variant "$variant" --wall-seconds "$secs" "$@"
}

# --- 1. llama.cpp (clone + build once; pinned ref if given) -------------------
if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi
if [ -n "$LLAMA_CPP_REF" ]; then
  git -C "$LLAMA_CPP_DIR" fetch --tags --quiet
  git -C "$LLAMA_CPP_DIR" checkout --quiet "$LLAMA_CPP_REF"
fi
LLAMA_COMMIT="$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
echo "llama.cpp commit: $LLAMA_COMMIT"

CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release"
if command -v nvcc >/dev/null 2>&1; then CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_CUDA=ON"; fi
if [ ! -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ] || [ "${REBUILD:-0}" = "1" ]; then
  # shellcheck disable=SC2086
  cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" $CMAKE_FLAGS
  cmake --build "$LLAMA_CPP_DIR/build" --config Release -j \
    --target llama-quantize llama-imatrix llama-cli
fi
# The converter pins its own torch (CPU), transformers, numpy and huggingface-hub.
# Install them into a SEPARATE venv so the training/validation environment is
# never downgraded (installing them into the main venv breaks GPU torch).
CONVERT_VENV="${CONVERT_VENV:-.build/convert-venv}"
if [ ! -x "$CONVERT_VENV/bin/python" ]; then
  python3 -m venv "$CONVERT_VENV"
  "$CONVERT_VENV/bin/python" -m pip install --quiet --upgrade pip
  "$CONVERT_VENV/bin/python" -m pip install --quiet \
    -r "$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt" psutil
fi
CONVERT_PY="$CONVERT_VENV/bin/python"
export PYTHONPATH="$LLAMA_CPP_DIR/gguf-py:${PYTHONPATH:-}"

# --- 2. architecture support check (fail loudly, never silently mis-convert) ---
# llama.cpp prints the registry to stderr as "  - <Architecture>" lines.
ARCH="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['architectures'][0])" "$SRC/config.json")"
SUPPORTED="$("$CONVERT_PY" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" --print-supported-models 2>&1 || true)"
if grep -qE "(^|[[:space:]:])-[[:space:]]*${ARCH}[[:space:]]*$" <<< "$SUPPORTED"; then
  echo "Architecture $ARCH is supported."
elif grep -q "TEXT models" <<< "$SUPPORTED"; then
  echo "llama.cpp $LLAMA_COMMIT does not list architecture $ARCH." >&2
  echo "Update llama.cpp (or pass --ref to a newer tag) before quantizing." >&2
  exit 1
else
  echo "[warn] could not read llama.cpp's model list; conversion will fail if $ARCH is unsupported."
fi

# --- 3. text-only BF16 GGUF (the quantization reference) ----------------------
REF="$OUT/$RELEASE-BF16.gguf"
if [ ! -f "$REF" ]; then
  t0=$(date +%s)
  "$CONVERT_PY" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$SRC" --outtype bf16 --outfile "$REF"
  secs=$(( $(date +%s) - t0 ))
  "$CONVERT_PY" experiments/validate_pintu_gguf.py --check-text-only "$REF"
  record gguf_bf16 "$secs" --inputs "$SRC" --outputs "$REF" \
    --extra "llama_cpp_commit=$LLAMA_COMMIT" "architecture=$ARCH" "text_only=true"
fi

# --- 4. optional importance matrix from TRAINING prompts only -----------------
IMAT_ARGS=()
SUFFIX=""
if [ "$USE_IMATRIX" = "1" ]; then
  CALIB="$OUT/imatrix_calibration.txt"
  [ -f "$CALIB" ] || python experiments/make_imatrix_calib.py --models "$MODEL"
  IMAT="$OUT/imatrix.dat"
  if [ ! -f "$IMAT" ]; then
    t0=$(date +%s)
    "$LLAMA_CPP_DIR/build/bin/llama-imatrix" -m "$REF" -f "$CALIB" -o "$IMAT" -ngl 99 -c 512
    record imatrix "$(( $(date +%s) - t0 ))" --inputs "$REF" "$CALIB" --outputs "$IMAT" \
      --extra "llama_cpp_commit=$LLAMA_COMMIT" "calibration=training split only"
  fi
  IMAT_ARGS=(--imatrix "$IMAT")
  SUFFIX="-imat"
fi

# --- 5. quantize each type from the same BF16 reference -----------------------
for Q in $QTYPES; do
  DST="$OUT/$RELEASE-$Q$SUFFIX.gguf"
  if [ -f "$DST" ]; then echo "[skip] $DST exists"; continue; fi
  t0=$(date +%s)
  "$LLAMA_CPP_DIR/build/bin/llama-quantize" ${IMAT_ARGS[@]+"${IMAT_ARGS[@]}"} "$REF" "$DST" "$Q"
  secs=$(( $(date +%s) - t0 ))
  "$CONVERT_PY" experiments/validate_pintu_gguf.py --check-text-only "$DST"
  VARIANT="gguf_$(echo "$Q" | tr '[:upper:]' '[:lower:]')${SUFFIX:+_imat}"
  record "$VARIANT" "$secs" --inputs "$REF" --outputs "$DST" \
    --extra "llama_cpp_commit=$LLAMA_COMMIT" "qtype=$Q" "imatrix=$USE_IMATRIX" "text_only=true"
done

echo
echo "Text-only GGUF builds in $OUT:"
ls -lh "$OUT"/*.gguf
echo "Next: python experiments/validate_pintu_gguf.py --models $MODEL --device gpu   (and --device cpu)"
