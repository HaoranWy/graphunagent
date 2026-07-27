#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ISE_PYTHON="${ISE_PYTHON:-/home/yhr/anaconda3/envs/uncertainty/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/data/newmodel/graphunagent/datasets}"
SUP_ROOT="$ROOT_DIR/.pipeline/archive/pwsc_retired_results_20260717/supplemental_experiments_20260718"
QWEN_ROOT="/data/newmodel/IE/LLM/Qwen2.5-7B"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_ROOT="$SUP_ROOT/qwen_main_20260721/core/Qwen2.5-7B-Instruct"
CACHE_ROOT="$SUP_ROOT/qwen_main_20260721/generation_cache/Qwen2.5-7B-Instruct"
LOG_FILE="$SUP_ROOT/qwen_main_20260721/logs/run_qwen_main_three_seeds.log"

if [[ $# -ne 0 ]]; then
  echo "Usage: bash scripts/run_qwen_main_20260721.sh" >&2
  exit 2
fi
if [[ ! -x "$ISE_PYTHON" ]]; then
  echo "ERROR: uncertainty Python is not executable: $ISE_PYTHON" >&2
  exit 2
fi
for required in "$DATASET_ROOT" "$QWEN_ROOT"; do
  if [[ ! -d "$required" ]]; then
    echo "ERROR: required directory is missing: $required" >&2
    exit 2
  fi
done
if [[ ! -f "$QWEN_ROOT/config.json" || ! -f "$QWEN_ROOT/tokenizer_config.json" ]]; then
  echo "ERROR: Qwen model directory is incomplete: $QWEN_ROOT" >&2
  exit 2
fi

for seed in 20260721 20260722 20260723; do
  run_dir="$OUTPUT_ROOT/seed_$seed"
  if find "$run_dir" -maxdepth 1 -type f -name 'manifest_*.json' -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: existing manifest found under $run_dir; refusing to mix or overwrite runs" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT" "$(dirname "$LOG_FILE")"
export PATH="$(dirname "$ISE_PYTHON"):$PATH"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/code${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

printf 'Python:     %q\n' "$ISE_PYTHON"
printf 'Dataset:    %q\n' "$DATASET_ROOT"
printf 'Model:      %q\n' "$MODEL_NAME"
printf 'Revision:   %q\n' "$QWEN_ROOT"
printf 'Endpoint:   %s\n' 'http://127.0.0.1:8000/v1'
printf 'Seeds:      20260721,20260722,20260723\n'
printf 'Output:     %q\n' "$OUTPUT_ROOT"
printf 'Log:        %q\n' "$LOG_FILE"

MODEL_NAME="$MODEL_NAME" \
MODEL_REVISION="$QWEN_ROOT" \
MODEL_URL=http://127.0.0.1:8000/v1 \
API_KEY=empty \
DATASET_ROOT="$DATASET_ROOT" \
TOP_P=0.9 \
REQUEST_TIMEOUT=3600 \
MAX_RETRIES=1 \
MAX_WORKERS=4 \
GENERATION_BATCH_SIZE=16 \
SEEDS='20260721 20260722 20260723' \
TASKS='GSM8K MATH-500 HumanEvalPlus FOLIO' \
OUTPUT_ROOT="$OUTPUT_ROOT" \
CACHE_ROOT="$CACHE_ROOT" \
bash scripts/run_paper_matrix.sh \
  2>&1 | tee "$LOG_FILE"
