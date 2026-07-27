#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ISE_PYTHON="${ISE_PYTHON:-/home/yhr/anaconda3/envs/uncertainty/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/data/newmodel/graphunagent/datasets}"
SUP_ROOT="$ROOT_DIR/.pipeline/archive/pwsc_retired_results_20260717/supplemental_experiments_20260718"
CACHE_DIR="$SUP_ROOT/generation_cache/Llama-3.1-8B-Instruct/seed_20260716"
CACHE_FILE="generations_MATH-500_1784389647220358198_6954315d92.jsonl"
PARTIAL_CACHE="$CACHE_DIR/$CACHE_FILE"
RESUME_OUT="$SUP_ROOT/core/Llama-3.1-8B-Instruct/seed_20260716"
LOG_FILE="$SUP_ROOT/logs/run_llama_math500_seed_20260716_resume.log"

if [[ ! -x "$ISE_PYTHON" ]]; then
  echo "ERROR: uncertainty Python is not executable: $ISE_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$PARTIAL_CACHE" ]]; then
  echo "ERROR: validated partial cache is missing: $PARTIAL_CACHE" >&2
  exit 2
fi

cache_lines="$(wc -l < "$PARTIAL_CACHE")"
if (( cache_lines < 2 || cache_lines > 501 )); then
  echo "ERROR: invalid MATH-500 cache line count: $cache_lines (expected 2..501)" >&2
  exit 2
fi

mkdir -p "$RESUME_OUT" "$(dirname "$LOG_FILE")"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/code${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

printf 'Python: %q\n' "$ISE_PYTHON"
printf 'Cache:  %q\n' "$PARTIAL_CACHE"
printf 'Lines:  %s (header + %s complete samples)\n' "$cache_lines" "$((cache_lines - 1))"
printf 'Output: %q\n' "$RESUME_OUT"
printf 'Log:    %q\n' "$LOG_FILE"

common_args=(
  --mode paper
  --task MATH-500
  --limit 0
  --seed 20260716
  --model_url http://127.0.0.1:8000/v1
  --model_name meta-llama/Llama-3.1-8B-Instruct
  --expected_model_revision /data/newmodel/IE/LLM/Llama-3.1-8B-Instruct
  --api_key empty
  --dataset_root "$DATASET_ROOT"
  --top_k 10
  --num_samples 16
  --ensemble_sizes 1,4,8,16
  --temperature 0.7
  --top_p 0.9
  --max_tokens 12000
  --max_workers 8
  --generation_batch_size 32
  --request_timeout 3600
  --max_retries 1
  --cache_path "$PARTIAL_CACHE"
  --resume_cache
)

"$ISE_PYTHON" code/run_pipeline.py "${common_args[@]}" --preflight

if [[ "${1:-}" == "--check-only" ]]; then
  echo "Preflight passed; generation was not started."
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: bash scripts/resume_llama_math500_20260716.sh [--check-only]" >&2
  exit 2
fi

"$ISE_PYTHON" code/run_pipeline.py \
  "${common_args[@]}" \
  --output_dir "$RESUME_OUT" \
  2>&1 | tee -a "$LOG_FILE"
