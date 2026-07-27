#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ISE_PYTHON="${ISE_PYTHON:-/home/yhr/anaconda3/envs/uncertainty/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/data/newmodel/graphunagent/datasets}"
SUP_ROOT="$ROOT_DIR/.pipeline/archive/pwsc_retired_results_20260717/supplemental_experiments_20260718"
MODEL_ROOT="$SUP_ROOT/core/Llama-3.1-8B-Instruct"
OUTPUT_DIR="$SUP_ROOT/aggregate/Llama-3.1-8B-Instruct"
LOG_FILE="$SUP_ROOT/logs/summarize_llama_three_seed_20260716_18.log"

if [[ "${1:-}" == "--check-only" ]]; then
  check_only=1
elif [[ $# -eq 0 ]]; then
  check_only=0
else
  echo "Usage: bash scripts/summarize_llama_supplement_20260716_18.sh [--check-only]" >&2
  exit 2
fi

if [[ ! -x "$ISE_PYTHON" ]]; then
  echo "ERROR: uncertainty Python is not executable: $ISE_PYTHON" >&2
  exit 2
fi
for path in "$DATASET_ROOT" "$MODEL_ROOT"; do
  if [[ ! -d "$path" ]]; then
    echo "ERROR: required directory is missing: $path" >&2
    exit 2
  fi
done

for seed in 20260716 20260717 20260718; do
  run_dir="$MODEL_ROOT/seed_$seed"
  if [[ ! -d "$run_dir" ]]; then
    echo "ERROR: required seed directory is missing: $run_dir" >&2
    exit 2
  fi
  for task in GSM8K MATH-500 HumanEvalPlus FOLIO; do
    count="$(find "$run_dir" -maxdepth 1 -type f -name "manifest_${task}_*.json" | wc -l)"
    if [[ "$count" -ne 1 ]]; then
      echo "ERROR: expected exactly one $task manifest under $run_dir; found $count" >&2
      exit 2
    fi
  done
done

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")"
printf 'Python:     %q\n' "$ISE_PYTHON"
printf 'Dataset:    %q\n' "$DATASET_ROOT"
printf 'Model root: %q\n' "$MODEL_ROOT"
printf 'Output:     %q\n' "$OUTPUT_DIR"
printf 'Seeds:      20260716,20260717,20260718\n'
echo 'Manifest checks passed.'

if (( check_only )); then
  echo 'Summary was not started.'
  exit 0
fi

"$ISE_PYTHON" scripts/summarize_pwsc_ci_supplement.py \
  --run-dir "$MODEL_ROOT/seed_20260716" \
  --run-dir "$MODEL_ROOT/seed_20260717" \
  --run-dir "$MODEL_ROOT/seed_20260718" \
  --tasks GSM8K MATH-500 HumanEvalPlus FOLIO \
  --ensemble-size 16 \
  --bootstrap 2000 \
  --curve-bootstrap 1000 \
  --bootstrap-seed 20260732 \
  --failure-count 10 \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"
