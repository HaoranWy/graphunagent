#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ISE_PYTHON="${ISE_PYTHON:-/home/yhr/anaconda3/envs/uncertainty/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/data/newmodel/graphunagent/datasets}"
SUP_ROOT="$ROOT_DIR/.pipeline/archive/pwsc_retired_results_20260717/supplemental_experiments_20260718"
ARTIFACT_ROOT="$SUP_ROOT/llama70b_main_20260723"
MODEL_ROOT="$ARTIFACT_ROOT/core/Llama-3.1-70B-Instruct"
OUTPUT_DIR="$ARTIFACT_ROOT/aggregate/Llama-3.1-70B-Instruct"
LOG_FILE="$ARTIFACT_ROOT/logs/summarize_llama70b_three_seeds.log"

if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
elif [[ $# -eq 0 ]]; then
  CHECK_ONLY=0
else
  echo "Usage: bash scripts/summarize_llama70b_main_20260723.sh [--check-only]" >&2
  exit 2
fi
if [[ ! -x "$ISE_PYTHON" || ! -d "$DATASET_ROOT" || ! -d "$MODEL_ROOT" ]]; then
  echo "ERROR: Python, dataset root, or completed 70B run root is missing" >&2
  exit 2
fi

for seed in 20260716 20260717 20260718; do
  run_dir="$MODEL_ROOT/seed_$seed"
  for task in GSM8K MATH-500 HumanEvalPlus FOLIO; do
    count="$(find "$run_dir" -maxdepth 1 -type f \
      -name "manifest_${task}_*.json" | wc -l)"
    if [[ "$count" -ne 1 ]]; then
      echo "ERROR: expected one $task manifest under $run_dir; found $count" >&2
      exit 2
    fi
  done
done

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")"
printf 'Run root: %q\n' "$MODEL_ROOT"
printf 'Output:   %q\n' "$OUTPUT_DIR"
printf 'Seeds:    20260716,20260717,20260718\n'
echo 'All 12 manifest checks passed.'

if (( CHECK_ONLY )); then
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
