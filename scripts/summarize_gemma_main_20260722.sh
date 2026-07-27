#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ISE_PYTHON="${ISE_PYTHON:-/home/yhr/anaconda3/envs/uncertainty/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/data/newmodel/graphunagent/datasets}"
SUP_ROOT="$ROOT_DIR/.pipeline/archive/pwsc_retired_results_20260717/supplemental_experiments_20260718"
RUN_DIRS=(
  "$SUP_ROOT/gemma_main_20260721/core/gemma-3-12b-it/seed_20260716"
  "$SUP_ROOT/gemma_main_20260721/core/gemma-3-12b-it/seed_20260717"
  "$SUP_ROOT/gemma_main_20260721/core/gemma-3-12b-it/seed_20260718"
)
OUTPUT_DIR="$SUP_ROOT/gemma_main_20260721/aggregate/gemma-3-12b-it"
LOG_FILE="$SUP_ROOT/gemma_main_20260721/logs/summarize_gemma_three_seeds.log"

if [[ "${1:-}" == "--check-only" ]]; then
  check_only=1
elif [[ $# -eq 0 ]]; then
  check_only=0
else
  echo "Usage: bash scripts/summarize_gemma_main_20260722.sh [--check-only]" >&2
  exit 2
fi
if [[ ! -x "$ISE_PYTHON" ]]; then
  echo "ERROR: uncertainty Python is not executable: $ISE_PYTHON" >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "ERROR: dataset root is missing: $DATASET_ROOT" >&2
  exit 2
fi

run_args=()
for run_dir in "${RUN_DIRS[@]}"; do
  if [[ ! -d "$run_dir" ]]; then
    echo "ERROR: Gemma run directory is missing: $run_dir" >&2
    exit 2
  fi
  for task in GSM8K MATH-500 HumanEvalPlus FOLIO; do
    count="$(find "$run_dir" -maxdepth 1 -type f -name "manifest_${task}_*.json" | wc -l)"
    if [[ "$count" -ne 1 ]]; then
      echo "ERROR: expected exactly one $task manifest under $run_dir; found $count" >&2
      exit 2
    fi
  done
  run_args+=(--run-dir "$run_dir")
done

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")"
printf 'Python:     %q\n' "$ISE_PYTHON"
printf 'Dataset:    %q\n' "$DATASET_ROOT"
printf 'Run:        %q\n' "${RUN_DIRS[@]}"
printf 'Output:     %q\n' "$OUTPUT_DIR"
printf 'Seeds:      20260716 20260717 20260718\n'
echo 'All 12 manifest checks passed.'

if (( check_only )); then
  echo 'Summary was not started.'
  exit 0
fi

"$ISE_PYTHON" scripts/summarize_pwsc_ci_supplement.py \
  "${run_args[@]}" \
  --tasks GSM8K MATH-500 HumanEvalPlus FOLIO \
  --ensemble-size 16 \
  --bootstrap 2000 \
  --curve-bootstrap 1000 \
  --bootstrap-seed 20260732 \
  --failure-count 10 \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"
