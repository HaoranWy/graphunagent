#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_NAME:?Set MODEL_NAME to the exact served model ID}"
: "${MODEL_REVISION:?Set MODEL_REVISION to the exact server-side root/revision}"

MODEL_URL="${MODEL_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-empty}"
DATASET_ROOT="${DATASET_ROOT:-/data/newmodel/graphunagent/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-code/diagnostics/pwsc_ci_runs/${MODEL_NAME}}"
TOP_P="${TOP_P:-}"
CACHE_ROOT="${CACHE_ROOT:-}"
MAX_WORKERS="${MAX_WORKERS:-8}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
MAX_RETRIES="${MAX_RETRIES:-0}"
STOP_TOKEN_IDS="${STOP_TOKEN_IDS:-}"
SAMPLING_TOP_K="${SAMPLING_TOP_K:-}"
SEEDS="${SEEDS:-20260615}"
# Include the two small AIME releases in the canonical matrix.  Set TASKS
# explicitly to run only a subset; every selected task still uses full-split
# paper mode, N=16, and the same fixed seed/model contract.
TASKS="${TASKS:-GSM8K MATH-500 HumanEvalPlus FOLIO AIME2025 AIME2026}"

# Accept both shell-style whitespace lists and comma-separated lists copied from
# a config/Python literal, e.g. "'GSM8K','MATH-500','FOLIO'".
normalize_words() {
  local value="${1//,/ }"
  value="${value//\'/}"
  value="${value//\"/}"
  value="${value//[/ }"
  value="${value//]/ }"
  printf '%s' "${value}"
}

read -r -a SEED_LIST <<< "$(normalize_words "${SEEDS}")"
read -r -a TASK_LIST <<< "$(normalize_words "${TASKS}")"
TOP_P_ARGS=()
if [[ -n "${TOP_P}" ]]; then
  TOP_P_ARGS=(--top_p "${TOP_P}")
fi
STOP_TOKEN_ARGS=()
if [[ -n "${STOP_TOKEN_IDS}" ]]; then
  STOP_TOKEN_ARGS=(--stop_token_ids "${STOP_TOKEN_IDS}")
fi
SAMPLING_TOP_K_ARGS=()
if [[ -n "${SAMPLING_TOP_K}" ]]; then
  SAMPLING_TOP_K_ARGS=(--sampling_top_k "${SAMPLING_TOP_K}")
fi

if (( ${#SEED_LIST[@]} == 0 )); then
  echo "SEEDS must contain at least one seed" >&2
  exit 2
fi
if (( ${#TASK_LIST[@]} == 0 )); then
  echo "TASKS must contain at least one task" >&2
  exit 2
fi

for seed in "${SEED_LIST[@]}"; do
  for task in "${TASK_LIST[@]}"; do
    output_dir="${OUTPUT_ROOT}/seed_${seed}"
    CACHE_OUTPUT_ARGS=()
    if [[ -n "${CACHE_ROOT}" ]]; then
      CACHE_OUTPUT_ARGS=(--cache_output_dir "${CACHE_ROOT}/seed_${seed}")
    fi
    python3 code/run_pipeline.py \
      --mode paper \
      --task "${task}" \
      --limit 0 \
      --seed "${seed}" \
      --model_url "${MODEL_URL}" \
      --model_name "${MODEL_NAME}" \
      --expected_model_revision "${MODEL_REVISION}" \
      --api_key "${API_KEY}" \
      --dataset_root "${DATASET_ROOT}" \
      --num_samples 16 \
      --ensemble_sizes 1,4,8,16 \
      --temperature 0.7 \
      "${TOP_P_ARGS[@]}" \
      "${STOP_TOKEN_ARGS[@]}" \
      "${SAMPLING_TOP_K_ARGS[@]}" \
      "${CACHE_OUTPUT_ARGS[@]}" \
      --max_tokens 12000 \
      --max_workers "${MAX_WORKERS}" \
      --generation_batch_size "${GENERATION_BATCH_SIZE}" \
      --request_timeout "${REQUEST_TIMEOUT}" \
      --max_retries "${MAX_RETRIES}" \
      --preflight
    python3 code/run_pipeline.py \
      --mode paper \
      --task "${task}" \
      --limit 0 \
      --seed "${seed}" \
      --model_url "${MODEL_URL}" \
      --model_name "${MODEL_NAME}" \
      --expected_model_revision "${MODEL_REVISION}" \
      --api_key "${API_KEY}" \
      --dataset_root "${DATASET_ROOT}" \
      --top_k 10 \
      --num_samples 16 \
      --ensemble_sizes 1,4,8,16 \
      --temperature 0.7 \
      "${TOP_P_ARGS[@]}" \
      "${STOP_TOKEN_ARGS[@]}" \
      "${SAMPLING_TOP_K_ARGS[@]}" \
      "${CACHE_OUTPUT_ARGS[@]}" \
      --max_tokens 12000 \
      --max_workers "${MAX_WORKERS}" \
      --generation_batch_size "${GENERATION_BATCH_SIZE}" \
      --request_timeout "${REQUEST_TIMEOUT}" \
      --max_retries "${MAX_RETRIES}" \
      --output_dir "${output_dir}"
  done
done
