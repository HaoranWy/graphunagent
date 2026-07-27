#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ISE_PYTHON="${ISE_PYTHON:-/home/yhr/anaconda3/envs/uncertainty/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/data/newmodel/graphunagent/datasets}"
MODEL_ROOT="/data/newmodel/IE/LLM/gemma-3-12b-it"
MODEL_NAME="google/gemma-3-12b-it"
SUP_ROOT="$ROOT_DIR/.pipeline/archive/pwsc_retired_results_20260717/supplemental_experiments_20260718"
ARTIFACT_ROOT="$SUP_ROOT/gemma_main_20260721"
OUTPUT_ROOT="$ARTIFACT_ROOT/core/gemma-3-12b-it"
CACHE_ROOT="$ARTIFACT_ROOT/generation_cache/gemma-3-12b-it"
LOG_FILE="$ARTIFACT_ROOT/logs/run_gemma_main_seed_20260716.log"

MODE="${1:-clean}"
if [[ $# -gt 1 || ( "$MODE" != "clean" && "$MODE" != "--check-only" && "$MODE" != "--resume" ) ]]; then
  echo "Usage: bash scripts/run_gemma_main_20260716_clean.sh [--check-only|--resume]" >&2
  exit 2
fi
if [[ ! -x "$ISE_PYTHON" ]]; then
  echo "ERROR: uncertainty Python is not executable: $ISE_PYTHON" >&2
  exit 2
fi
for required in "$DATASET_ROOT" "$MODEL_ROOT"; do
  if [[ ! -d "$required" ]]; then
    echo "ERROR: required directory is missing: $required" >&2
    exit 2
  fi
done

mapfile -t service_lines < <(pgrep -af "vllm serve $MODEL_ROOT" || true)
if [[ ${#service_lines[@]} -ne 1 ]]; then
  echo "ERROR: expected exactly one Gemma vLLM service; found ${#service_lines[@]}" >&2
  printf '  %s\n' "${service_lines[@]}" >&2
  exit 2
fi
service_command="${service_lines[0]}"
for required_arg in \
  '--port 8001' \
  '--served-model-name google/gemma-3-12b-it' \
  '--tensor-parallel-size 2' \
  '--max-model-len 24000' \
  '--max-logprobs 100' \
  '--generation-config vllm'; do
  if [[ "$service_command" != *"$required_arg"* ]]; then
    echo "ERROR: Gemma service is missing frozen argument: $required_arg" >&2
    echo "Observed: $service_command" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT" "$(dirname "$LOG_FILE")"
export PATH="$(dirname "$ISE_PYTHON"):$PATH"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/code${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

printf 'Python:       %q\n' "$ISE_PYTHON"
printf 'Dataset:      %q\n' "$DATASET_ROOT"
printf 'Model:        %q\n' "$MODEL_NAME"
printf 'Revision:     %q\n' "$MODEL_ROOT"
printf 'Endpoint:     http://127.0.0.1:8001/v1\n'
printf 'Seed:         20260716\n'
printf 'Stop IDs:     106 (<end_of_turn>)\n'
printf 'Sampling k:   0 (explicit neutral vLLM top-k)\n'
printf 'Artifact root:%q\n' "$ARTIFACT_ROOT"

"$ISE_PYTHON" scripts/probe_gemma_stop_20260721.py \
  --base-url http://127.0.0.1:8001/v1 \
  --model "$MODEL_NAME" \
  --expected-root "$MODEL_ROOT" \
  --stop-token-id 106 \
  --sampling-top-k 0 \
  2>&1 | tee -a "$LOG_FILE"

preflight_args=(
  --mode paper
  --task GSM8K
  --limit 0
  --seed 20260716
  --model_url http://127.0.0.1:8001/v1
  --model_name "$MODEL_NAME"
  --expected_model_revision "$MODEL_ROOT"
  --api_key empty
  --dataset_root "$DATASET_ROOT"
  --top_k 10
  --num_samples 16
  --ensemble_sizes 1,4,8,16
  --temperature 0.7
  --top_p 0.9
  --max_tokens 12000
  --max_workers 2
  --generation_batch_size 8
  --request_timeout 3600
  --max_retries 0
  --stop_token_ids 106
  --sampling_top_k 0
  --cache_output_dir "$CACHE_ROOT/seed_20260716"
  --preflight
)
"$ISE_PYTHON" code/run_pipeline.py "${preflight_args[@]}"

if [[ "$MODE" == "--check-only" ]]; then
  echo "Service and clean-run preflight passed; generation was not started."
  exit 0
fi

SEED_OUTPUT="$OUTPUT_ROOT/seed_20260716"
SEED_CACHE="$CACHE_ROOT/seed_20260716"

if [[ "$MODE" == "clean" ]]; then
  if find "$SEED_OUTPUT" -maxdepth 1 -type f \
    -name 'manifest_*.json' -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: clean Gemma output already contains a manifest; use --resume" >&2
    exit 2
  fi
  if find "$SEED_CACHE" -maxdepth 1 -type f \
    -name 'generations_*.jsonl' -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: clean Gemma cache root is not empty; use --resume" >&2
    exit 2
  fi
fi

common_args=(
  --mode paper
  --limit 0
  --seed 20260716
  --model_url http://127.0.0.1:8001/v1
  --model_name "$MODEL_NAME"
  --expected_model_revision "$MODEL_ROOT"
  --api_key empty
  --dataset_root "$DATASET_ROOT"
  --top_k 10
  --num_samples 16
  --ensemble_sizes 1,4,8,16
  --temperature 0.7
  --top_p 0.9
  --max_tokens 12000
  --max_workers 2
  --generation_batch_size 8
  --request_timeout 3600
  --max_retries 0
  --stop_token_ids 106
  --sampling_top_k 0
)

for task in GSM8K MATH-500 HumanEvalPlus FOLIO; do
  mapfile -t manifests < <(
    find "$SEED_OUTPUT" -maxdepth 1 -type f \
      -name "manifest_${task}_*.json" -print 2>/dev/null | sort
  )
  if (( ${#manifests[@]} > 1 )); then
    echo "ERROR: expected at most one $task manifest; found ${#manifests[@]}" >&2
    printf '  %s\n' "${manifests[@]}" >&2
    exit 2
  fi
  if (( ${#manifests[@]} == 1 )); then
    echo "SKIP: $task already completed: ${manifests[0]}"
    continue
  fi

  mapfile -t caches < <(
    find "$SEED_CACHE" -maxdepth 1 -type f \
      -name "generations_${task}_*.jsonl" -print 2>/dev/null | sort
  )
  if (( ${#caches[@]} > 1 )); then
    echo "ERROR: expected at most one $task cache; found ${#caches[@]}" >&2
    printf '  %s\n' "${caches[@]}" >&2
    exit 2
  fi

  cache_args=(--cache_output_dir "$SEED_CACHE")
  if (( ${#caches[@]} == 1 )); then
    if [[ "$MODE" != "--resume" ]]; then
      echo "ERROR: found an existing $task cache; rerun with --resume" >&2
      echo "  ${caches[0]}" >&2
      exit 2
    fi
    cache_args=(--cache_path "${caches[0]}" --resume_cache)
    echo "RESUME: $task from ${caches[0]}"
  else
    echo "START: $task with a new cache under $SEED_CACHE"
  fi

  "$ISE_PYTHON" code/run_pipeline.py \
    "${common_args[@]}" \
    --task "$task" \
    "${cache_args[@]}" \
    --preflight \
    2>&1 | tee -a "$LOG_FILE"
  "$ISE_PYTHON" code/run_pipeline.py \
    "${common_args[@]}" \
    --task "$task" \
    "${cache_args[@]}" \
    --output_dir "$SEED_OUTPUT" \
    2>&1 | tee -a "$LOG_FILE"

  mapfile -t completed_manifests < <(
    find "$SEED_OUTPUT" -maxdepth 1 -type f \
      -name "manifest_${task}_*.json" -print 2>/dev/null | sort
  )
  if (( ${#completed_manifests[@]} != 1 )); then
    echo "ERROR: $task finished without exactly one manifest" >&2
    exit 2
  fi
done

echo "Gemma seed 20260716 complete: all four task manifests are present."
