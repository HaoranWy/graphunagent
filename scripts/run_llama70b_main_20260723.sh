#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ISE_PYTHON="${ISE_PYTHON:-/home/yhr/anaconda3/envs/uncertainty/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/data/newmodel/graphunagent/datasets}"
MODEL_ROOT="/data/newmodel/IE/LLM/Llama-3.1-70B-Instruct"
MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
MODEL_URLS=(
  "http://127.0.0.1:8002/v1"
  "http://127.0.0.1:8003/v1"
)
SUP_ROOT="$ROOT_DIR/.pipeline/archive/pwsc_retired_results_20260717/supplemental_experiments_20260718"
ARTIFACT_ROOT="$SUP_ROOT/llama70b_main_20260723"
OUTPUT_ROOT="$ARTIFACT_ROOT/core/Llama-3.1-70B-Instruct"
CACHE_ROOT="$ARTIFACT_ROOT/generation_cache/Llama-3.1-70B-Instruct"
LOG_FILES=(
  "$ARTIFACT_ROOT/logs/run_llama70b_replica0.log"
  "$ARTIFACT_ROOT/logs/run_llama70b_replica1.log"
)
PROBE_FILES=(
  "$ARTIFACT_ROOT/logs/logprob_probe_llama31_70b_replica0.json"
  "$ARTIFACT_ROOT/logs/logprob_probe_llama31_70b_replica1.json"
)
MAX_WORKERS="${MAX_WORKERS:-4}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"

MODE="${1:-clean}"
if [[ $# -gt 1 || ( "$MODE" != "clean" && "$MODE" != "--check-only" && "$MODE" != "--resume" ) ]]; then
  echo "Usage: bash scripts/run_llama70b_main_20260723.sh [--check-only|--resume]" >&2
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
if (( MAX_WORKERS != 2 && MAX_WORKERS != 4 )); then
  echo "ERROR: MAX_WORKERS must be 2 or 4 across the two independent TP=4 services" >&2
  exit 2
fi
WORKERS_PER_REPLICA=$((MAX_WORKERS / 2))
if (( GENERATION_BATCH_SIZE < WORKERS_PER_REPLICA )); then
  echo "ERROR: GENERATION_BATCH_SIZE must be at least workers per replica" >&2
  exit 2
fi

mapfile -t service_lines < <(pgrep -af "vllm serve $MODEL_ROOT" || true)
if [[ ${#service_lines[@]} -ne 2 ]]; then
  echo "ERROR: expected exactly two independent Llama-70B services; found ${#service_lines[@]}" >&2
  printf '  %s\n' "${service_lines[@]}" >&2
  exit 2
fi
for replica in 0 1; do
  port=$((8002 + replica))
  matching=()
  for line in "${service_lines[@]}"; do
    if [[ "$line" == *"--port $port"* ]]; then
      matching+=("$line")
    fi
  done
  if [[ ${#matching[@]} -ne 1 ]]; then
    echo "ERROR: expected one Llama-70B service on port $port; found ${#matching[@]}" >&2
    exit 2
  fi
  service_command="${matching[0]}"
  for required_arg in \
    '--served-model-name meta-llama/Llama-3.1-70B-Instruct' \
    '--tensor-parallel-size 4' \
    '--distributed-executor-backend mp' \
    '--max-model-len 14336' \
    '--max-num-batched-tokens 32768' \
    '--max-num-seqs 32' \
    '--max-logprobs 10' \
    '--generation-config vllm'; do
    if [[ "$service_command" != *"$required_arg"* ]]; then
      echo "ERROR: port $port service is missing required argument: $required_arg" >&2
      echo "Observed: $service_command" >&2
      exit 2
    fi
  done
  if [[ "$service_command" == *'--data-parallel-size'* ]]; then
    echo "ERROR: port $port unexpectedly belongs to an internal-DP service" >&2
    exit 2
  fi
  service_pid="${service_command%% *}"
  if [[ ! "$service_pid" =~ ^[0-9]+$ || ! -r "/proc/$service_pid/environ" ]]; then
    echo "ERROR: cannot inspect the environment for port $port service" >&2
    exit 2
  fi
  service_env="$(tr '\0' '\n' < "/proc/$service_pid/environ")"
  for required_env in \
    'NCCL_CUMEM_ENABLE=0' \
    'NCCL_P2P_LEVEL=NVL' \
    'NCCL_IB_DISABLE=1' \
    'NCCL_SOCKET_IFNAME=lo'; do
    if ! grep -Fxq "$required_env" <<< "$service_env"; then
      echo "ERROR: port $port service is missing required environment: $required_env" >&2
      exit 2
    fi
  done
done

mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT" "$(dirname "${LOG_FILES[0]}")"
export PATH="$(dirname "$ISE_PYTHON"):$PATH"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/code${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

probe_request="$(jq -nc --arg model "$MODEL_NAME" '{
  model:$model,
  messages:[{role:"user",content:"Return the number 2."}],
  temperature:0,
  max_tokens:8,
  logprobs:true,
  top_logprobs:10
}')"

printf 'Python:       %q\n' "$ISE_PYTHON"
printf 'Dataset:      %q\n' "$DATASET_ROOT"
printf 'Model:        %q\n' "$MODEL_NAME"
printf 'Revision:     %q\n' "$MODEL_ROOT"
printf 'Endpoints:    %s %s\n' "${MODEL_URLS[@]}"
printf 'Seeds:        20260716,20260717,20260718\n'
printf 'Tasks:        GSM8K,MATH-500,HumanEvalPlus,FOLIO\n'
printf 'Concurrency:  %s total prompts; %s prompts x 16 paths per endpoint\n' \
  "$MAX_WORKERS" "$WORKERS_PER_REPLICA"
printf 'Artifact root:%q\n' "$ARTIFACT_ROOT"

for replica in 0 1; do
  endpoint="${MODEL_URLS[$replica]}"
  models_json="$(curl -fsS -H 'Authorization: Bearer empty' "$endpoint/models")"
  if ! jq -e --arg id "$MODEL_NAME" --arg root "$MODEL_ROOT" \
    '.data[] | select(.id == $id and .root == $root)' \
    >/dev/null <<< "$models_json"; then
    echo "ERROR: replica $replica model ID/root does not match the frozen revision" >&2
    printf '%s\n' "$models_json" >&2
    exit 2
  fi

  curl -fsS -H 'Authorization: Bearer empty' \
    -H 'Content-Type: application/json' \
    "$endpoint/chat/completions" \
    -d "$probe_request" | tee "${PROBE_FILES[$replica]}" >/dev/null
  if ! jq -e '.choices[0].logprobs.content[0].logprob | numbers' \
    "${PROBE_FILES[$replica]}" >/dev/null; then
    echo "ERROR: replica $replica chosen-token logprob probe failed" >&2
    exit 2
  fi

  "$ISE_PYTHON" code/run_pipeline.py \
    --mode paper \
    --task GSM8K \
    --limit 0 \
    --seed 20260716 \
    --model_url "$endpoint" \
    --model_name "$MODEL_NAME" \
    --expected_model_revision "$MODEL_ROOT" \
    --api_key empty \
    --dataset_root "$DATASET_ROOT" \
    --top_k 10 \
    --num_samples 16 \
    --ensemble_sizes 1,4,8,16 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_tokens 12000 \
    --max_workers "$WORKERS_PER_REPLICA" \
    --generation_batch_size "$GENERATION_BATCH_SIZE" \
    --request_timeout 7200 \
    --max_retries 1 \
    --cache_output_dir "$CACHE_ROOT/seed_20260716" \
    --preflight
done

if [[ "$MODE" == "--check-only" ]]; then
  echo 'Both services, logprob probes, data, model, and paper preflights passed.'
  echo 'Generation was not started.'
  exit 0
fi

common_args=(
  --mode paper
  --limit 0
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
  --max_workers "$WORKERS_PER_REPLICA"
  --generation_batch_size "$GENERATION_BATCH_SIZE"
  --request_timeout 7200
  --max_retries 1
)

for seed in 20260716 20260717 20260718; do
  seed_output="$OUTPUT_ROOT/seed_$seed"
  seed_cache="$CACHE_ROOT/seed_$seed"
  mkdir -p "$seed_output" "$seed_cache"
  for task in GSM8K MATH-500 HumanEvalPlus FOLIO; do
    mapfile -t manifests < <(
      find "$seed_output" -maxdepth 1 -type f \
        -name "manifest_${task}_*.json" -print 2>/dev/null | sort
    )
    if (( ${#manifests[@]} > 1 )); then
      echo "ERROR: expected at most one $task manifest; found ${#manifests[@]}" >&2
      exit 2
    fi
    if (( ${#manifests[@]} == 1 )); then
      if [[ "$MODE" != "--resume" ]]; then
        echo "ERROR: existing result found; use --resume: ${manifests[0]}" >&2
        exit 2
      fi
    fi
    mapfile -t caches < <(
      find "$seed_cache" -maxdepth 1 -type f \
        -name "generations_${task}_*.jsonl" -print 2>/dev/null | sort
    )
    if (( ${#caches[@]} > 1 )); then
      echo "ERROR: expected at most one $task cache; found ${#caches[@]}" >&2
      exit 2
    fi
    if (( ${#caches[@]} == 1 && ${#manifests[@]} == 0 )) && \
      [[ "$MODE" != "--resume" ]]; then
      echo "ERROR: partial cache found; use --resume: ${caches[0]}" >&2
      exit 2
    fi
  done
done

run_job() {
  local replica="$1"
  local endpoint="$2"
  local seed="$3"
  local task="$4"
  local seed_output="$OUTPUT_ROOT/seed_$seed"
  local seed_cache="$CACHE_ROOT/seed_$seed"
  local log_file="${LOG_FILES[$replica]}"
  local cache_args
  local manifests
  local caches

  mapfile -t manifests < <(
    find "$seed_output" -maxdepth 1 -type f \
      -name "manifest_${task}_*.json" -print 2>/dev/null | sort
  )
  if (( ${#manifests[@]} == 1 )); then
    echo "SKIP: replica=$replica seed=$seed task=$task already complete" | \
      tee -a "$log_file"
    return 0
  fi

  mapfile -t caches < <(
    find "$seed_cache" -maxdepth 1 -type f \
      -name "generations_${task}_*.jsonl" -print 2>/dev/null | sort
  )
  cache_args=(--cache_output_dir "$seed_cache")
  if (( ${#caches[@]} == 1 )); then
    cache_args=(--cache_path "${caches[0]}" --resume_cache)
    echo "RESUME: replica=$replica seed=$seed task=$task cache=${caches[0]}" | \
      tee -a "$log_file"
  else
    echo "START: replica=$replica seed=$seed task=$task endpoint=$endpoint" | \
      tee -a "$log_file"
  fi

  "$ISE_PYTHON" code/run_pipeline.py \
    "${common_args[@]}" \
    --model_url "$endpoint" \
    --seed "$seed" \
    --task "$task" \
    "${cache_args[@]}" \
    --preflight \
    2>&1 | tee -a "$log_file"
  "$ISE_PYTHON" code/run_pipeline.py \
    "${common_args[@]}" \
    --model_url "$endpoint" \
    --seed "$seed" \
    --task "$task" \
    "${cache_args[@]}" \
    --output_dir "$seed_output" \
    2>&1 | tee -a "$log_file"

  manifest_count="$(find "$seed_output" -maxdepth 1 -type f \
    -name "manifest_${task}_*.json" | wc -l)"
  if [[ "$manifest_count" -ne 1 ]]; then
    echo "ERROR: replica=$replica seed=$seed task=$task finished without one manifest" >&2
    return 2
  fi
}

run_queue() {
  local replica="$1"
  local endpoint="$2"
  shift 2
  local job
  local seed
  local task
  for job in "$@"; do
    IFS=: read -r seed task <<< "$job"
    run_job "$replica" "$endpoint" "$seed" "$task"
  done
}

queue0=(
  20260716:GSM8K
  20260716:HumanEvalPlus
  20260717:MATH-500
  20260717:FOLIO
  20260718:GSM8K
  20260718:HumanEvalPlus
)
queue1=(
  20260716:MATH-500
  20260716:FOLIO
  20260717:GSM8K
  20260717:HumanEvalPlus
  20260718:MATH-500
  20260718:FOLIO
)

printf 'Replica 0 queue: %s\n' "${queue0[*]}"
printf 'Replica 1 queue: %s\n' "${queue1[*]}"

run_queue 0 "${MODEL_URLS[0]}" "${queue0[@]}" &
queue0_pid=$!
run_queue 1 "${MODEL_URLS[1]}" "${queue1[@]}" &
queue1_pid=$!

set +e
wait "$queue0_pid"
queue0_status=$?
wait "$queue1_pid"
queue1_status=$?
set -e
if (( queue0_status != 0 || queue1_status != 0 )); then
  echo "ERROR: replica queues failed: replica0=$queue0_status replica1=$queue1_status" >&2
  exit 2
fi

manifest_total="$(find "$OUTPUT_ROOT" -type f -name 'manifest_*.json' | wc -l)"
if [[ "$manifest_total" -ne 12 ]]; then
  echo "ERROR: expected 12 complete manifests; found $manifest_total" >&2
  exit 2
fi

echo 'Llama-3.1-70B complete: 3 seeds x 4 tasks x 16 paths are manifest-bound.'
