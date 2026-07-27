#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VLLM_BIN="${VLLM_BIN:-/home/yhr/anaconda3/envs/vllm/bin/vllm}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/yhr/anaconda3/envs/vllm/bin/torchrun}"
MODEL_ROOT="/data/newmodel/IE/LLM/Llama-3.1-70B-Instruct"
MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
NCCL_PROBE_SCRIPT="$ROOT_DIR/scripts/check_nccl_tp4.py"
PORTS=(8002 8003)
SUP_ROOT="$ROOT_DIR/.pipeline/archive/pwsc_retired_results_20260717/supplemental_experiments_20260718"
ARTIFACT_ROOT="$SUP_ROOT/llama70b_main_20260723"
LOG_FILES=(
  "$ARTIFACT_ROOT/logs/llama31_70b_tp4_replica0_vllm.log"
  "$ARTIFACT_ROOT/logs/llama31_70b_tp4_replica1_vllm.log"
)

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--check-only" ) ]]; then
  echo "Usage: bash scripts/serve_llama70b_8gpu_20260723.sh [--check-only]" >&2
  exit 2
fi
CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
fi

for required in "$VLLM_BIN" "$TORCHRUN_BIN" "$NCCL_PROBE_SCRIPT" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer_config.json" \
  "$MODEL_ROOT/model.safetensors.index.json"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required file is missing: $required" >&2
    exit 2
  fi
done

shard_count="$(find "$MODEL_ROOT" -maxdepth 1 -type f \
  -name 'model-*.safetensors' | wc -l)"
if [[ "$shard_count" -ne 30 ]]; then
  echo "ERROR: expected 30 safetensors shards; found $shard_count" >&2
  exit 2
fi
weight_bytes="$(jq -r '.metadata.total_size // 0' \
  "$MODEL_ROOT/model.safetensors.index.json")"
if [[ "$weight_bytes" -lt 140000000000 ]]; then
  echo "ERROR: model index reports an implausible weight size: $weight_bytes" >&2
  exit 2
fi

for port in "${PORTS[@]}"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN | tail -n +2 | rg -q .; then
    echo "ERROR: port $port is already in use" >&2
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >&2
    exit 2
  fi
done

gpu_rows="$(nvidia-smi \
  --query-gpu=index,name,memory.total,memory.free \
  --format=csv,noheader,nounits)"
gpu_count="$(printf '%s\n' "$gpu_rows" | sed '/^[[:space:]]*$/d' | wc -l)"
if [[ "$gpu_count" -ne 8 ]]; then
  echo "ERROR: two independent TP=4 replicas require exactly eight visible GPUs; found $gpu_count" >&2
  exit 2
fi
while IFS=',' read -r index name total free; do
  index="${index//[[:space:]]/}"
  total="${total//[[:space:]]/}"
  free="${free//[[:space:]]/}"
  if [[ "$total" -lt 80000 || "$free" -lt 76000 ]]; then
    echo "ERROR: GPU $index lacks the required free memory: $free/$total MiB" >&2
    exit 2
  fi
done <<< "$gpu_rows"

compute_rows="$(nvidia-smi \
  --query-compute-apps=pid,process_name,gpu_uuid,used_memory \
  --format=csv,noheader 2>/dev/null || true)"
if [[ -n "${compute_rows//[[:space:]]/}" ]]; then
  echo "ERROR: existing GPU compute processes must be released before two TP=4 replicas start:" >&2
  printf '%s\n' "$compute_rows" >&2
  exit 2
fi

mkdir -p "$(dirname "${LOG_FILES[0]}")"
printf 'Model:        %q\n' "$MODEL_NAME"
printf 'Revision:     %q\n' "$MODEL_ROOT"
printf 'Weights:      %s bytes in %s shards\n' "$weight_bytes" "$shard_count"
printf 'Endpoints:    http://127.0.0.1:8002/v1 and http://127.0.0.1:8003/v1\n'
printf 'Parallelism:  two independent TP=4 services; no eight-rank NCCL world\n'
printf 'Topology:     replica 0 uses GPU 0--3; replica 1 uses GPU 4--7\n'
printf 'Context:      14336 tokens (audited maximum requirement: 13022)\n'
printf 'Client limit: two concurrent prompts x 16 paths per endpoint\n'
printf 'Logs:         %q %q\n' "${LOG_FILES[@]}"
printf '%s\n' "$gpu_rows"

if (( CHECK_ONLY )); then
  echo 'Hardware, model, process, and port checks passed; service was not started.'
  exit 0
fi

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

printf 'NCCL route:   P2P on NVLink only; SHM between GPU pairs; socket bootstrap on lo\n'

replica_pids=()

run_collective_probe() {
  local replica="$1"
  local devices="$2"

  echo "Checking NCCL collectives for replica $replica on GPU $devices ..."
  if ! CUDA_VISIBLE_DEVICES="$devices" timeout 60s \
    "$TORCHRUN_BIN" --standalone --nproc-per-node=4 \
    "$NCCL_PROBE_SCRIPT"; then
    echo "ERROR: NCCL collective probe failed for replica $replica (GPU $devices)" >&2
    exit 2
  fi
}

start_replica() {
  local replica="$1"
  local devices="$2"
  local port="$3"
  local log_file="$4"

  CUDA_VISIBLE_DEVICES="$devices" "$VLLM_BIN" serve "$MODEL_ROOT" \
    --host 127.0.0.1 --port "$port" \
    --served-model-name "$MODEL_NAME" \
    --api-key empty \
    --tensor-parallel-size 4 \
    --distributed-executor-backend mp \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 14336 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 32 \
    --block-size 16 \
    --max-logprobs 10 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code \
    --generation-config vllm \
    > >(sed -u "s/^/[replica-${replica}] /" | tee "$log_file") 2>&1 &
  replica_pids+=("$!")
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${replica_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${replica_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  return "$status"
}
trap cleanup EXIT INT TERM

run_collective_probe 0 0,1,2,3
run_collective_probe 1 4,5,6,7
start_replica 0 0,1,2,3 8002 "${LOG_FILES[0]}"
start_replica 1 4,5,6,7 8003 "${LOG_FILES[1]}"
printf 'Started replica 0 pid=%s and replica 1 pid=%s\n' \
  "${replica_pids[0]}" "${replica_pids[1]}"

set +e
wait -n "${replica_pids[@]}"
status=$?
set -e
echo "One Llama-70B replica exited with status $status; stopping its peer." >&2
exit "$status"
