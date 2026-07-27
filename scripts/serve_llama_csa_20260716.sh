#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VLLM_BIN="${VLLM_BIN:-/home/yhr/anaconda3/envs/vllm/bin/vllm}"
MODEL_ROOT="/data/newmodel/IE/LLM/Llama-3.1-8B-Instruct"
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
SERVER_GPUS="${CSA_SERVER_GPUS:-0,1}"
SUP_ROOT="$ROOT_DIR/.pipeline/archive/pwsc_retired_results_20260717/supplemental_experiments_20260718"
LOG_FILE="$SUP_ROOT/logs/llama31_8b_csa_logprob_safe_vllm.log"

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "ERROR: vLLM executable is missing: $VLLM_BIN" >&2
  exit 2
fi
if [[ ! -d "$MODEL_ROOT" ]]; then
  echo "ERROR: model revision is missing: $MODEL_ROOT" >&2
  exit 2
fi
if curl -fsS --max-time 2 \
  -H 'Authorization: Bearer empty' \
  http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "ERROR: port 8000 already has a model service; stop it before CSA restart." >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG_FILE")"

printf 'vLLM:       %q\n' "$VLLM_BIN"
printf 'Model:      %q\n' "$MODEL_ROOT"
printf 'GPUs:       %s\n' "$SERVER_GPUS"
printf 'Endpoint:   http://127.0.0.1:8000/v1\n'
printf 'CSA limits: gpu-memory=0.70, prefill-chunk=1024, sequences=1\n'
printf 'Log:        %q\n' "$LOG_FILE"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES="$SERVER_GPUS" \
"$VLLM_BIN" serve "$MODEL_ROOT" \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name "$MODEL_ID" \
  --api-key empty \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 24000 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  --block-size 16 \
  --max-logprobs 100 \
  --enforce-eager \
  --trust-remote-code \
  2>&1 | tee "$LOG_FILE"
