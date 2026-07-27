#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ISE_PYTHON="${ISE_PYTHON:-/home/yhr/anaconda3/envs/uncertainty/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/data/newmodel/graphunagent/datasets}"
BASELINE_GPU="${BASELINE_GPU:-6}"
BASELINE_GPUS="${BASELINE_GPUS:-$BASELINE_GPU}"
SESE_NLI_BATCH_SIZE="${SESE_NLI_BATCH_SIZE:-16}"
SUP_ROOT="$ROOT_DIR/.pipeline/archive/pwsc_retired_results_20260717/supplemental_experiments_20260718"
RUN_DIR="$SUP_ROOT/core/Llama-3.1-8B-Instruct/seed_20260716"
SIDE_DIR="$SUP_ROOT/external_baselines/Llama-3.1-8B-Instruct/seed_20260716"
NLI_MODEL="$ROOT_DIR/external_models/deberta-v2-xlarge-mnli"
EMBEDDING_MODEL="$ROOT_DIR/external_models/static-similarity-mrl-multilingual-v1"
CACHE_FILE="$SIDE_DIR/sese_source.jsonl"
OUTPUT_FILE="$SIDE_DIR/sese_comparison.json"
LOG_FILE="$SUP_ROOT/logs/run_llama_sese_seed_20260716.log"
EXPECTED_NLI_REVISION="2b65bccdedcead4bc685988bcc30153d4922a789"
EXPECTED_EMBEDDING_REVISION="7264ea07c5365a11d7e6d87dbb6195889a13054f"

# CUDA_VISIBLE_DEVICES remaps the selected physical cards to cuda:0,... inside
# the process.  The evaluator uses one model runtime per logical device.
VISIBLE_GPUS="${BASELINE_GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_LIST <<< "$VISIBLE_GPUS"
SESE_DEVICES=""
for index in "${!GPU_LIST[@]}"; do
  if [[ -z "${GPU_LIST[$index]}" ]]; then
    echo "ERROR: BASELINE_GPUS contains an empty device: $BASELINE_GPUS" >&2
    exit 2
  fi
  if [[ -n "$SESE_DEVICES" ]]; then
    SESE_DEVICES+=","
  fi
  SESE_DEVICES+="cuda:${index}"
done
if [[ -z "$SESE_DEVICES" ]]; then
  echo "ERROR: BASELINE_GPUS must name at least one physical GPU" >&2
  exit 2
fi
if [[ ! "$SESE_NLI_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SESE_NLI_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi

if [[ "${1:-}" == "--check-only" ]]; then
  check_only=1
elif [[ $# -eq 0 ]]; then
  check_only=0
else
  echo "Usage: bash scripts/run_llama_sese_20260716.sh [--check-only]" >&2
  exit 2
fi

if [[ ! -x "$ISE_PYTHON" ]]; then
  echo "ERROR: uncertainty Python is not executable: $ISE_PYTHON" >&2
  exit 2
fi
for path in "$DATASET_ROOT" "$RUN_DIR" "$NLI_MODEL" "$EMBEDDING_MODEL"; do
  if [[ ! -d "$path" ]]; then
    echo "ERROR: required directory is missing: $path" >&2
    exit 2
  fi
done

for task in GSM8K MATH-500 HumanEvalPlus FOLIO; do
  manifest_count="$(find "$RUN_DIR" -maxdepth 1 -type f -name "manifest_${task}_*.json" | wc -l)"
  if [[ "$manifest_count" -ne 1 ]]; then
    echo "ERROR: expected exactly one $task manifest in $RUN_DIR; found $manifest_count" >&2
    exit 2
  fi
done

nli_revision="$(jq -r '.sha // empty' "$NLI_MODEL/.hfd/repo_metadata.json")"
embedding_revision="$(jq -r '.sha // empty' "$EMBEDDING_MODEL/.hfd/repo_metadata.json")"
if [[ "$nli_revision" != "$EXPECTED_NLI_REVISION" ]]; then
  echo "ERROR: SeSE NLI revision mismatch: $nli_revision" >&2
  exit 2
fi
if [[ "$embedding_revision" != "$EXPECTED_EMBEDDING_REVISION" ]]; then
  echo "ERROR: SeSE embedding revision mismatch: $embedding_revision" >&2
  exit 2
fi

mkdir -p "$SIDE_DIR" "$(dirname "$LOG_FILE")"
unset TQDM_DISABLE

printf 'Python:     %q\n' "$ISE_PYTHON"
printf 'Run dir:    %q\n' "$RUN_DIR"
printf 'GPUs:       physical %s -> %s\n' "$VISIBLE_GPUS" "$SESE_DEVICES"
printf 'NLI batch:  %s\n' "$SESE_NLI_BATCH_SIZE"
printf 'NLI model:  %q @ %s\n' "$NLI_MODEL" "$nli_revision"
printf 'Embedding:  %q @ %s\n' "$EMBEDDING_MODEL" "$embedding_revision"
printf 'Cache:      %q\n' "$CACHE_FILE"
printf 'Output:     %q\n' "$OUTPUT_FILE"

if [[ -s "$CACHE_FILE" ]]; then
  cache_lines="$(wc -l < "$CACHE_FILE")"
  printf 'Resume:     header + %s cached records\n' "$((cache_lines - 1))"
else
  echo 'Resume:     new cache'
fi

if (( check_only )); then
  echo "Checks passed; SeSE replay was not started."
  exit 0
fi

TORCH_WARM_POOL=0 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
"$ISE_PYTHON" scripts/evaluate_sese_2026.py \
  --run_dir "$RUN_DIR" \
  --dataset_root "$DATASET_ROOT" \
  --tasks 'GSM8K MATH-500 HumanEvalPlus FOLIO' \
  --ensemble_size 16 \
  --nli_model "$NLI_MODEL" \
  --embedding_model "$EMBEDDING_MODEL" \
  --device cuda:0 \
  --devices "$SESE_DEVICES" \
  --nli_batch_size "$SESE_NLI_BATCH_SIZE" \
  --seed 20260716 \
  --bootstrap 2000 \
  --evaluation_status development_blackbox_baseline_comparison \
  --cache "$CACHE_FILE" \
  --output "$OUTPUT_FILE" \
  2>&1 | tee -a "$LOG_FILE"
