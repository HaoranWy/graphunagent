#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ISE_PYTHON="${ISE_PYTHON:-/home/yhr/anaconda3/envs/uncertainty/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/data/newmodel/graphunagent/datasets}"
SUP_ROOT="$ROOT_DIR/.pipeline/archive/pwsc_retired_results_20260717/supplemental_experiments_20260718"
RUN_DIR="$SUP_ROOT/core/Llama-3.1-8B-Instruct/seed_20260716"
SIDE_DIR="$SUP_ROOT/external_baselines/Llama-3.1-8B-Instruct/seed_20260716"
NLI_CACHE="$SIDE_DIR/csa_nli.jsonl"
ASSESSMENT_CACHE="$SIDE_DIR/csa_assessment.jsonl"
OUTPUT_FILE="$SIDE_DIR/csa_comparison.json"
LOG_FILE="$SUP_ROOT/logs/run_llama_csa_seed_20260716_resume.log"
MODEL_URL="http://127.0.0.1:8000/v1"
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MODEL_ROOT="/data/newmodel/IE/LLM/Llama-3.1-8B-Instruct"
BASELINE_GPU="${BASELINE_GPU:-6}"

if [[ "${1:-}" == "--check-only" ]]; then
  check_only=1
elif [[ $# -eq 0 ]]; then
  check_only=0
else
  echo "Usage: bash scripts/resume_llama_csa_20260716.sh [--check-only]" >&2
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
if [[ ! -d "$MODEL_ROOT" ]]; then
  echo "ERROR: expected model revision is missing: $MODEL_ROOT" >&2
  exit 2
fi
if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: fixed supplemental run directory is missing: $RUN_DIR" >&2
  exit 2
fi

for task in GSM8K MATH-500 HumanEvalPlus FOLIO; do
  manifest_count="$(find "$RUN_DIR" -maxdepth 1 -type f -name "manifest_${task}_*.json" | wc -l)"
  if [[ "$manifest_count" -ne 1 ]]; then
    echo "ERROR: expected exactly one $task manifest in $RUN_DIR; found $manifest_count" >&2
    exit 2
  fi
done

if [[ ! -s "$NLI_CACHE" || ! -s "$ASSESSMENT_CACHE" ]]; then
  echo "ERROR: CSA caches are missing or empty under $SIDE_DIR" >&2
  exit 2
fi

nli_lines="$(wc -l < "$NLI_CACHE")"
assessment_lines="$(wc -l < "$ASSESSMENT_CACHE")"
if [[ "$nli_lines" -ne 2188 ]]; then
  echo "ERROR: invalid NLI cache line count: $nli_lines (expected 2188)" >&2
  exit 2
fi
if (( assessment_lines < 2 || assessment_lines > 2188 )); then
  echo "ERROR: invalid assessment cache line count: $assessment_lines (expected 2..2188)" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG_FILE")"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/code${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

printf 'Python:     %q\n' "$ISE_PYTHON"
printf 'Run dir:    %q\n' "$RUN_DIR"
printf 'NLI cache:  %q (%s/2187 records)\n' "$NLI_CACHE" "$((nli_lines - 1))"
printf 'Assessment: %q (%s/2187 records)\n' "$ASSESSMENT_CACHE" "$((assessment_lines - 1))"
printf 'Output:     %q\n' "$OUTPUT_FILE"
printf 'Endpoint:   %s\n' "$MODEL_URL"

model_response="$(
  curl -fsS --max-time 30 \
    -H 'Authorization: Bearer empty' \
    "$MODEL_URL/models"
)" || {
  echo "ERROR: Llama CSA-safe service is not reachable at $MODEL_URL" >&2
  exit 2
}
printf '%s\n' "$model_response" | "$ISE_PYTHON" -m json.tool >/dev/null

if (( check_only )); then
  echo "Checks passed; CSA assessment was not started."
  exit 0
fi

TORCH_WARM_POOL=0 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
CUDA_VISIBLE_DEVICES="$BASELINE_GPU" \
"$ISE_PYTHON" scripts/evaluate_csa_2026.py \
  --stage assessment \
  --run_dir "$RUN_DIR" \
  --dataset_root "$DATASET_ROOT" \
  --tasks 'GSM8K MATH-500 HumanEvalPlus FOLIO' \
  --ensemble_sizes 16 \
  --nli_model external_models/deberta-large-mnli \
  --nli_device cuda:0 \
  --nli_batch_size 8 \
  --nli_pair_chunk_size 2048 \
  --model_urls "$MODEL_URL" \
  --model_name "$MODEL_ID" \
  --expected_model_revision "$MODEL_ROOT" \
  --api_key empty \
  --assessment_batch_size 1 \
  --max_workers 1 \
  --request_timeout 1800 \
  --request_retries 0 \
  --seed 20260716 \
  --bootstrap 2000 \
  --evaluation_status development_blackbox_baseline_comparison \
  --nli_cache "$NLI_CACHE" \
  --assessment_cache "$ASSESSMENT_CACHE" \
  --output "$OUTPUT_FILE" \
  2>&1 | tee -a "$LOG_FILE"
