#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/compat:${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}"

PYTHON_BIN=${PYTHON_BIN:-python}
MEMORY_DIR=${MEMORY_DIR:-outputs/evaluation/global_step_75_retry/memory_states}
DATASET=${DATASET:-data/longmemeval/longmemeval_s_cleaned.json}
OUTPUT=${OUTPUT:-$(dirname "${MEMORY_DIR}")/memory_qa_results.json}
TRAINING_CONFIG=${TRAINING_CONFIG:-configs/online_grpo.yaml}
ANSWER_API_BASE=${ANSWER_API_BASE:-http://localhost:8003/v1}
ANSWER_MODEL=${ANSWER_MODEL:-${ATTACKER_MODEL:-/mnt/local2/wxy/models/Qwen3-0.6B}}
ANSWER_API_KEY=${ANSWER_API_KEY:-${ATTACKER_API_KEY:-dummy-key}}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

args=(
  --memory-dir "${MEMORY_DIR}"
  --dataset "${DATASET}"
  --output "${OUTPUT}"
  --training-config "${TRAINING_CONFIG}"
  --answer-api-base "${ANSWER_API_BASE}"
  --answer-model "${ANSWER_MODEL}"
  --answer-api-key "${ANSWER_API_KEY}"
)

[[ -n "${GRAPHS:-}" ]] && args+=(--graphs "${GRAPHS}")
[[ -n "${COVERAGE_DIR:-}" ]] && args+=(--coverage-dir "${COVERAGE_DIR}")
is_true "${REQUIRE_CERTIFIED_MEMORY:-}" && args+=(--require-certified-memory)
[[ -n "${CASE_ID:-}" ]] && args+=(--case-id "${CASE_ID}")
[[ -n "${START_INDEX:-}" ]] && args+=(--start-index "${START_INDEX}")
[[ -n "${MAX_CASES:-}" ]] && args+=(--max-cases "${MAX_CASES}")
is_true "${RESUME:-}" && args+=(--resume)
[[ -n "${SAVE_EVERY:-}" ]] && args+=(--save-every "${SAVE_EVERY}")

[[ -n "${TOP_K:-}" ]] && args+=(--top-k "${TOP_K}")
[[ -n "${TOP_K_POINTS:-}" ]] && args+=(--top-k-points "${TOP_K_POINTS}")
[[ -n "${MIN_SCORE:-}" ]] && args+=(--min-score "${MIN_SCORE}")
[[ -n "${RETRIEVER_TYPE:-}" ]] && args+=(--retriever-type "${RETRIEVER_TYPE}")
[[ -n "${RETRIEVER_MODEL_NAME:-}" ]] && args+=(--retriever-model-name "${RETRIEVER_MODEL_NAME}")
[[ -n "${RETRIEVER_EMBEDDING_MODEL:-}" ]] && args+=(--retriever-embedding-model "${RETRIEVER_EMBEDDING_MODEL}")
[[ -n "${RETRIEVER_RETRIEVAL_MODE:-}" ]] && args+=(--retriever-retrieval-mode "${RETRIEVER_RETRIEVAL_MODE}")
[[ -n "${RETRIEVER_DEVICE:-}" ]] && args+=(--retriever-device "${RETRIEVER_DEVICE}")
[[ -n "${RETRIEVER_CACHE_DIR:-}" ]] && args+=(--retriever-cache-dir "${RETRIEVER_CACHE_DIR}")
[[ -n "${RETRIEVER_MAX_LENGTH:-}" ]] && args+=(--retriever-max-length "${RETRIEVER_MAX_LENGTH}")
is_true "${RETRIEVER_REQUIRE_MODEL:-}" && args+=(--retriever-require-model)
is_true "${ALLOW_RETRIEVER_FALLBACK:-}" && args+=(--allow-retriever-fallback)

[[ -n "${ANSWER_TIMEOUT:-}" ]] && args+=(--answer-timeout "${ANSWER_TIMEOUT}")
[[ -n "${ANSWER_MAX_OUTPUT_TOKENS:-}" ]] && args+=(--answer-max-output-tokens "${ANSWER_MAX_OUTPUT_TOKENS}")
[[ -n "${ANSWER_THINKING:-}" ]] && args+=(--answer-thinking "${ANSWER_THINKING}")
[[ -n "${JUDGE_MODE:-}" ]] && args+=(--judge-mode "${JUDGE_MODE}")
[[ -n "${JUDGE_API_BASE:-}" ]] && args+=(--judge-api-base "${JUDGE_API_BASE}")
[[ -n "${JUDGE_MODEL:-}" ]] && args+=(--judge-model "${JUDGE_MODEL}")
[[ -n "${JUDGE_API_KEY:-}" ]] && args+=(--judge-api-key "${JUDGE_API_KEY}")
[[ -n "${JUDGE_TIMEOUT:-}" ]] && args+=(--judge-timeout "${JUDGE_TIMEOUT}")
[[ -n "${JUDGE_MAX_OUTPUT_TOKENS:-}" ]] && args+=(--judge-max-output-tokens "${JUDGE_MAX_OUTPUT_TOKENS}")
[[ -n "${JUDGE_THINKING:-}" ]] && args+=(--judge-thinking "${JUDGE_THINKING}")
is_true "${SKIP_ENDPOINT_PREFLIGHT:-}" && args+=(--skip-endpoint-preflight)

echo "===== Fixed-Memory QA Evaluation ====="
echo "Memory directory: ${MEMORY_DIR}"
echo "Dataset: ${DATASET}"
echo "Answer model: ${ANSWER_MODEL}"
echo "Answer API: ${ANSWER_API_BASE}"
echo "Retriever device: ${RETRIEVER_DEVICE:-from ${TRAINING_CONFIG}}"
echo "Output: ${OUTPUT}"
echo "======================================"

"${PYTHON_BIN}" -m Evaluation.evaluate_memory_qa_cli "${args[@]}" "$@"
