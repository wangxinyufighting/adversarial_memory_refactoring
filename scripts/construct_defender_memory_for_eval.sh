#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/compat:${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}"
PYTHON_BIN=${PYTHON_BIN:-python}

GRAPHS=${GRAPHS:-outputs/case_graphs_test}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/eval_memory_construction/defender}
DEFENDER_API_BASE=${DEFENDER_API_BASE:-http://localhost:8004/v1}
DEFENDER_SERVED_MODEL=${DEFENDER_SERVED_MODEL:-defender-current}
ATTACKER_MODE=${ATTACKER_MODE:-coverage}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

args=(
  --graphs "${GRAPHS}"
  --output-dir "${OUTPUT_DIR}"
  --defender-api-base "${DEFENDER_API_BASE}"
  --defender-served-model "${DEFENDER_SERVED_MODEL}"
)

[[ -n "${INITIAL_MEMORY_DIR:-}" ]] && args+=(--initial-memory-dir "${INITIAL_MEMORY_DIR}")
[[ -n "${TRAINING_CONFIG:-}" ]] && args+=(--training-config "${TRAINING_CONFIG}")
[[ -n "${DEFENDER_API_KEY:-}" ]] && args+=(--defender-api-key "${DEFENDER_API_KEY}")
[[ -n "${DEFENDER_TIMEOUT:-}" ]] && args+=(--defender-timeout "${DEFENDER_TIMEOUT}")
[[ -n "${DEFENDER_MAX_OUTPUT_TOKENS:-}" ]] && args+=(--defender-max-output-tokens "${DEFENDER_MAX_OUTPUT_TOKENS}")
[[ -n "${DEFENDER_PROPOSAL_RETRIES:-}" ]] && args+=(--defender-proposal-retries "${DEFENDER_PROPOSAL_RETRIES}")
[[ -n "${DEFENDER_CHECKPOINT:-}" ]] && args+=(--defender-checkpoint "${DEFENDER_CHECKPOINT}")
is_true "${MANAGE_DEFENDER_SERVER:-}" && args+=(--manage-defender-server)
[[ -n "${DEFENDER_SERVER_PORT:-}" ]] && args+=(--defender-server-port "${DEFENDER_SERVER_PORT}")
[[ -n "${DEFENDER_SERVER_HOST:-}" ]] && args+=(--defender-server-host "${DEFENDER_SERVER_HOST}")
[[ -n "${DEFENDER_SERVER_DTYPE:-}" ]] && args+=(--defender-server-dtype "${DEFENDER_SERVER_DTYPE}")
[[ -n "${DEFENDER_SERVER_TP:-}" ]] && args+=(--defender-server-tp "${DEFENDER_SERVER_TP}")
[[ -n "${DEFENDER_SERVER_GPU_MEMORY_UTILIZATION:-}" ]] && args+=(--defender-server-gpu-memory-utilization "${DEFENDER_SERVER_GPU_MEMORY_UTILIZATION}")
[[ -n "${DEFENDER_SERVER_MAX_MODEL_LEN:-}" ]] && args+=(--defender-server-max-model-len "${DEFENDER_SERVER_MAX_MODEL_LEN}")
if [[ -n "${DEFENDER_SERVER_API_KEY:-}" ]]; then
  args+=(--defender-server-api-key "${DEFENDER_SERVER_API_KEY}")
elif is_true "${MANAGE_DEFENDER_SERVER:-}" && [[ -n "${DEFENDER_API_KEY:-}" ]]; then
  args+=(--defender-server-api-key "${DEFENDER_API_KEY}")
fi
[[ -n "${DEFENDER_SERVER_STARTUP_TIMEOUT:-}" ]] && args+=(--defender-server-startup-timeout "${DEFENDER_SERVER_STARTUP_TIMEOUT}")
[[ -n "${DEFENDER_CHECKPOINT_BACKEND:-}" ]] && args+=(--defender-checkpoint-backend "${DEFENDER_CHECKPOINT_BACKEND}")
[[ -n "${DEFENDER_CHECKPOINT_SUBDIR:-}" ]] && args+=(--defender-checkpoint-subdir "${DEFENDER_CHECKPOINT_SUBDIR}")

[[ -n "${ATTACKER_API_BASE:-}" ]] && args+=(--attacker-api-base "${ATTACKER_API_BASE}")
[[ -n "${ATTACKER_MODEL:-}" ]] && args+=(--attacker-model "${ATTACKER_MODEL}")
[[ -n "${ATTACKER_API_KEY:-}" ]] && args+=(--attacker-api-key "${ATTACKER_API_KEY}")
[[ -n "${ATTACKER_TIMEOUT:-}" ]] && args+=(--attacker-timeout "${ATTACKER_TIMEOUT}")
[[ -n "${ATTACKER_MAX_OUTPUT_TOKENS:-}" ]] && args+=(--attacker-max-output-tokens "${ATTACKER_MAX_OUTPUT_TOKENS}")
args+=(--attacker-mode "${ATTACKER_MODE}")

[[ -n "${ANSWER_API_BASE:-}" ]] && args+=(--answer-api-base "${ANSWER_API_BASE}")
[[ -n "${ANSWER_MODEL:-}" ]] && args+=(--answer-model "${ANSWER_MODEL}")
[[ -n "${ANSWER_API_KEY:-}" ]] && args+=(--answer-api-key "${ANSWER_API_KEY}")
[[ -n "${ANSWER_TIMEOUT:-}" ]] && args+=(--answer-timeout "${ANSWER_TIMEOUT}")
[[ -n "${ANSWER_MAX_OUTPUT_TOKENS:-}" ]] && args+=(--answer-max-output-tokens "${ANSWER_MAX_OUTPUT_TOKENS}")
[[ -n "${JUDGE_API_BASE:-}" ]] && args+=(--judge-api-base "${JUDGE_API_BASE}")
[[ -n "${JUDGE_MODEL:-}" ]] && args+=(--judge-model "${JUDGE_MODEL}")
[[ -n "${JUDGE_API_KEY:-}" ]] && args+=(--judge-api-key "${JUDGE_API_KEY}")
[[ -n "${JUDGE_TIMEOUT:-}" ]] && args+=(--judge-timeout "${JUDGE_TIMEOUT}")
[[ -n "${JUDGE_MAX_OUTPUT_TOKENS:-}" ]] && args+=(--judge-max-output-tokens "${JUDGE_MAX_OUTPUT_TOKENS}")
is_true "${SKIP_LLM_JUDGE:-}" && args+=(--skip-llm-judge)

[[ -n "${EPISODES_PER_CASE:-}" ]] && args+=(--episodes-per-case "${EPISODES_PER_CASE}")
is_true "${DISABLE_DYNAMIC_QUESTION_BUDGET:-}" && args+=(--disable-dynamic-question-budget)
[[ -n "${QUESTIONS_PER_UNIT:-}" ]] && args+=(--questions-per-unit "${QUESTIONS_PER_UNIT}")
[[ -n "${HARD_MAX_QUESTIONS_PER_CASE:-}" ]] && args+=(--hard-max-questions-per-case "${HARD_MAX_QUESTIONS_PER_CASE}")
[[ -n "${MIN_QUESTIONS_PER_CASE:-}" ]] && args+=(--min-questions-per-case "${MIN_QUESTIONS_PER_CASE}")
[[ -n "${COVERAGE_THRESHOLD:-}" ]] && args+=(--coverage-threshold "${COVERAGE_THRESHOLD}")
[[ -n "${CRITICAL_COVERAGE_THRESHOLD:-}" ]] && args+=(--critical-coverage-threshold "${CRITICAL_COVERAGE_THRESHOLD}")
[[ -n "${CERTIFICATION_QUESTIONS:-}" ]] && args+=(--certification-questions "${CERTIFICATION_QUESTIONS}")
is_true "${DISABLE_ADAPTIVE_STOPPING:-}" && args+=(--disable-adaptive-stopping)
is_true "${FORCE_ADD:-}" && args+=(--force-add)
[[ -n "${PROPOSAL_COUNT:-}" ]] && args+=(--proposal-count "${PROPOSAL_COUNT}")
[[ -n "${TAU:-}" ]] && args+=(--tau "${TAU}")
[[ -n "${TOP_K:-}" ]] && args+=(--top-k "${TOP_K}")
[[ -n "${TOP_K_POINTS:-}" ]] && args+=(--top-k-points "${TOP_K_POINTS}")
[[ -n "${MIN_SCORE:-}" ]] && args+=(--min-score "${MIN_SCORE}")
[[ -n "${REGRESSION_SAMPLE_SIZE:-}" ]] && args+=(--regression-sample-size "${REGRESSION_SAMPLE_SIZE}")
[[ -n "${COMMIT_THRESHOLD:-}" ]] && args+=(--commit-threshold "${COMMIT_THRESHOLD}")
[[ -n "${RETRIEVER_TYPE:-}" ]] && args+=(--retriever-type "${RETRIEVER_TYPE}")
[[ -n "${RETRIEVER_MODEL_NAME:-}" ]] && args+=(--retriever-model-name "${RETRIEVER_MODEL_NAME}")
[[ -n "${RETRIEVER_EMBEDDING_MODEL:-}" ]] && args+=(--retriever-embedding-model "${RETRIEVER_EMBEDDING_MODEL}")
[[ -n "${RETRIEVER_RETRIEVAL_MODE:-}" ]] && args+=(--retriever-retrieval-mode "${RETRIEVER_RETRIEVAL_MODE}")
[[ -n "${RETRIEVER_DEVICE:-}" ]] && args+=(--retriever-device "${RETRIEVER_DEVICE}")
[[ -n "${RETRIEVER_CACHE_DIR:-}" ]] && args+=(--retriever-cache-dir "${RETRIEVER_CACHE_DIR}")
[[ -n "${RETRIEVER_MAX_LENGTH:-}" ]] && args+=(--retriever-max-length "${RETRIEVER_MAX_LENGTH}")
is_true "${RETRIEVER_REQUIRE_MODEL:-}" && args+=(--retriever-require-model)
[[ -n "${REWARD_MODE:-}" ]] && args+=(--reward-mode "${REWARD_MODE}")
[[ -n "${SEED:-}" ]] && args+=(--seed "${SEED}")
[[ -n "${PROGRESS_LOG_INTERVAL:-}" ]] && args+=(--progress-log-interval "${PROGRESS_LOG_INTERVAL}")
[[ -n "${ROUTING_MAX_STEPS:-}" ]] && args+=(--routing-max-steps "${ROUTING_MAX_STEPS}")
[[ -n "${ROUTING_MIN_NODES:-}" ]] && args+=(--routing-min-nodes "${ROUTING_MIN_NODES}")
[[ -n "${ROUTING_ATTEMPTS:-}" ]] && args+=(--routing-attempts "${ROUTING_ATTEMPTS}")
[[ -n "${MAX_ATTACK_FAILURES:-}" ]] && args+=(--max-attack-failures "${MAX_ATTACK_FAILURES}")
[[ -n "${MAX_CONSECUTIVE_PROPOSAL_FAILURES:-}" ]] && args+=(--max-consecutive-proposal-failures "${MAX_CONSECUTIVE_PROPOSAL_FAILURES}")
[[ -n "${MAX_RETRIES_PER_UNIT:-}" ]] && args+=(--max-retries-per-unit "${MAX_RETRIES_PER_UNIT}")
is_true "${DISABLE_COMPOSITIONAL_PROBES:-}" && args+=(--disable-compositional-probes)
[[ -n "${TRACE_DETAIL:-}" ]] && args+=(--trace-detail "${TRACE_DETAIL}")
[[ -n "${CASE_WORKERS:-}" ]] && args+=(--case-workers "${CASE_WORKERS}")
[[ -n "${EXP_NAME:-}" ]] && args+=(--exp-name "${EXP_NAME}")
is_true "${NO_TRACES:-}" && args+=(--no-traces)

"${PYTHON_BIN}" -m Evaluation.construct_memory_cli "${args[@]}" "$@"
