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

args=(
  --graphs "${GRAPHS}"
  --output-dir "${OUTPUT_DIR}"
  --defender-api-base "${DEFENDER_API_BASE}"
  --defender-served-model "${DEFENDER_SERVED_MODEL}"
)

[[ -n "${INITIAL_MEMORY_DIR:-}" ]] && args+=(--initial-memory-dir "${INITIAL_MEMORY_DIR}")
[[ -n "${DEFENDER_API_KEY:-}" ]] && args+=(--defender-api-key "${DEFENDER_API_KEY}")
[[ -n "${DEFENDER_TIMEOUT:-}" ]] && args+=(--defender-timeout "${DEFENDER_TIMEOUT}")
[[ -n "${DEFENDER_MAX_OUTPUT_TOKENS:-}" ]] && args+=(--defender-max-output-tokens "${DEFENDER_MAX_OUTPUT_TOKENS}")
[[ -n "${DEFENDER_CHECKPOINT:-}" ]] && args+=(--defender-checkpoint "${DEFENDER_CHECKPOINT}")
[[ -n "${MANAGE_DEFENDER_SERVER:-}" ]] && args+=(--manage-defender-server)
[[ -n "${DEFENDER_SERVER_PORT:-}" ]] && args+=(--defender-server-port "${DEFENDER_SERVER_PORT}")
[[ -n "${DEFENDER_SERVER_HOST:-}" ]] && args+=(--defender-server-host "${DEFENDER_SERVER_HOST}")
[[ -n "${DEFENDER_SERVER_DTYPE:-}" ]] && args+=(--defender-server-dtype "${DEFENDER_SERVER_DTYPE}")
[[ -n "${DEFENDER_SERVER_TP:-}" ]] && args+=(--defender-server-tp "${DEFENDER_SERVER_TP}")
[[ -n "${DEFENDER_SERVER_GPU_MEMORY_UTILIZATION:-}" ]] && args+=(--defender-server-gpu-memory-utilization "${DEFENDER_SERVER_GPU_MEMORY_UTILIZATION}")
[[ -n "${DEFENDER_SERVER_MAX_MODEL_LEN:-}" ]] && args+=(--defender-server-max-model-len "${DEFENDER_SERVER_MAX_MODEL_LEN}")
if [[ -n "${DEFENDER_SERVER_API_KEY:-}" ]]; then
  args+=(--defender-server-api-key "${DEFENDER_SERVER_API_KEY}")
elif [[ -n "${MANAGE_DEFENDER_SERVER:-}" && -n "${DEFENDER_API_KEY:-}" ]]; then
  args+=(--defender-server-api-key "${DEFENDER_API_KEY}")
fi
[[ -n "${DEFENDER_SERVER_STARTUP_TIMEOUT:-}" ]] && args+=(--defender-server-startup-timeout "${DEFENDER_SERVER_STARTUP_TIMEOUT}")
[[ -n "${DEFENDER_CHECKPOINT_BACKEND:-}" ]] && args+=(--defender-checkpoint-backend "${DEFENDER_CHECKPOINT_BACKEND}")
[[ -n "${DEFENDER_CHECKPOINT_SUBDIR:-}" ]] && args+=(--defender-checkpoint-subdir "${DEFENDER_CHECKPOINT_SUBDIR}")

[[ -n "${ATTACKER_API_BASE:-}" ]] && args+=(--attacker-api-base "${ATTACKER_API_BASE}")
[[ -n "${ATTACKER_MODEL:-}" ]] && args+=(--attacker-model "${ATTACKER_MODEL}")
[[ -n "${ATTACKER_API_KEY:-}" ]] && args+=(--attacker-api-key "${ATTACKER_API_KEY}")
[[ -n "${ATTACKER_TIMEOUT:-}" ]] && args+=(--attacker-timeout "${ATTACKER_TIMEOUT}")
args+=(--attacker-mode "${ATTACKER_MODE}")

[[ -n "${ANSWER_API_BASE:-}" ]] && args+=(--answer-api-base "${ANSWER_API_BASE}")
[[ -n "${ANSWER_MODEL:-}" ]] && args+=(--answer-model "${ANSWER_MODEL}")
[[ -n "${ANSWER_API_KEY:-}" ]] && args+=(--answer-api-key "${ANSWER_API_KEY}")
[[ -n "${JUDGE_API_BASE:-}" ]] && args+=(--judge-api-base "${JUDGE_API_BASE}")
[[ -n "${JUDGE_MODEL:-}" ]] && args+=(--judge-model "${JUDGE_MODEL}")
[[ -n "${JUDGE_API_KEY:-}" ]] && args+=(--judge-api-key "${JUDGE_API_KEY}")
[[ -n "${SKIP_LLM_JUDGE:-}" ]] && args+=(--skip-llm-judge)

[[ -n "${EPISODES_PER_CASE:-}" ]] && args+=(--episodes-per-case "${EPISODES_PER_CASE}")
[[ -n "${PROPOSAL_COUNT:-}" ]] && args+=(--proposal-count "${PROPOSAL_COUNT}")
[[ -n "${TAU:-}" ]] && args+=(--tau "${TAU}")
[[ -n "${TOP_K:-}" ]] && args+=(--top-k "${TOP_K}")
[[ -n "${MIN_SCORE:-}" ]] && args+=(--min-score "${MIN_SCORE}")
[[ -n "${REGRESSION_SAMPLE_SIZE:-}" ]] && args+=(--regression-sample-size "${REGRESSION_SAMPLE_SIZE}")
[[ -n "${COMMIT_THRESHOLD:-}" ]] && args+=(--commit-threshold "${COMMIT_THRESHOLD}")
[[ -n "${SEED:-}" ]] && args+=(--seed "${SEED}")
[[ -n "${ROUTING_MAX_STEPS:-}" ]] && args+=(--routing-max-steps "${ROUTING_MAX_STEPS}")
[[ -n "${ROUTING_MIN_NODES:-}" ]] && args+=(--routing-min-nodes "${ROUTING_MIN_NODES}")
[[ -n "${ROUTING_ATTEMPTS:-}" ]] && args+=(--routing-attempts "${ROUTING_ATTEMPTS}")
[[ -n "${MAX_ATTACK_FAILURES:-}" ]] && args+=(--max-attack-failures "${MAX_ATTACK_FAILURES}")
[[ -n "${EXP_NAME:-}" ]] && args+=(--exp-name "${EXP_NAME}")
[[ -n "${NO_TRACES:-}" ]] && args+=(--no-traces)

"${PYTHON_BIN}" -m Evaluation.construct_memory_cli "${args[@]}" "$@"
