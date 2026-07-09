#!/usr/bin/env bash
set -euo pipefail

# Online GRPO training for memory refactoring
# Generates attacks on-demand during training and maintains per-case memory states

# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/compat:${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}"

# Defaults
GRAPHS_DIR=${GRAPHS_DIR:-outputs/case_graphs_train}
MODEL_PATH=${MODEL_PATH:-/mnt/local2/wxy/models/Qwen3-0.6B}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/online_grpo}
CONFIG_FILE=${CONFIG_FILE:-configs/online_grpo.yaml}

echo "===== Online GRPO Training Configuration ====="
echo "Graphs directory: ${GRAPHS_DIR}"
echo "Model path: ${MODEL_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Config file: ${CONFIG_FILE}"
echo "Optional overrides: set env vars such as ROLLOUT_N, TRAIN_BATCH_SIZE, ATTACKER_API_BASE."
echo "=============================================="

args=(
  --graphs "${GRAPHS_DIR}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --config "${CONFIG_FILE}"
)

[[ -n "${ROLLOUT_N:-}" ]] && args+=(--rollout-n "${ROLLOUT_N}")
[[ -n "${TRAIN_BATCH_SIZE:-}" ]] && args+=(--train-batch-size "${TRAIN_BATCH_SIZE}")
[[ -n "${PPO_MINI_BATCH_SIZE:-}" ]] && args+=(--ppo-mini-batch-size "${PPO_MINI_BATCH_SIZE}")
[[ -n "${TOTAL_EPOCHS:-}" ]] && args+=(--total-epochs "${TOTAL_EPOCHS}")
[[ -n "${SAVE_FREQ:-}" ]] && args+=(--save-freq "${SAVE_FREQ}")
[[ -n "${TAU:-}" ]] && args+=(--tau "${TAU}")
[[ -n "${TOP_K:-}" ]] && args+=(--top-k "${TOP_K}")
[[ -n "${EPISODES_PER_CASE:-}" ]] && args+=(--episodes-per-case "${EPISODES_PER_CASE}")
[[ -n "${COMMIT_THRESHOLD:-}" ]] && args+=(--commit-threshold "${COMMIT_THRESHOLD}")
[[ -n "${MEMORY_TRAJECTORY_DIR:-}" ]] && args+=(--memory-trajectory-dir "${MEMORY_TRAJECTORY_DIR}")
[[ -n "${DISABLE_MEMORY_TRAJECTORY:-}" ]] && args+=(--disable-memory-trajectory)
[[ -n "${SEED:-}" ]] && args+=(--seed "${SEED}")
[[ -n "${ATTACKER_LLM:-}" ]] && args+=(--attacker-llm "${ATTACKER_LLM}")
[[ -n "${ATTACKER_API_BASE:-}" ]] && args+=(--attacker-api-base "${ATTACKER_API_BASE}")
[[ -n "${ATTACKER_API_KEY:-}" ]] && args+=(--attacker-api-key "${ATTACKER_API_KEY}")
[[ -n "${INFER_BACKEND:-}" ]] && args+=(--infer-backend "${INFER_BACKEND}")
[[ -n "${ROLLOUT_TP:-}" ]] && args+=(--rollout-tp "${ROLLOUT_TP}")
[[ -n "${ROLLOUT_GPU_MEMORY_UTILIZATION:-}" ]] && args+=(--rollout-gpu-memory-utilization "${ROLLOUT_GPU_MEMORY_UTILIZATION}")
[[ -n "${MAX_PROMPT_LENGTH:-}" ]] && args+=(--max-prompt-length "${MAX_PROMPT_LENGTH}")
[[ -n "${MAX_RESPONSE_LENGTH:-}" ]] && args+=(--max-response-length "${MAX_RESPONSE_LENGTH}")
[[ -n "${MODEL_DTYPE:-}" ]] && args+=(--model-dtype "${MODEL_DTYPE}")
[[ -n "${ROLLOUT_DTYPE:-}" ]] && args+=(--rollout-dtype "${ROLLOUT_DTYPE}")
[[ -n "${ATTN_IMPLEMENTATION:-}" ]] && args+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
[[ -n "${PROJECT_NAME:-}" ]] && args+=(--project-name "${PROJECT_NAME}")
[[ -n "${EXPERIMENT_NAME:-}" ]] && args+=(--experiment-name "${EXPERIMENT_NAME}")

python3 -m case_graph.online_memory_cli "${args[@]}" "$@"
