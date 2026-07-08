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

# Training hyperparameters
ROLLOUT_N=${ROLLOUT_N:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}

# Environment parameters
TAU=${TAU:-0.7}
TOP_K=${TOP_K:-5}
EPISODES_PER_CASE=${EPISODES_PER_CASE:-100}
COMMIT_THRESHOLD=${COMMIT_THRESHOLD:-0.0}
SEED=${SEED:-42}

# Frozen attacker backend for on-demand attack generation
ATTACKER_LLM=${ATTACKER_LLM:-/mnt/local2/wxy/models/Qwen3-0.6B}
ATTACKER_API_BASE=${ATTACKER_API_BASE:-http://localhost:8003/v1}
ATTACKER_API_KEY=${ATTACKER_API_KEY:-dummy-key}

# LLM backend for inference
INFER_BACKEND=${INFER_BACKEND:-vllm}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}

# Max lengths
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

# Logging
PROJECT_NAME=${PROJECT_NAME:-memory_refactor_grpo_online}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-online_training_$(date +%Y%m%d_%H%M%S)}

echo "===== Online GRPO Training Configuration ====="
echo "Graphs directory: ${GRAPHS_DIR}"
echo "Model path: ${MODEL_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Config file: ${CONFIG_FILE}"
echo "Rollout N: ${ROLLOUT_N}"
echo "Train batch size: ${TRAIN_BATCH_SIZE}"
echo "Total epochs: ${TOTAL_EPOCHS}"
echo "Episodes per case: ${EPISODES_PER_CASE}"
echo "Attacker API base: ${ATTACKER_API_BASE}"
echo "=============================================="

python3 -m case_graph.online_memory_cli \
  --graphs "${GRAPHS_DIR}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --config "${CONFIG_FILE}" \
  --rollout-n "${ROLLOUT_N}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --ppo-mini-batch-size "${PPO_MINI_BATCH_SIZE}" \
  --total-epochs "${TOTAL_EPOCHS}" \
  --tau "${TAU}" \
  --top-k "${TOP_K}" \
  --episodes-per-case "${EPISODES_PER_CASE}" \
  --commit-threshold "${COMMIT_THRESHOLD}" \
  --seed "${SEED}" \
  --attacker-llm "${ATTACKER_LLM}" \
  --attacker-api-base "${ATTACKER_API_BASE}" \
  --attacker-api-key "${ATTACKER_API_KEY}" \
  --infer-backend "${INFER_BACKEND}" \
  --rollout-tp "${ROLLOUT_TP}" \
  --rollout-gpu-memory-utilization "${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH}" \
  --max-response-length "${MAX_RESPONSE_LENGTH}" \
  --project-name "${PROJECT_NAME}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  "$@"
