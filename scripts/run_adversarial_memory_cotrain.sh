#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/compat:${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}"

GRAPHS_DIR=${GRAPHS_DIR:-outputs/case_graphs_train}
ATTACKER_MODEL_PATH=${ATTACKER_MODEL_PATH:-/mnt/local2/wxy/models/Qwen3-0.6B}
DEFENDER_MODEL_PATH=${DEFENDER_MODEL_PATH:-/mnt/local2/wxy/models/Qwen3-0.6B}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/adversarial_cotrain}
CONFIG_FILE=${CONFIG_FILE:-configs/adversarial_cotrain.yaml}

args=(
  --graphs "${GRAPHS_DIR}"
  --attacker-model-path "${ATTACKER_MODEL_PATH}"
  --defender-model-path "${DEFENDER_MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --config "${CONFIG_FILE}"
)

[[ -n "${INITIAL_MEMORY_DIR:-}" ]] && args+=(--initial-memory-dir "${INITIAL_MEMORY_DIR}")
[[ -n "${COTRAIN_ROUNDS:-}" ]] && args+=(--cotrain-rounds "${COTRAIN_ROUNDS}")
[[ -n "${ATTACKER_API_BASE:-}" ]] && args+=(--attacker-api-base "${ATTACKER_API_BASE}")
[[ -n "${ATTACKER_SERVED_MODEL:-}" ]] && args+=(--attacker-served-model "${ATTACKER_SERVED_MODEL}")

python3 -m case_graph.adversarial_cotrain_cli "${args[@]}" "$@"
