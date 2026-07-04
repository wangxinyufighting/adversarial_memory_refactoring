#!/usr/bin/env bash
set -euo pipefail

INPUT_PATH="${1:-outputs/case_graphs_deepseek_test}"
OUTPUT_PATH="${2:-outputs/routes_deepseek_test.json}"
SELECTED_POLICY="${3:-${POLICY:-}}"
POLICIES="${SELECTED_POLICY:-${POLICIES:-random_walk,heuristic,feature_scored_llm_rerank}}"
RANDOM_SEED="${RANDOM_SEED:-0}"
RANDOM_WALK_STEPS="${RANDOM_WALK_STEPS:-3}"
RANDOM_WALK_MIN_NODES="${RANDOM_WALK_MIN_NODES:-1}"
RANDOM_WALK_ATTEMPTS="${RANDOM_WALK_ATTEMPTS:-8}"
ROUTE_TOP_K="${ROUTE_TOP_K:-5}"
ROUTE_MAX_OUTPUT_TOKENS="${ROUTE_MAX_OUTPUT_TOKENS:-300}"

args=(
  --input "$INPUT_PATH"
  --output "$OUTPUT_PATH"
  --policies "$POLICIES"
  --seed "$RANDOM_SEED"
  --random-walk-steps "$RANDOM_WALK_STEPS"
  --random-walk-min-nodes "$RANDOM_WALK_MIN_NODES"
  --random-walk-attempts "$RANDOM_WALK_ATTEMPTS"
  --route-top-k "$ROUTE_TOP_K"
  --route-max-output-tokens "$ROUTE_MAX_OUTPUT_TOKENS"
)

if [[ "${USE_LLM_RERANK:-0}" == "1" ]]; then
  args+=(--use-llm-rerank)
fi

python3 -m case_graph.route_cli "${args[@]}"
