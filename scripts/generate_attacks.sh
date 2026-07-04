#!/usr/bin/env bash
set -euo pipefail

# Generate adversarial questions from CaseGraph JSON files.
#
# Usage:
#   CASE_GRAPH_PROVIDER=deepseek DEEPSEEK_API_KEY=... ./scripts/generate_attacks.sh outputs/case_graphs outputs/attacks.json
#
# Optional env vars:
#   POLICIES                 Comma-separated routing policies.
#   RANDOM_SEED              Seed for random walk routing. Defaults to 0.
#   RANDOM_WALK_STEPS        Maximum random walk steps. Defaults to 3.
#   RANDOM_WALK_MIN_NODES    Preferred minimum random-walk nodes.
#   RANDOM_WALK_ATTEMPTS     Attempts used to satisfy min nodes.
#   ROUTE_TOP_K              Top feature-scored routes sent to reranker. Defaults to 5.
#   ROUTE_MAX_OUTPUT_TOKENS  Max output tokens for route reranking. Defaults to 300.
#   ATTACK_MAX_OUTPUT_TOKENS Max output tokens for attacker generation. Defaults to 700.

INPUT_PATH="${1:-outputs/case_graphs}"
OUTPUT_PATH="${2:-outputs/attacks.json}"

ARGS=(--input "$INPUT_PATH" --output "$OUTPUT_PATH")

if [[ -n "${POLICIES:-}" ]]; then
  ARGS+=(--policies "$POLICIES")
fi

if [[ -n "${RANDOM_SEED:-}" ]]; then
  ARGS+=(--seed "$RANDOM_SEED")
fi

if [[ -n "${RANDOM_WALK_STEPS:-}" ]]; then
  ARGS+=(--random-walk-steps "$RANDOM_WALK_STEPS")
fi

if [[ -n "${RANDOM_WALK_MIN_NODES:-}" ]]; then
  ARGS+=(--random-walk-min-nodes "$RANDOM_WALK_MIN_NODES")
fi

if [[ -n "${RANDOM_WALK_ATTEMPTS:-}" ]]; then
  ARGS+=(--random-walk-attempts "$RANDOM_WALK_ATTEMPTS")
fi

if [[ -n "${ROUTE_TOP_K:-}" ]]; then
  ARGS+=(--route-top-k "$ROUTE_TOP_K")
fi

if [[ -n "${ROUTE_MAX_OUTPUT_TOKENS:-}" ]]; then
  ARGS+=(--route-max-output-tokens "$ROUTE_MAX_OUTPUT_TOKENS")
fi

if [[ -n "${ATTACK_MAX_OUTPUT_TOKENS:-}" ]]; then
  ARGS+=(--attack-max-output-tokens "$ATTACK_MAX_OUTPUT_TOKENS")
fi

python3 -m case_graph.attack_cli "${ARGS[@]}"
