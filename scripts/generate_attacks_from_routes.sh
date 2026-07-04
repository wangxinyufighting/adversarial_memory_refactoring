#!/usr/bin/env bash
set -euo pipefail

# Generate Q/F from route JSON with any OpenAI-compatible chat API.
#
# Hosted API:
#   CASE_GRAPH_PROVIDER=deepseek DEEPSEEK_API_KEY=... ./scripts/generate_attacks_from_routes.sh routes.json attacks.json
#
# Local API:
#   CASE_GRAPH_PROVIDER=local LOCAL_API_BASE_URL=http://localhost:8000/v1 LOCAL_MODEL=qwen ./scripts/generate_attacks_from_routes.sh routes.json attacks.json

INPUT_PATH="${1:-outputs/routes_random_min4_test.json}"
OUTPUT_PATH="${2:-outputs/attacks_from_routes.json}"
ATTACK_MAX_OUTPUT_TOKENS="${ATTACK_MAX_OUTPUT_TOKENS:-700}"

python3 -m case_graph.attack_routes_cli \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  --attack-max-output-tokens "$ATTACK_MAX_OUTPUT_TOKENS"
