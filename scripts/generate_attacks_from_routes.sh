#!/usr/bin/env bash
set -euo pipefail

# Generate Q/A from route JSON and attach verification-only F.
#
# Hosted API:
#   CASE_GRAPH_PROVIDER=deepseek DEEPSEEK_API_KEY=... ./scripts/generate_attacks_from_routes.sh routes.json attacks.json
#
# Local API:
#   CASE_GRAPH_PROVIDER=local LOCAL_API_BASE_URL=http://localhost:8000/v1 LOCAL_MODEL=qwen ./scripts/generate_attacks_from_routes.sh routes.json attacks.json
#
# Optional env vars:
#   ATTACK_MAX_OUTPUT_TOKENS Max output tokens for attacker generation. Defaults to 700.
#   VERIFY_MAX_OUTPUT_TOKENS Max output tokens for answer verification. Defaults to 300.
#   SKIP_ATTACK_VERIFICATION Set to 1 to skip answer verification.
#   KEEP_FAILED_VERIFICATION Set to 1 to keep unsupported or ambiguous attacks.

INPUT_PATH="${1:-outputs/routes_random_min4_test.json}"
OUTPUT_PATH="${2:-outputs/attacks_from_routes.json}"
ATTACK_MAX_OUTPUT_TOKENS="${ATTACK_MAX_OUTPUT_TOKENS:-700}"
VERIFY_MAX_OUTPUT_TOKENS="${VERIFY_MAX_OUTPUT_TOKENS:-300}"

ARGS=(
  --input "$INPUT_PATH"
  --output "$OUTPUT_PATH"
  --attack-max-output-tokens "$ATTACK_MAX_OUTPUT_TOKENS"
  --verify-max-output-tokens "$VERIFY_MAX_OUTPUT_TOKENS"
)

if [[ "${SKIP_ATTACK_VERIFICATION:-0}" == "1" ]]; then
  ARGS+=(--skip-verification)
fi

if [[ "${KEEP_FAILED_VERIFICATION:-0}" == "1" ]]; then
  ARGS+=(--keep-failed-verification)
fi

python3 -m case_graph.attack_routes_cli "${ARGS[@]}"
