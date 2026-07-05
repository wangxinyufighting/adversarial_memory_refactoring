#!/usr/bin/env bash
set -euo pipefail

# Feed golden facts F directly to the Answer Agent and keep only attacks it can answer.
#
# Hosted API:
#   CASE_GRAPH_PROVIDER=deepseek DEEPSEEK_API_KEY=... ./scripts/run_baseline_sanity_test.sh attacks.json passed.json
#
# Local API:
#   CASE_GRAPH_PROVIDER=local LOCAL_API_BASE_URL=http://localhost:8000/v1 LOCAL_MODEL=qwen ./scripts/run_baseline_sanity_test.sh attacks.json passed.json
#
# Optional env vars:
#   ANSWER_MAX_OUTPUT_TOKENS Max output tokens for the Answer Agent. Defaults to 200.
#   JUDGE_MAX_OUTPUT_TOKENS  Max output tokens for the correctness judge. Defaults to 200.
#   SKIP_LLM_JUDGE           Set to 1 to use string matching only.
#   KEEP_FAILED_BASELINE     Set to 1 to keep failed samples for debugging.

INPUT_PATH="${1:-outputs/attacks_random_min4_e47becba.json}"
OUTPUT_PATH="${2:-outputs/attacks_random_min4_e47becba_baseline_passed.json}"
ANSWER_MAX_OUTPUT_TOKENS="${ANSWER_MAX_OUTPUT_TOKENS:-200}"
JUDGE_MAX_OUTPUT_TOKENS="${JUDGE_MAX_OUTPUT_TOKENS:-200}"

ARGS=(
  --input "$INPUT_PATH"
  --output "$OUTPUT_PATH"
  --answer-max-output-tokens "$ANSWER_MAX_OUTPUT_TOKENS"
  --judge-max-output-tokens "$JUDGE_MAX_OUTPUT_TOKENS"
)

if [[ "${SKIP_LLM_JUDGE:-0}" == "1" ]]; then
  ARGS+=(--skip-llm-judge)
fi

if [[ "${KEEP_FAILED_BASELINE:-0}" == "1" ]]; then
  ARGS+=(--keep-failed)
fi

python3 -m case_graph.baseline_cli "${ARGS[@]}"
