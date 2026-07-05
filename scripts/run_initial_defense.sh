#!/usr/bin/env bash
set -euo pipefail

# Run stage-three initial defense:
# retrieve Top-K old memories, answer using only those memories, then bind Q
# to supporting memories and update the success pool when the answer is correct.
#
# Usage:
#   ./scripts/run_initial_defense.sh memory.json "Question?" "Gold answer" outputs/defense.json outputs/memory_updated.json outputs/success_pool.json 5

MEMORY_PATH="${1:?memory JSON path is required}"
QUESTION="${2:?question is required}"
ANSWER="${3:?gold answer is required}"
OUTPUT_PATH="${4:-}"
MEMORY_OUTPUT_PATH="${5:-}"
SUCCESS_POOL_PATH="${6:-}"
TOP_K="${7:-5}"

ARGS=(--memory "$MEMORY_PATH" --question "$QUESTION" --answer "$ANSWER" --top-k "$TOP_K")

if [[ -n "$OUTPUT_PATH" ]]; then
  ARGS+=(--output "$OUTPUT_PATH")
fi

if [[ -n "$MEMORY_OUTPUT_PATH" ]]; then
  ARGS+=(--memory-output "$MEMORY_OUTPUT_PATH")
fi

if [[ -n "$SUCCESS_POOL_PATH" ]]; then
  ARGS+=(--success-pool "$SUCCESS_POOL_PATH")
fi

if [[ "${SKIP_LLM_JUDGE:-0}" == "1" ]]; then
  ARGS+=(--skip-llm-judge)
fi

if [[ -n "${ANSWER_MAX_OUTPUT_TOKENS:-}" ]]; then
  ARGS+=(--answer-max-output-tokens "$ANSWER_MAX_OUTPUT_TOKENS")
fi

if [[ -n "${JUDGE_MAX_OUTPUT_TOKENS:-}" ]]; then
  ARGS+=(--judge-max-output-tokens "$JUDGE_MAX_OUTPUT_TOKENS")
fi

python3 -m case_graph.defense_cli "${ARGS[@]}"
