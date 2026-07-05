#!/usr/bin/env bash
set -euo pipefail

# Retrieve Top-K old-memory chunks from the current memory library Mt.
#
# Usage:
#   ./scripts/retrieve_memories.sh memory.json "What degree did I graduate with?" outputs/retrieval.json 5

MEMORY_PATH="${1:?memory JSON path is required}"
QUESTION="${2:?question is required}"
OUTPUT_PATH="${3:-}"
TOP_K="${4:-5}"

ARGS=(--memory "$MEMORY_PATH" --question "$QUESTION" --top-k "$TOP_K")

if [[ -n "$OUTPUT_PATH" ]]; then
  ARGS+=(--output "$OUTPUT_PATH")
fi

python3 -m case_graph.retrieval_cli "${ARGS[@]}"
