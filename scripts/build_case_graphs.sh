#!/usr/bin/env bash
set -euo pipefail

# Build UnifiedMem-style case graphs from a LongMemEval JSON file.
#
# Usage:
#   OPENAI_API_KEY=... OPENAI_MODEL=gpt-4o-mini ./scripts/build_case_graphs.sh
#   CASE_GRAPH_PROVIDER=deepseek DEEPSEEK_API_KEY=... ./scripts/build_case_graphs.sh
#   LIMIT=3 ./scripts/build_case_graphs.sh data/longmemeval/longmemeval_s_cleaned.json outputs/case_graphs
#
# Optional env vars:
#   OPENAI_BASE_URL       OpenAI-compatible API base URL. Defaults to https://api.openai.com/v1
#   OPENAI_MODEL          Model name. Used if CASE_GRAPH_MODEL is not set.
#   DEEPSEEK_API_KEY      DeepSeek API key.
#   DEEPSEEK_BASE_URL     DeepSeek-compatible API base URL. Defaults to https://api.deepseek.com
#   DEEPSEEK_MODEL        DeepSeek model name. Defaults to deepseek-v4-flash.
#   DEEPSEEK_THINKING     DeepSeek thinking mode. Defaults to disabled.
#   CASE_GRAPH_PROVIDER   Set to deepseek to force DeepSeek env resolution.
#   CASE_GRAPH_MODEL      Model name specifically for graph extraction.
#   CASE_GRAPH_TIMEOUT    HTTP timeout in seconds. Defaults to 120.
#   CASE_GRAPH_MAX_INPUT_CHARS     Max session text chars sent to the LLM. Defaults to 12000.
#   CASE_GRAPH_MAX_OUTPUT_TOKENS   Max output tokens per extraction call. Defaults to 1200.
#   EXTRACTION_CACHE_PATH          Persistent extraction cache path. Defaults to OUTPUT_DIR/extraction_cache.json.
#   LIMIT                 Number of cases to process.
#   INCLUDE_ASSISTANT=1   Include assistant turns in chunks. Default keeps user turns only.
#   INCLUDE_RAW_CHUNKS=1  Write raw session text into graph JSON. Default omits it.

INPUT_PATH="${1:-data/longmemeval/longmemeval_s_cleaned.json}"
OUTPUT_DIR="${2:-outputs/case_graphs}"

ARGS=(--input "$INPUT_PATH" --output-dir "$OUTPUT_DIR")

if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

if [[ "${INCLUDE_ASSISTANT:-0}" == "1" ]]; then
  ARGS+=(--include-assistant)
fi

if [[ -n "${EXTRACTION_CACHE_PATH:-}" ]]; then
  ARGS+=(--cache-path "$EXTRACTION_CACHE_PATH")
fi

if [[ "${INCLUDE_RAW_CHUNKS:-0}" == "1" ]]; then
  ARGS+=(--include-raw-chunks)
fi

python3 -m case_graph.cli "${ARGS[@]}"
