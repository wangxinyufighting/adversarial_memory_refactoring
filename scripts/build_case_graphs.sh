#!/usr/bin/env bash
set -euo pipefail

# Build UnifiedMem-style case graphs from a LongMemEval JSON file.
#
# Usage:
#   OPENAI_API_KEY=... OPENAI_MODEL=gpt-4o-mini ./scripts/build_case_graphs.sh
#   CASE_GRAPH_PROVIDER=deepseek DEEPSEEK_API_KEY=... ./scripts/build_case_graphs.sh
#   CASE_GRAPH_PROVIDER=local LOCAL_API_BASE_URL=http://localhost:8000/v1 LOCAL_MODEL=qwen ./scripts/build_case_graphs.sh
#   LIMIT=3 ./scripts/build_case_graphs.sh data/longmemeval/longmemeval_s_cleaned.json outputs/case_graphs
#
# Optional env vars:
#   OPENAI_BASE_URL       OpenAI-compatible API base URL. Defaults to https://api.openai.com/v1
#   OPENAI_MODEL          Model name. Used if CASE_GRAPH_MODEL is not set.
#   DEEPSEEK_API_KEY      DeepSeek API key.
#   DEEPSEEK_BASE_URL     DeepSeek-compatible API base URL. Defaults to https://api.deepseek.com
#   DEEPSEEK_MODEL        DeepSeek model name. Defaults to deepseek-v4-flash.
#   DEEPSEEK_THINKING     DeepSeek thinking mode. Defaults to disabled.
#   LOCAL_API_BASE_URL    Local OpenAI-compatible API base URL.
#   LOCAL_MODEL           Local model name. Used if CASE_GRAPH_MODEL is not set.
#   LOCAL_API_KEY         Optional local API key. Defaults to dummy-key.
#   CASE_GRAPH_PROVIDER   Set to openai, deepseek, or local.
#   CASE_GRAPH_MODEL      Model name specifically for graph extraction.
#   CASE_GRAPH_TIMEOUT    HTTP timeout in seconds. Defaults to 120.
#   CASE_GRAPH_MAX_INPUT_CHARS     Max session text chars sent to the LLM. Defaults to 12000.
#   CASE_GRAPH_MAX_OUTPUT_TOKENS   Max output tokens per extraction call. Defaults to 1200.
#   EXTRACTION_CACHE_PATH          Persistent extraction cache path. Defaults to OUTPUT_DIR/extraction_cache.json.
#   LIMIT                 Number of cases to process.
#   START_INDEX           One-based first case index. Defaults to 1.
#   INCLUDE_ASSISTANT=1   Include assistant turns in chunks. Default keeps user turns only.
#   OMIT_RAW_CHUNKS=1     Omit raw session text from graph JSON. Default keeps it for verification.

INPUT_PATH="${1:-data/longmemeval/longmemeval_s_cleaned.json}"
OUTPUT_DIR="${2:-outputs/case_graphs}"
PYTHON_BIN=${PYTHON_BIN:-python3}

ARGS=(--input "$INPUT_PATH" --output-dir "$OUTPUT_DIR")

if [[ -n "${START_INDEX:-}" ]]; then
  ARGS+=(--start-index "$START_INDEX")
fi

if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

if [[ "${INCLUDE_ASSISTANT:-0}" == "1" ]]; then
  ARGS+=(--include-assistant)
fi

if [[ -n "${EXTRACTION_CACHE_PATH:-}" ]]; then
  ARGS+=(--cache-path "$EXTRACTION_CACHE_PATH")
fi

if [[ "${OMIT_RAW_CHUNKS:-0}" == "1" ]]; then
  ARGS+=(--omit-raw-chunks)
fi

"${PYTHON_BIN}" -m case_graph.cli "${ARGS[@]}"
