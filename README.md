# Adversarial Memory Refactoring

This repo currently contains a lightweight `CaseGraph` implementation inspired by UnifiedMem's graph pipeline.

## Build Case Graphs

The graph builder treats each LongMemEval entry as a case and each session as a chunk. By default, it keeps only user turns, extracts entities and relationships with UnifiedMem's graph extraction prompt through an OpenAI-compatible LLM API, and writes one compact graph JSON per case.
The CLI prints a chunk-level progress bar and stores `extraction_cache.json` so repeated runs skip sessions that were already extracted.

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"

LIMIT=1 ./scripts/build_case_graphs.sh \
  data/longmemeval/longmemeval_s_cleaned.json \
  outputs/case_graphs
```

DeepSeek-compatible mode:

```bash
export CASE_GRAPH_PROVIDER="deepseek"
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_MODEL="deepseek-v4-flash"
export DEEPSEEK_THINKING="disabled"

LIMIT=1 ./scripts/build_case_graphs.sh \
  data/longmemeval/longmemeval_s_cleaned.json \
  outputs/case_graphs_deepseek
```

Optional environment variables:

- `OPENAI_BASE_URL`: OpenAI-compatible API base URL.
- `DEEPSEEK_API_KEY`: DeepSeek API key. Used when `CASE_GRAPH_PROVIDER=deepseek`, or when no `OPENAI_API_KEY` is set.
- `DEEPSEEK_BASE_URL`: DeepSeek-compatible API base URL. Defaults to `https://api.deepseek.com`.
- `DEEPSEEK_MODEL`: DeepSeek model name. Defaults to `deepseek-v4-flash`.
- `DEEPSEEK_THINKING`: DeepSeek thinking mode. Defaults to `disabled` to avoid reasoning-token cost.
- `CASE_GRAPH_MODEL`: extraction model override.
- `CASE_GRAPH_PROVIDER=deepseek`: force DeepSeek environment variable resolution.
- `CASE_GRAPH_TIMEOUT`: HTTP timeout in seconds.
- `CASE_GRAPH_MAX_INPUT_CHARS`: maximum session text characters sent to the LLM. Defaults to `12000`.
- `CASE_GRAPH_MAX_OUTPUT_TOKENS`: `max_tokens` for each extraction response. Defaults to `1200`.
- `EXTRACTION_CACHE_PATH`: persistent extraction cache path. Defaults to `<output-dir>/extraction_cache.json`.
- `INCLUDE_ASSISTANT=1`: include assistant turns in session chunks.
- `INCLUDE_RAW_CHUNKS=1`: include raw session text in graph JSON for debugging. Default omits it to keep outputs compact.

## Output Shape

Each `*.case_graph.json` contains:

- `chunks`: compact session provenance with `chunk_id`, `timestamp`, and order. Raw `content` is omitted unless `INCLUDE_RAW_CHUNKS=1`.
- `entities`: normalized entity nodes with descriptions and `source_ids`.
- `relationships`: undirected entity edges with descriptions, accumulated weight, and `source_ids`.
- `extraction_cache.json`: persisted LLM extraction results keyed by session text and timestamp, used to avoid duplicate API calls across reruns.
# adversarial_memory_refactoring
