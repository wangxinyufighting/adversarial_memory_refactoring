# Adversarial Memory Refactoring

This repo currently contains a lightweight `CaseGraph` implementation inspired by UnifiedMem's graph pipeline.

## Build Case Graphs

The graph builder treats each LongMemEval entry as a case and each session as a chunk. By default, it keeps only user turns, extracts entities and relationships with UnifiedMem's graph extraction prompt through an OpenAI-compatible LLM API, and writes one graph JSON per case. Raw chunk text is kept by default so later stages can attach verification-only gold sessions.
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

Local OpenAI-compatible mode:

```bash
export CASE_GRAPH_PROVIDER="local"
export LOCAL_API_BASE_URL="http://localhost:8000/v1"
export LOCAL_MODEL="qwen-local"

LIMIT=1 ./scripts/build_case_graphs.sh \
  data/longmemeval/longmemeval_s_cleaned.json \
  outputs/case_graphs_local
```

Optional environment variables:

- `OPENAI_BASE_URL`: OpenAI-compatible API base URL.
- `OPENAI_API_KEY`: hosted OpenAI-compatible API key.
- `DEEPSEEK_API_KEY`: DeepSeek API key. Used when `CASE_GRAPH_PROVIDER=deepseek`, or when no `OPENAI_API_KEY` is set.
- `DEEPSEEK_BASE_URL`: DeepSeek-compatible API base URL. Defaults to `https://api.deepseek.com`.
- `DEEPSEEK_MODEL`: DeepSeek model name. Defaults to `deepseek-v4-flash`.
- `DEEPSEEK_THINKING`: DeepSeek thinking mode. Defaults to `disabled` to avoid reasoning-token cost.
- `LOCAL_API_BASE_URL`: local OpenAI-compatible API base URL, such as `http://localhost:8000/v1`.
- `LOCAL_MODEL`: local model name. Used if `CASE_GRAPH_MODEL` is not set.
- `LOCAL_API_KEY`: optional local API key. Defaults to `dummy-key` when omitted.
- `CASE_GRAPH_MODEL`: extraction model override.
- `CASE_GRAPH_PROVIDER`: set to `openai`, `deepseek`, or `local`.
- `CASE_GRAPH_TIMEOUT`: HTTP timeout in seconds.
- `CASE_GRAPH_MAX_INPUT_CHARS`: maximum session text characters sent to the LLM. Defaults to `12000`.
- `CASE_GRAPH_MAX_OUTPUT_TOKENS`: `max_tokens` for each extraction response. Defaults to `1200`.
- `EXTRACTION_CACHE_PATH`: persistent extraction cache path. Defaults to `<output-dir>/extraction_cache.json`.
- `INCLUDE_ASSISTANT=1`: include assistant turns in session chunks.
- `OMIT_RAW_CHUNKS=1`: omit raw session text from graph JSON. Default keeps it for verification evidence.

## Output Shape

Each `*.case_graph.json` contains:

- `target`: target question, answer, and answer-session provenance from LongMemEval.
- `chunks`: session provenance with `chunk_id`, `timestamp`, order, and raw `content` by default.
- `entities`: normalized entity nodes with descriptions and `source_ids`.
- `relationships`: directed entity edges with descriptions, accumulated weight, and `source_ids`.
- `extraction_cache.json`: persisted LLM extraction results keyed by session text and timestamp, used to avoid duplicate API calls across reruns.

The graph keeps `USER` as a compact anchor node instead of concatenating every session-level user summary into one very long description. This keeps downstream graph routing prompts smaller while preserving provenance through `source_ids` and directed relationships.
When target answer text is missing from the extracted entities/relationships, the builder adds a deterministic evaluator-safeguard entity for offline coverage checks. Routing policies filter this evaluator-injected edge and never use the private `target` fields for route selection.

## Generate Routes Only

Use this when you only want to inspect the attack paths before they are sent to the attacker. Each route item contains public route evidence, public entity descriptions for the route nodes, plus verification-only `golden_facts` resolved from the raw sessions touched by the route. The attacker receives only the public `route` field, not `golden_facts`. A route may incidentally contain the gold answer if that fact was naturally extracted into the graph, but routing does not read the private `target` fields. By default, this does not call the attacker and does not call the LLM reranker; `feature_scored_llm_rerank` falls back to feature-score selection.

```bash
./scripts/generate_routes.sh \
  outputs/case_graphs_deepseek_test \
  outputs/routes_deepseek_test.json
```

Select a single routing policy:

```bash
./scripts/generate_routes.sh \
  outputs/case_graphs_deepseek_test \
  outputs/routes_heuristic_test.json \
  heuristic
```

Prefer at least four nodes for `random_walk` when the graph can support it:

```bash
RANDOM_WALK_MIN_NODES=4 RANDOM_WALK_ATTEMPTS=16 ./scripts/generate_routes.sh \
  outputs/case_graphs_deepseek_test \
  outputs/routes_random_test.json \
  random_walk
```

Set `USE_LLM_RERANK=1` if you want the feature-scored candidates reranked by the configured frozen LLM.

## Generate Attacks

Attack generation uses frozen LLM calls. The graph routing policy first chooses an attack path, then the attacker receives only public route evidence and generates a tricky question `Q` plus a candidate answer `A`. Gold facts `F` are the relevant raw sessions resolved from the route and are copied into the output for verification. A separate verifier checks whether `A` is supported by `F`. The original target question and answer metadata remain private evaluation fields and are not sent to the attacker or routing policy.

Recommended two-stage workflow:

```bash
./scripts/generate_routes.sh \
  outputs/case_graphs_deepseek_test \
  outputs/routes_random_min4_test.json \
  random_walk

export CASE_GRAPH_PROVIDER="deepseek"
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_MODEL="deepseek-v4-flash"
export DEEPSEEK_THINKING="disabled"

./scripts/generate_attacks_from_routes.sh \
  outputs/routes_random_min4_test.json \
  outputs/attacks_random_min4_test.json
```

Using a local OpenAI-compatible model:

```bash
export CASE_GRAPH_PROVIDER="local"
export LOCAL_API_BASE_URL="http://localhost:8000/v1"
export LOCAL_MODEL="qwen-local"

./scripts/generate_attacks_from_routes.sh \
  outputs/routes_random_min4_test.json \
  outputs/attacks_random_min4_local.json
```

The attacker prompt asks for uniquely grounded questions and requires qualifiers such as sport, league, city, event, role, or full name when a vague phrase like "the player" or "the team" could be ambiguous.

The attack output contains `question`, `answer`, verification-only raw-session `golden_facts`, and the public `route` used to generate the question.
By default it also contains `verification`, with `supported`, `unambiguous`, `temporal_supported`, `evidence_session_ids`, and `reason`. Attacks with failed verification are filtered out by default. Set `KEEP_FAILED_VERIFICATION=1` to keep failed samples for debugging, or `SKIP_ATTACK_VERIFICATION=1` for cheaper debugging runs.

You can also run routing and attacking in one command from CaseGraph files:

```bash
export CASE_GRAPH_PROVIDER="deepseek"
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_MODEL="deepseek-v4-flash"
export DEEPSEEK_THINKING="disabled"

./scripts/generate_attacks.sh \
  outputs/case_graphs_deepseek_test \
  outputs/attacks_deepseek_test.json
```

## Baseline Sanity Test

Before defense testing, feed each attack's verification-only raw sessions `F` directly to an Answer Agent and ask it to answer `Q`. If the Answer Agent cannot recover the attack's `answer` from `F`, the sample is discarded because even a perfect memory context is insufficient.

```bash
CASE_GRAPH_PROVIDER=deepseek \
DEEPSEEK_API_KEY=... \
DEEPSEEK_MODEL=deepseek-v4-flash \
DEEPSEEK_THINKING=disabled \
ANSWER_MAX_OUTPUT_TOKENS=200 \
JUDGE_MAX_OUTPUT_TOKENS=200 \
./scripts/run_baseline_sanity_test.sh \
  outputs/attacks_random_min4_test.json \
  outputs/attacks_random_min4_baseline_passed.json
```

The output keeps only baseline-passable attacks by default and adds `baseline_sanity` to each retained sample. The correctness judge first uses normalized string matching and only calls the LLM judge when the match is uncertain. Set `KEEP_FAILED_BASELINE=1` to keep failed samples for inspection, or `SKIP_LLM_JUDGE=1` to avoid the extra judge call.

Routing policies are target-agnostic:

- `random_walk`: seeded random walk from `USER` when available.
- `heuristic`: highest-weight user-centered evidence edge, falling back to the highest-weight graph edge.
- `feature_scored_llm_rerank`: deterministic graph-evidence candidates, target-agnostic feature scoring, then optional frozen LLM reranking over public route evidence.

Use `POLICIES=random_walk,heuristic` to run a subset.

## Old Memory Retrieval

Stage-three defense uses a frozen BM25-style retriever over the current memory library `M_t`. Each memory item is a structured chunk with `content` and `linked_questions`; retrieval never updates model parameters.

```bash
./scripts/retrieve_memories.sh \
  data/memory_store.json \
  "What degree did I graduate with?" \
  outputs/retrieval.json \
  5
```

Accepted memory JSON shapes are either `{"memories": [...]}` or `{"chunks": [...]}`. Each item may use `memory_id`, `id`, `chunk_id`, or `session_id` as its identifier.

## Initial Defense Test

After retrieval, the Answer Agent answers using only the retrieved old memories. If the judge marks the answer correct, the question is bound to the supporting memories' `linked_questions` and added to the success pool. Otherwise, the output is marked `needs_refactor=true` for the memory refactoring stage.

```bash
CASE_GRAPH_PROVIDER=deepseek \
DEEPSEEK_API_KEY=... \
./scripts/run_initial_defense.sh \
  data/memory_store.json \
  "Which baseball player did the user admire after watching the Astros?" \
  "Jose Altuve" \
  outputs/initial_defense.json \
  outputs/memory_store_updated.json \
  outputs/success_pool.json \
  5
```

## Similarity Routing

Stage-four routing first retrieves Top-K old memories, then compares the highest BM25-style similarity score with `tau`. If `max_score < tau`, the router chooses `add`; otherwise it chooses `merge` and records the hit memory ids whose scores meet the threshold. This step only selects the action and never mutates `M_t`.

```bash
python3 -m case_graph.refactoring_cli \
  --memory data/memory_store.json \
  --question "Which Astros player did the user admire?" \
  --tau 1.0 \
  --top-k 5 \
  --output outputs/refactor_decision.json
```

## Sandbox Refactoring Test

The sandbox stage applies a proposed `add` or `merge` edit on a deep copy of `M_t`, producing `M_temp`. It then answers the current question and selected regression questions from `M_temp`, and computes reward from current correctness, regression accuracy, regression failures, new chunk count, and new chunk length. The main memory is not changed here; commit/rollback is handled by the next stage.

For GRPO, multiple rollouts produce multiple sandbox results. Environment settlement selects the rollout with the highest reward: commit its `M_temp` only when the best reward is positive; otherwise discard every candidate and put the question into the high-priority buffer. Before a commit replaces `M_t`, the old memory can be archived with `case_id`, `step`, `exp_name`, and a UTC timestamp for debugging memory evolution.

## GRPO Training With verl

The minimal verl integration keeps verl unchanged and plugs in a custom reward function from `case_graph/grpo_adapter.py`. Each training row stores one refactoring state in `reward_model.ground_truth`; the policy model only needs to output JSON chunks. During reward computation, the adapter parses those chunks, builds `M_temp`, runs local answer-presence regression checks, and returns the sandbox reward for GRPO.

Prepare parquet data from a JSON list of refactoring states:

```bash
python3 -m case_graph.grpo_data_cli \
  --states outputs/memory_grpo/train_states.json \
  --output outputs/memory_grpo/train.parquet
```

Run minimal GRPO:

```bash
TRAIN_FILE=outputs/memory_grpo/train.parquet \
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct \
bash scripts/run_memory_grpo_verl.sh
```

## Full Online Algorithm

Use the pipeline runner to connect attack generation, oracle checking, retrieval, initial answering, action routing, sandbox refactoring, reward calculation, GRPO-style rollout settlement, commit/rollback, success-pool updates, and debug memory archiving.

```bash
CASE_GRAPH_PROVIDER=deepseek \
DEEPSEEK_API_KEY=... \
./scripts/run_memory_algorithm.sh \
  --memory data/memory_store.json \
  --graphs outputs/case_graphs_deepseek_test \
  --max-graphs 5 \
  --routes-per-graph 2 \
  --memory-output outputs/final_memory.json \
  --trace-output outputs/memory_algorithm_trace.json \
  --attack-trace-output outputs/attack_oracle_trace.json \
  --success-pool-output outputs/success_pool.json \
  --high-priority-buffer-output outputs/high_priority_buffer.json \
  --memory-archive-dir outputs/memory_archive \
  --exp-name debug_run \
  --tau 1.0 \
  --policies random_walk \
  --top-k 5 \
  --proposal-count 4
```

With `--graphs`, the runner first creates graph routes, generates attacks, verifies each attack against `golden_facts`, and runs the oracle check by answering from `golden_facts` directly. Only oracle-passable attacks enter the defense loop. You can still pass a prebuilt attack file with `--attacks`; each attack must provide `question`, `answer`, and optional `golden_facts`. If an attack file lacks `answer`, add `--derive-missing-answer` to ask the Golden Fact Answer Agent to fill it from `golden_facts`.

Sampling counts are explicit: `--max-graphs` controls how many case graphs are used, `--routes-per-graph` controls how many routes/questions are sampled per graph and policy, and `--proposal-count` controls how many refactoring proposals are evaluated per question in the online sandbox. In verl GRPO training, the equivalent rollout count is `ROLLOUT_N` in `scripts/run_memory_grpo_verl.sh`.

Build offline GRPO training data directly from the trace:

```bash
./scripts/build_grpo_data_from_trace.sh \
  --trace outputs/memory_grpo/trace.json \
  --states-output outputs/memory_grpo/train_states.json \
  --parquet-output outputs/memory_grpo/train.parquet \
  --top-k 5
```

Each parquet row contains only the minimal GRPO state `S_t = (M_t, Q, F, action, regression_set)` plus the gold answer for reward calculation. Regenerate `trace.json` with the current pipeline if an older trace is missing `current_memory`.

## Evaluate Compressed Memory on Target Questions

After online training, evaluate held-out CaseGraph target questions by loading the compressed memory checkpoint, retrieving with the existing frozen BM25 retriever, and asking the configured LLM answer agent:

```bash
./scripts/evaluate_target_questions.sh \
  --graphs outputs/case_graphs_test \
  --memory-dir outputs/online_grpo/checkpoint_final/memory_states \
  --output outputs/eval_target_questions.json \
  --top-k 5
```

Use `--memory outputs/final_memory.json` instead of `--memory-dir` when every test case should share one memory store. The output contains per-case retrieval hits, answer-agent output, judge result, and aggregate accuracy.
# adversarial_memory_refactoring
