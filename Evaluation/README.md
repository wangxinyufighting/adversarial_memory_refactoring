# Defender Memory Construction for Evaluation

This directory contains the held-out evaluation memory builder. It uses a trained defender checkpoint as the Memory Refactoring Policy to construct a fixed memory state for every new case before downstream evaluation.

## Flow

1. Load each held-out CaseGraph.
2. Strip evaluator-only target metadata from the graph view used by routing and attacker probes.
3. Sample public graph routes and generate construction probes with the attacker.
4. Run the current memory through the same retrieval and initial-defense check used in training.
5. If refactoring is needed, call the defender checkpoint through an OpenAI-compatible endpoint.
6. Sandbox the proposed memory edit, score it with the existing refactoring reward, and commit only the best positive rollout.
7. Save per-case memory files under `memory_states/` for later evaluation.

The output memory is intentionally fixed. Later evaluation should read `memory_states/<case_id>.json` instead of letting the defender keep editing memory while answering target questions.

## Run With an Existing Server

```bash
DEFENDER_API_BASE=http://localhost:8004/v1 \
DEFENDER_SERVED_MODEL=defender-current \
GRAPHS=outputs/case_graphs_test \
OUTPUT_DIR=outputs/eval_memory_construction/defender-current \
EPISODES_PER_CASE=100 \
./scripts/construct_defender_memory_for_eval.sh
```

## Run and Manage the Defender Server

If `DEFENDER_CHECKPOINT` points to a verl checkpoint rather than a HuggingFace model directory, the CLI first merges it with `python -m verl.model_merger merge`, then starts vLLM, waits for `/v1/models`, and stops the process at the end.

```bash
MANAGE_DEFENDER_SERVER=true \
DEFENDER_CHECKPOINT=outputs/grpo/checkpoints/global_step_100 \
DEFENDER_SERVER_PORT=8004 \
GRAPHS=outputs/case_graphs_test \
OUTPUT_DIR=outputs/eval_memory_construction/global_step_100 \
./scripts/construct_defender_memory_for_eval.sh
```

Important outputs:

- `memory_states/<case_id>.json`: final memory for evaluation.
- `memory_states/<case_id>_pool.json`: successful construction questions bound to memory chunks.
- `memory_states/<case_id>_buffer.json`: failed probes to inspect or retry.
- `traces/<case_id>.trace.json`: episode-level construction decisions.
- `construction_summary.json`: aggregate counts.
