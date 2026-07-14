# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements an adversarial memory refactoring system for testing and improving LLM memory mechanisms. The core pipeline:

1. **Graph Construction**: Converts LongMemEval conversation data into case graphs with entities and relationships
2. **Attack Generation**: Creates adversarial questions via graph routing policies to challenge memory systems
3. **Defense & Refactoring**: Tests memory retrieval, determines add/merge actions, and refactors memory stores
4. **GRPO Training**: Trains memory refactoring policies using Group Relative Policy Optimization via verl

The architecture follows a multi-stage pipeline where each stage is independently testable. Core logic lives in `case_graph/` modules; shell scripts in `scripts/` provide workflow entrypoints.

## Key Architecture Components

### Core Modules (`case_graph/`)

**Graph Construction & Attack Generation:**
- **`models.py`**: Data structures for graphs (CaseGraph, EntityRecord, RelationshipEdge, SessionChunk)
- **`builder.py`**: Extracts entities/relationships from conversation sessions via LLM API
- **`routing.py`**: Attack path selection (random_walk, heuristic, feature_scored_llm_rerank)
- **`attacker.py`**: Generates adversarial questions from graph routes
- **`verifier.py`**: Checks if attack answers are supported by golden facts
- **`baseline.py`**: Oracle test - can golden facts alone answer the question?

**Memory Defense & Retrieval:**
- **`retriever.py`**: Memory retrieval (BM25, FrozenBM25Retriever, DenseStructuredRetriever with Contriever)
- **`defense.py`**: Initial defense using retrieved memories
- **`refactoring.py`**: Similarity routing (add/merge decisions), sandbox evaluation, GRPO settlement

**Pipeline & Training:**
- **`pipeline.py`**: End-to-end offline algorithm orchestration
- **`online_memory_dataset.py`**: Dataset for online GRPO training with per-case memory states
- **`online_memory_trainer.py`**: Trainer orchestrating verl PPO with custom dataset and commit logic
- **`grpo_adapter.py`**: verl integration - custom reward function for memory refactoring
- **`evaluation.py`**: Memory evaluator for offline and online evaluation

**Coverage & Certification:**
- **`coverage.py`**: Coverage tracking with answer-source boosting (CaseCoverageTracker, CoverageAwareRouteScheduler)
- **`certification.py`**: Post-construction memory validation and coverage certification

**Infrastructure:**
- **`llm.py`**: Multi-provider LLM client (OpenAI, DeepSeek, local OpenAI-compatible APIs)
- **`evidence.py`**: Evidence extraction and completeness checking
- **`progress.py`**: Progress bar utilities
- **`cache.py`**: Extraction caching to avoid redundant LLM calls

**Evaluation Module (`Evaluation/`):**
- **`agents.py`**: Answer agents (LongMemEvalMemoryAnswerAgent) and judges (LongMemEvalAnswerJudge)
- **`loaders.py`**: Data loading utilities for LongMemEval and CaseGraphs
- **`evaluator.py`**: Core evaluation logic (evaluate_memory_question, failed_result)
- **`summarization.py`**: Result summarization and comparison (summarize_memory_qa, compare_memory_qa_results)
- **`metrics.py`**: Retrieval and answer metrics (Recall@K, NDCG@K, memory compression)
- **`memory_construction.py`**: Memory construction orchestration with coverage tracking
- **`construct_memory_cli.py`**: CLI for memory construction
- **`evaluate_memory_qa_cli.py`**: CLI for memory QA evaluation

### PYTHONPATH Configuration

The verl submodule and compat shim must be in PYTHONPATH for GRPO training:

```bash
export PYTHONPATH="${ROOT_DIR}/compat:${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}"
```

The `compat/sitecustomize.py` provides Python 3.9 compatibility by shimming `enum.StrEnum` for libraries requiring Python 3.10+.

### LLM Provider Configuration

Three provider modes via `CASE_GRAPH_PROVIDER`:

- **openai**: `OPENAI_API_KEY`, `OPENAI_MODEL` (default: gpt-4o-mini)
- **deepseek**: `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` (default: deepseek-v4-flash), `DEEPSEEK_THINKING=disabled`
- **local**: `LOCAL_API_BASE_URL` (e.g., http://localhost:8000/v1), `LOCAL_MODEL`, optional `LOCAL_API_KEY`

## Training Modes

The system supports two training paradigms:

1. **Offline Training** (`scripts/run_memory_algorithm.sh` + `scripts/run_memory_grpo_verl.sh`):
   - Pre-generate attack dataset from case graphs
   - Build GRPO training parquet from offline trace
   - Train on static dataset

2. **Online Training** (`scripts/run_online_memory_grpo.sh`):
   - Generate attacks on-demand during training episodes
   - Maintain per-case memory states that evolve with training
   - Integrated pipeline: graph → route → attack → defense → refactor → commit
   - Preferred for research and production training

## Quick Start

For a complete end-to-end example, see `scripts/QUICK_START.sh` which demonstrates:
1. Building case graphs from LongMemEval data
2. Generating attack routes
3. Running the full memory algorithm pipeline
4. Evaluating memory systems

Basic workflow:
```bash
# 1. Build case graphs (requires LLM API)
LIMIT=5 ./scripts/build_case_graphs.sh \
  data/longmemeval/longmemeval_s_cleaned.json \
  outputs/case_graphs

# 2. Run online GRPO training
GRAPHS_DIR=outputs/case_graphs \
MODEL_PATH=/path/to/model \
OUTPUT_DIR=outputs/online_grpo \
./scripts/run_online_memory_grpo.sh

# 3. Certify memory coverage (optional but recommended)
./scripts/certify_memories.sh \
  --memory-dir outputs/memory_states \
  --graphs outputs/case_graphs \
  --output outputs/certification.json

# 4. Evaluate trained memory system
./scripts/evaluate_target_questions.sh \
  --graphs outputs/case_graphs \
  --memory outputs/online_grpo/final_memory.json \
  --output outputs/eval_results.json
```

## Common Commands

### Build & Test

```bash
# Run all unit tests
python3 -m unittest discover -s tests

# Run specific test module
python3 -m unittest tests.test_refactoring

# Build small case graph sample (requires LLM API)
LIMIT=1 ./scripts/build_case_graphs.sh \
  data/longmemeval/longmemeval_s_cleaned.json \
  outputs/case_graphs
```

### Attack Generation Pipeline

```bash
# Two-stage: Generate routes first, then attacks
./scripts/generate_routes.sh \
  outputs/case_graphs \
  outputs/routes.json \
  random_walk

./scripts/generate_attacks_from_routes.sh \
  outputs/routes.json \
  outputs/attacks.json

# One-stage: Route and attack together
./scripts/generate_attacks.sh \
  outputs/case_graphs \
  outputs/attacks.json

# Oracle baseline test - filter attacks answerable from golden facts
./scripts/run_baseline_sanity_test.sh \
  outputs/attacks.json \
  outputs/attacks_baseline_passed.json
```

### Memory Defense & Refactoring

```bash
# Test initial defense (retrieval + answering)
./scripts/run_initial_defense.sh \
  data/memory_store.json \
  "Which baseball player did the user admire?" \
  "Jose Altuve" \
  outputs/initial_defense.json \
  outputs/memory_store_updated.json \
  outputs/success_pool.json \
  5

# Similarity routing decision (add vs merge)
python3 -m case_graph.refactoring_cli \
  --memory data/memory_store.json \
  --question "Which Astros player did the user admire?" \
  --tau 1.0 \
  --top-k 5 \
  --output outputs/refactor_decision.json
```

### Memory Evaluation

```bash
# Certify memory coverage before evaluation
./scripts/certify_memories.sh \
  --memory-dir outputs/memory_states \
  --graphs outputs/case_graphs \
  --output outputs/certification.json

# Evaluate memory system on target questions from case graphs
./scripts/evaluate_target_questions.sh \
  --graphs outputs/case_graphs \
  --memory outputs/final_memory.json \
  --output outputs/eval_results.json

# Construct memory for baseline evaluation (UnifiedMem-style)
./scripts/construct_defender_memory_for_eval.sh \
  outputs/case_graphs \
  outputs/baseline_memory.json

# Retrieve memories for a question
./scripts/retrieve_memories.sh \
  data/memory_store.json \
  "Which baseball player did the user admire?" \
  5
```

### Full Online Algorithm

```bash
# End-to-end pipeline: graph→route→attack→oracle→defense→refactor→commit
./scripts/run_memory_algorithm.sh \
  --memory data/memory_store.json \
  --graphs outputs/case_graphs \
  --max-graphs 5 \
  --routes-per-graph 2 \
  --memory-output outputs/final_memory.json \
  --trace-output outputs/trace.json \
  --success-pool-output outputs/success_pool.json \
  --exp-name experiment_name \
  --tau 1.0 \
  --policies random_walk \
  --top-k 5 \
  --proposal-count 4
```

### GRPO Training with verl

**Offline Training (two-stage):**

```bash
# Stage 1: Build GRPO training data from algorithm trace
./scripts/build_grpo_data_from_trace.sh \
  --trace outputs/trace.json \
  --states-output outputs/train_states.json \
  --parquet-output outputs/train.parquet \
  --top-k 5

# Stage 2: Run GRPO training on static dataset
TRAIN_FILE=outputs/train.parquet \
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct \
bash scripts/run_memory_grpo_verl.sh
```

**Online Training (integrated pipeline):**

```bash
# Basic usage - uses defaults from configs/online_grpo.yaml
GRAPHS_DIR=outputs/case_graphs_train \
MODEL_PATH=/path/to/model \
OUTPUT_DIR=outputs/online_grpo \
./scripts/run_online_memory_grpo.sh

# With overrides
GRAPHS_DIR=outputs/case_graphs_train \
MODEL_PATH=/path/to/Qwen3-0.6B \
OUTPUT_DIR=outputs/online_grpo \
ROLLOUT_N=8 \
TRAIN_BATCH_SIZE=8 \
EPISODES_PER_CASE=500 \
RETRIEVER_TYPE=dense_structured \
RETRIEVER_MODEL_NAME=facebook/contriever \
ATTACKER_API_BASE=http://localhost:8003/v1 \
./scripts/run_online_memory_grpo.sh
```

### Key GRPO Training Variables

**Offline Training:**
- `MODEL_PATH`: Base model path (default: /mnt/local2/wxy/models/Qwen3-0.6B)
- `TRAIN_FILE`: Parquet training data with refactoring states
- `ROLLOUT_N`: Number of policy samples per state (default: 4)
- `TRAIN_BATCH_SIZE`: Samples per training batch (default: 2)
- `MAX_PROMPT_LENGTH`: Token limit for prompts (default: 16384)
- `MAX_RESPONSE_LENGTH`: Token limit for responses (default: 512)

**Online Training:**
- `GRAPHS_DIR`: Case graphs directory or single file
- `MODEL_PATH`: Base model for training
- `OUTPUT_DIR`: Checkpoints and memory trajectory output
- `ROLLOUT_N`: Rollouts per state (default: 8)
- `TRAIN_BATCH_SIZE`: Training batch size (default: 8)
- `EPISODES_PER_CASE`: Training episodes per graph (default: 500)
- `TAU`: Dense retriever add/merge threshold (default: 0.55)
- `TOP_K`: Top-K memory chunks retrieved (default: 8)
- `TOP_K_POINTS`: Dense retrieval points before aggregation (default: 24)
- `COMMIT_THRESHOLD`: Minimum reward to commit (default: 1.0)
- `RETRIEVER_TYPE`: `bm25` or `dense_structured`
- `RETRIEVER_MODEL_NAME`: HuggingFace model (e.g., `facebook/contriever`)
- `RETRIEVER_DEVICE`: `cpu` or `cuda:0`
- `REWARD_MODE`: `semantic_complete` (fast) or `evaluation_aligned` (LLM judge)
- `ATTACKER_LLM`: vLLM server model name/path
- `ATTACKER_API_BASE`: vLLM server endpoint (e.g., `http://localhost:8003/v1`)
- `MEMORY_TRAJECTORY_DIR`: Per-step M_t snapshot subdirectory name
- `DISABLE_MEMORY_TRAJECTORY`: Set to disable trajectory logging

## Data Flow

**Offline Pipeline:**
1. **LongMemEval JSON** → `builder.py` → **CaseGraph JSON** (entities, relationships, chunks)
2. **CaseGraph** → `routing.py` → **Routes JSON** (attack paths, golden_facts)
3. **Routes** → `attacker.py` → **Attacks JSON** (question, answer, verification)
4. **Attacks** → `baseline.py` → **Baseline-passed attacks** (oracle-answerable only)
5. **Attack + Memory** → `defense.py` → **Initial defense result** (needs_refactor flag)
6. **Need-refactor** → `refactoring.py` → **Sandbox results** → **Settlement** → **Commit/rollback**
7. **Algorithm trace** → `trace_to_grpo_data_cli.py` → **GRPO training parquet**
8. **Parquet** → `verl.trainer.main_ppo` + `grpo_adapter.py` → **Trained policy model**

**Online Pipeline:**
1. **CaseGraph** → `OnlineMemoryDataset.__getitem__()` at each episode
2. → `routing.py` (select attack route from graph)
3. → `attacker.py` (generate question via vLLM server)
4. → `baseline.py` (oracle verification)
5. → `defense.py` (retrieve + answer)
6. → Policy model rollout (generate N refactor proposals)
7. → `grpo_adapter.py` reward computation (sandbox evaluation)
8. → GRPO settlement (commit best if reward > threshold)
9. → Memory state persists for next episode
10. → verl PPO gradient update

## Important Implementation Details

### Routing Policies are Target-Agnostic

Routing policies select attack paths using only public graph structure and edge weights. The private `target` field from LongMemEval is never exposed to routing or attack generation. Routes may incidentally include gold answers if naturally extracted, but this is not guaranteed.

### Verification-Only Golden Facts

`golden_facts` are raw conversation sessions resolved from route `source_ids`. They are used only for:
- Attack verification (does the answer exist in the facts?)
- Oracle baseline testing (can facts alone answer the question?)
- Sandbox regression testing (does refactored memory preserve answerable questions?)

Attack generation receives only the public `route` field, never `golden_facts` or `target`.

### Memory Refactoring Rewards

Sandbox reward balances multiple factors (see `refactoring.py:compute_reward`):
- Current question correctness
- Regression accuracy on success pool
- Failed regression penalty
- New chunk count penalty
- New chunk length penalty
- Completeness (relation/fact coverage from golden evidence)
- Groundedness (avoid answer-only or raw-copy chunks)
- Duplicate penalty

GRPO settlement commits only when best reward > threshold (default: 1.0 for online, 0.0 for offline), otherwise puts question into high-priority buffer for retry.

### Retriever Types

The system supports multiple retrieval backends:

1. **BM25/FrozenBM25** (lightweight, no external models):
   - BM25 scoring on chunk text
   - No GPU required

2. **DenseStructuredRetriever** (semantic retrieval):
   - Embedding model: Contriever (`facebook/contriever`) or deterministic hash fallback
   - Retrieval modes: `flatten` (all fields), `merge` (concatenate), `separate` (per-field)
   - Fields: `["facts", "summary", "keywords", "content"]`
   - Configurable via `RETRIEVER_TYPE`, `RETRIEVER_MODEL_NAME`, `RETRIEVER_RETRIEVAL_MODE`
   - Set `require_model: false` to fallback to hash when HF model unavailable

Online training defaults to `dense_structured` with Contriever for better semantic matching.

### verl Integration

The `grpo_adapter.py` module plugs into verl's custom reward function interface:
- Each training state `S_t = (M_t, Q, F, action, regression_set)`
- Policy model outputs JSON chunks for add/merge actions
- Adapter parses chunks, builds `M_temp`, runs local answer checks, returns reward
- verl handles gradient computation and model updates; no verl code is modified

**Online Training Architecture:**
- `OnlineMemoryDataset`: Per-case memory states, on-demand attack generation
- `OnlineMemoryTrainer`: Orchestrates verl PPO trainer with custom dataset
- `grpo_adapter.compute_score`: Reward function called by verl workers
- verl `main_ppo`: Standard verl entry point with custom dataset class injection
- Memory trajectory: Optional per-commit snapshots saved to `{output_dir}/{memory_trajectory_dir}/`

## Testing Guidelines

- Use fake LLM clients (`FakeLLMClient`) in unit tests to avoid API calls
- Name test files `test_*.py`, classes `*Tests`, methods `test_<behavior>`
- Run full suite before committing: `python3 -m unittest discover -s tests`
- For live API testing, set appropriate `CASE_GRAPH_PROVIDER` and API keys
- Use `LIMIT=1` for quick smoke tests of graph building or attack generation
- Test files cover all major modules: builders, attackers, retrievers, refactoring, GRPO adapters, datasets, trainers

## Git Conventions

- Keep commits focused: one behavioral change per commit
- Subject line: lowercase imperative style (e.g., "add merge action router", "fix baseline judge")
- Do not commit: secrets, raw datasets (`data/`), generated outputs (`outputs/`), `__pycache__`, `extraction_cache.json`
- Current branch: `online_1` (check git status before pushing)
- Main branch: `main` (use for PRs)

## Important File Locations

- **Configuration**: `configs/online_grpo.yaml` - main training config with verl settings
- **Scripts**: `scripts/*.sh` - workflow entrypoints for all pipelines
- **Tests**: `tests/test_*.py` - unit tests for all modules
- **verl submodule**: `verl/` - GRPO training framework (external dependency)
- **Python 3.9 compat**: `compat/sitecustomize.py` - StrEnum shim for older Python
- **Memory stores**: JSON files with `chunks` array containing `memory_id`, `content`, `metadata`
- **Case graphs**: `*.case_graph.json` with `entities`, `relationships`, `chunks`, `target`

## Debugging Tips

- Set `KEEP_FAILED_VERIFICATION=1` to retain attacks that fail verification checks
- Set `SKIP_ATTACK_VERIFICATION=1` for faster debugging without verification
- Set `KEEP_FAILED_BASELINE=1` to keep oracle-failing attacks for inspection
- Set `SKIP_LLM_JUDGE=1` to avoid LLM judge calls in baseline testing
- Check `extraction_cache.json` to see which sessions were already processed
- Use `--memory-archive-dir` in pipeline to save memory snapshots at each commit for debugging evolution
- For online training: check `{output_dir}/memory_trajectory/{case_id}/` for per-commit memory states
- Set `RETRIEVER_DEVICE=cpu` if GPU memory is limited during training
- Use `--skip-retriever-preflight` to bypass retriever model loading checks before Ray starts
- Check `{output_dir}/training_config.yaml` for merged configuration after training starts
- Monitor wandb logs with `PROJECT_NAME` and `EXPERIMENT_NAME` environment variables

### Common Training Errors

**Contriever "index out of range" error:**
- Caused by input sequences exceeding model's max_position_embeddings (512 for Contriever)
- Fix: Set `RETRIEVER_MAX_LENGTH=256` or lower (already fixed in retriever.py)
- Verify with: check logs for "Resolved retriever max_length" message

**vLLM CUDA illegal memory access:**
- Caused by GPU memory pressure or fragmentation
- Fixes:
  - Reduce `ROLLOUT_GPU_MEMORY_UTILIZATION` (default: 0.6, try 0.5 or 0.4)
  - Use separate GPUs for vLLM server and training: `CUDA_VISIBLE_DEVICES=4,5` with vLLM on GPU 0 (maps to 4), training on GPU 1 (maps to 5)
  - Reduce batch size: `TRAIN_BATCH_SIZE=4` or `PPO_MINI_BATCH_SIZE=2`
  - Restart vLLM server between training runs to clear memory
- Enable debug: `CUDA_LAUNCH_BLOCKING=1` or compile with `TORCH_USE_CUDA_DSA=1`

**Ray worker failures:**
- Check Ray dashboard: `ray dashboard` (usually http://localhost:8265)
- Increase Ray worker timeout: set `RAY_worker_register_timeout_seconds=300`
- Reduce `reward_num_workers` in config (default: 4)
