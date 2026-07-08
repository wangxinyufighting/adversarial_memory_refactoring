# Online GRPO Training Guide

## Quick Start

```bash
# 1. Run complete test (recommended first)
bash scripts/test_online_grpo_complete.sh

# 2. Full training on 3 graphs
GRAPHS_DIR=outputs/case_graphs_train \
MODEL_PATH=/path/to/Qwen3-0.6B \
bash scripts/run_online_memory_grpo.sh
```

## System Overview

This implements **online GRPO** training where:
- Attacks are generated **on-demand** during training (not pre-computed)
- Each case graph maintains **persistent memory M_t** across episodes
- Policy model learns to refactor memory via **group-relative policy optimization**

### Architecture

```
Training Loop
    ↓
Dataset.__getitem__() → Generate attack → Prepare state
    ↓
verl Rollout → Policy generates N proposals per state
    ↓
Reward Function → Sandbox evaluation (existing grpo_adapter.compute_score)
    ↓
verl GRPO → Compute advantages (grouped by UID) → Update policy
    ↓
Post-batch hook → Commit best proposals to per-case M_t
```

## Files Implemented

### Core Components
- `case_graph/online_memory_dataset.py` - Dataset with on-demand state generation
- `case_graph/online_memory_environment.py` - (integrated in dataset.py)
- `case_graph/online_memory_trainer.py` - Training orchestration
- `case_graph/online_memory_cli.py` - CLI entry point

### Scripts
- `scripts/run_online_memory_grpo.sh` - Main training launcher
- `scripts/test_online_grpo_complete.sh` - Complete test workflow
- `scripts/split_case_graphs.py` - Split graphs into train/val/test

### Configuration
- `configs/online_grpo.yaml` - Training hyperparameters

### Tests
- `tests/test_online_memory_dataset.py` - Unit tests (10 tests, all passing ✓)

### Modified Files
- `case_graph/grpo_adapter.py` - Added `build_verl_row_online()`
- `case_graph/pipeline.py` - Extracted `prepare_refactor_state()`
- `verl/verl/trainer/ppo/v1/trainer_base.py` - Calls dataset post-batch/final hooks
- `verl/verl/trainer/ppo/ray_trainer.py` - Calls dataset post-batch/final hooks

## Dataset Split

Current split (seed=42):
- **Train**: 3 graphs (58bf7951, 1e043500, 51a45a95)
- **Val**: 1 graph (e47becba)
- **Test**: 1 graph (118b2229)

To re-split:
```bash
python3 scripts/split_case_graphs.py outputs/case_graphs_deepseek_test \
  --train-ratio 0.6 --val-ratio 0.2 --seed 42
```

## Configuration Parameters

### Environment (configs/online_grpo.yaml)
```yaml
tau: 0.7                    # Add/merge similarity threshold
top_k: 5                    # Retrieval top-K
regression_sample_size: 3   # Regression test sample size
episodes_per_case: 100      # Episodes per case graph
commit_threshold: 0.0       # Commit if best reward > threshold
```

### Training
```yaml
rollout_n: 4                # N proposals per state (GRPO)
train_batch_size: 4         # Batch size
num_epochs: 3               # Training epochs
seed: 42                    # Random seed
```

### Attack Generation
```yaml
routing_policy: "random_walk"
routing_max_steps: 3
routing_min_nodes: 1
max_attack_attempts: 10
```

## Usage Examples

### 1. Basic Training
```bash
GRAPHS_DIR=outputs/case_graphs_train \
MODEL_PATH=/path/to/model \
OUTPUT_DIR=outputs/my_training \
bash scripts/run_online_memory_grpo.sh
```

### 2. Custom Parameters
```bash
GRAPHS_DIR=outputs/case_graphs_train \
MODEL_PATH=/path/to/model \
OUTPUT_DIR=outputs/custom_run \
ROLLOUT_N=8 \
TRAIN_BATCH_SIZE=8 \
EPISODES_PER_CASE=200 \
TAU=0.8 \
bash scripts/run_online_memory_grpo.sh
```

### 3. Validation Run
```bash
GRAPHS_DIR=outputs/case_graphs_val \
MODEL_PATH=outputs/my_training/checkpoint_final \
OUTPUT_DIR=outputs/validation \
EPISODES_PER_CASE=50 \
bash scripts/run_online_memory_grpo.sh
```

### 4. Direct Python CLI
```bash
python3 -m case_graph.online_memory_cli \
  --graphs outputs/case_graphs_train \
  --model-path /path/to/model \
  --output-dir outputs/test_run \
  --config configs/online_grpo.yaml \
  --episodes-per-case 50 \
  --rollout-n 4 \
  --tau 0.7
```

## Key Features

✅ **On-demand Attack Generation**: Uses `RandomWalkRoutingPolicy` + `FrozenLLMAttacker`  
✅ **Per-Case Memory**: Each case maintains independent M_t  
✅ **Persistent Memory**: M_t accumulates across episodes  
✅ **GRPO Compatibility**: UIDs enable group-relative advantages  
✅ **Commit/Rollback**: Best proposal selection with threshold  
✅ **Checkpointing**: Saves model + memory states  
✅ **Reproducible**: Deterministic seeding for attacks  

## Data Flow

### Episode Generation
1. **Select case** (round-robin): case_1 → case_2 → case_3 → case_1 ...
2. **Generate attack**: Route graph → Generate Q/A with seed
3. **Initial defense**: Check if M_t already answers Q
4. **If not**: Prepare refactor state S_t with action (add/merge)
5. **Return verl row**: Format for training with UID

### Training Step
1. **Batch sampling**: DataLoader gets batch of states
2. **Policy rollout**: Generate N proposals per state
3. **Reward**: Sandbox evaluation for each proposal
4. **GRPO update**: Compute advantages grouped by UID → Update policy
5. **Commit**: Apply best proposal (reward > threshold) to M_t

### Memory Update
```python
if best_reward > commit_threshold:
    M_temp = build_sandbox_memory(M_t, best_proposal)
    M_t = M_temp  # Commit
    success_pool.add(question, ...)
else:
    # Rollback: M_t unchanged
    high_priority_buffer.add(question, ...)
```

## Testing

### Run All Tests
```bash
python3 -m pytest tests/test_online_memory_dataset.py -v
```

### Test Coverage
- ✓ CaseMemoryState initialization and episode counting
- ✓ OnlineMemoryEnvironment case state management
- ✓ Episode generation with attack creation
- ✓ Initial defense success handling
- ✓ Memory commit updates
- ✓ OnlineMemoryDataset initialization
- ✓ Graph validation
- ✓ verl format compatibility
- ✓ Commit delegation to environment

All 10 tests passing!

## Output Structure

```
outputs/
├── case_graphs_train/           # Training graphs (3)
├── case_graphs_val/             # Validation graphs (1)
├── case_graphs_test/            # Test graphs (1)
└── my_training/
    ├── training_config.yaml     # Merged config
    ├── initial_memory_states/   # M_t at start
    │   ├── case_1_ep0.json
    │   ├── case_1_ep0_pool.json
    │   └── case_1_ep0_buffer.json
    ├── checkpoint_step100/
    │   ├── memory_states/       # M_t at checkpoint
    │   └── training_metadata.json
    └── checkpoint_final/
        └── memory_states/       # Final M_t
```

## Monitoring Training

### Check Memory Evolution
```bash
# Initial state
cat outputs/my_training/initial_memory_states/58bf7951_ep0.json

# After 100 steps
cat outputs/my_training/checkpoint_step100/memory_states/58bf7951_ep*.json
```

### Training Metadata
```bash
cat outputs/my_training/checkpoint_final/training_metadata.json
```

Shows:
- `global_step`: Total training steps
- `epoch`: Current epoch
- `case_episodes`: Episode count per case

## Troubleshooting

### Issue: No valid graphs found
```bash
# Check graphs exist
ls outputs/case_graphs_deepseek_test/*.case_graph.json

# Verify graph structure
python3 -c "
import json
g = json.load(open('outputs/case_graphs_deepseek_test/58bf7951.case_graph.json'))
print(f'Entities: {len(g[\"entities\"])}')
print(f'Relationships: {len(g[\"relationships\"])}')
"
```

### Issue: Attack generation fails
- Check LLM API keys are set (OPENAI_API_KEY, DEEPSEEK_API_KEY, etc.)
- Verify graphs have sufficient structure (≥3 entities, ≥2 relationships)
- Increase `max_attack_attempts` in config

### Issue: All episodes skip (initial defense succeeds)
- Memory M_t is too good (already answers everything)
- Lower `tau` threshold to force more refactoring
- Or: start with empty memory (`--initial-memory-dir` not set)

## Evaluation

Evaluate target questions on compressed online memory:

```bash
./scripts/evaluate_target_questions.sh \
  --graphs outputs/case_graphs_test \
  --memory-dir outputs/online_grpo/checkpoint_final/memory_states \
  --output outputs/eval_target_questions.json \
  --top-k 5
```

For a detailed code-flow map, see `ONLINE_GRPO_CODE_FLOW.md`.

## References

- **Plan**: `/Users/ganning/.claude/plans/now-this-codebase-is-abundant-scott.md`
- **Offline GRPO**: `scripts/run_memory_grpo_verl.sh` (for comparison)
- **Dataset docs**: `case_graph/online_memory_dataset.py` docstrings
