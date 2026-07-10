# GPU Utilization Optimization for Online GRPO Training

## Problem
Low GPU utilization during `ray::WorkerDict.actor_rollout_update_actor` phase, indicating rollout/generation bottleneck.

## Root Causes

1. **Micro batch sizes too small (1)**: GPUs underutilized when processing only 1 sample at a time
2. **Retriever on CPU**: Dense embedding computation in reward function runs slowly on CPU
3. **Limited reward parallelism**: Only 4 workers processing rewards sequentially

## Optimizations Applied

### 1. Increased Micro Batch Sizes (1 → 4)

**Changed in `configs/online_grpo.yaml`:**
```yaml
# Top-level (added new)
ppo_micro_batch_size_per_gpu: 4
log_prob_micro_batch_size_per_gpu: 4

# verl.actor_rollout_ref.model
ppo_micro_batch_size_per_gpu: 4  # was 1

# verl.actor_rollout_ref.rollout
log_prob_micro_batch_size_per_gpu: 4  # was 1

# verl.actor_rollout_ref.ref
log_prob_micro_batch_size_per_gpu: 4  # was 1
```

**Impact**: 
- Processes 4 samples per GPU pass instead of 1
- Better GPU saturation during rollout and log-prob computation
- Should see ~3-4x speedup in rollout phase

### 2. Moved Retriever to GPU

**Changed in `configs/online_grpo.yaml`:**
```yaml
retriever:
  device: cuda  # was cpu
```

**Impact**:
- Dense Contriever embeddings computed on GPU in reward workers
- Faster reward computation (each rollout needs top-k retrieval)
- May use additional GPU memory (~500MB per worker)

### 3. Increased Reward Workers (4 → 8)

**Changed in `configs/online_grpo.yaml`:**
```yaml
reward_num_workers: 8  # was 4
```

**Impact**:
- More parallel reward computation
- Better throughput when reward function is the bottleneck
- Requires more CPU memory (each worker loads retriever)

## Expected Improvements

**Before:**
- Micro batch = 1: GPU processes one sample at a time (low utilization)
- Retriever on CPU: slow embedding computation in reward loop
- 4 reward workers: limited parallelism

**After:**
- Micro batch = 4: GPU processes 4 samples in parallel (higher utilization)
- Retriever on GPU: fast embedding computation
- 8 reward workers: better parallelism for reward computation

**Estimated speedup**: 3-5x faster rollout phase depending on GPU memory bandwidth and reward complexity.

## Monitoring

Watch for these metrics after restarting training:

1. **GPU utilization**: Should increase from ~20-30% to 60-80% during rollout
2. **Samples/sec**: Should increase 3-5x
3. **Ray task durations**: `actor_rollout_update_actor` should take less time per batch

## Tuning Further (if needed)

If GPU memory allows, you can increase micro batch size further:
```bash
# Try 8 samples per batch
PPO_MICRO_BATCH_SIZE_PER_GPU=8 \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8 \
./scripts/run_online_memory_grpo.sh
```

If GPU memory is tight:
- Reduce back to `ppo_micro_batch_size_per_gpu: 2`
- Reduce `gpu_memory_utilization: 0.5` in rollout config
- Use separate GPUs for training and reward computation

## Additional Bottleneck Checks

If GPU utilization is still low after these changes:

1. **Check reward computation time**: If reward function is slow, consider:
   - Set `REWARD_MODE=semantic_complete` (default, no LLM judge)
   - Reduce `regression_sample_size` from 12 to 8
   - Reduce `top_k_points` from 32 to 16

2. **Check attack generation**: If vLLM server is slow:
   - Increase vLLM `--max-model-len` if OOM
   - Use faster attacker model
   - Reduce `max_attack_attempts` from 20 to 10

3. **Check data loading**: If dataset is slow:
   - Reduce `routing_attempts` from 12 to 8
   - Reduce `routing_max_steps` from 4 to 3
   - Pre-generate attack routes offline

## Files Modified

- `configs/online_grpo.yaml` - increased micro batch sizes, moved retriever to GPU, increased workers
- `case_graph/grpo_adapter.py` - added defensive handling for malformed policy outputs (fixes string chunk error)
- `case_graph/llm.py` - added retry logic with exponential backoff for API timeouts, increased default timeout from 120s to 180s

## API Reliability Improvements

### Automatic Retry on Transient Failures

Added retry logic to handle transient API failures (HTTP 522 connection timeouts, 5xx server errors):
- **3 retries** with exponential backoff (1s, 2s, 4s between attempts)
- Retries on: HTTP 5xx errors, HTTP 522 (connection timeout), network errors
- No retry on: HTTP 4xx errors (client errors like invalid API key)

### Increased Timeout

Default timeout increased from 120s to 180s to handle complex judge prompts. Override with:
```bash
CASE_GRAPH_TIMEOUT=240 ./scripts/run_online_memory_grpo.sh
```

### Common API Issues

**HTTP 522 Connection Timeout:**
- API provider overloaded or network congestion
- Now handled automatically with retries
- If persistent, check API provider status or try different provider

**Rate Limiting:**
- Set `CASE_GRAPH_TIMEOUT=300` for slower API providers
- Reduce `reward_num_workers` to 4 if hitting rate limits
- Use local vLLM server instead of cloud APIs for judge calls
