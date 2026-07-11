# Online GRPO Memory Optimization Report

## 背景

当前目标是从 raw sessions 中学习得到足够小、但足够完备的 memory。原先 online GRPO 的 reward 主要检查 gold answer 是否出现在 retrieved memory 文本中；这会把 defender policy 推向 answer-only memory，例如只写 `Jose Altuve` 或 `Santa Cruz`。这种 memory 很短，但缺少 subject/relation/context，后续 evaluation 的 retrieved-memory answer agent 并不能可靠恢复事实。

本次优化围绕三件事展开：

1. 让 reward 不再只奖励 answer string presence，而是加入关系/事实完整性 gating。
2. 让 retrieval 从简单 BM25 切换到 UnifiedMem 风格的 non graph-based dense structured retrieval。
3. 让 training、evaluation memory construction、target evaluation 使用同一套 retriever 配置和更接近的评估逻辑。

## 关键问题

### Reward 过度奖励短答案

旧逻辑在 `case_graph/grpo_adapter.py` 中使用 `_presence_test`：

- 用 BM25 从 `M_t` 中 retrieve top-k chunks。
- 拼接 evidence。
- 检查 normalized answer 是否为 evidence substring。

这无法区分：

- `Jose Altuve`
- `The user admired Jose Altuve after watching the Astros.`

两者都包含 answer，但只有后者包含完整 relation。

### Retrieval 只索引 chunk.content

旧 `FrozenBM25Retriever` 只对 `MemoryChunk.content` 做 sparse matching。它不支持：

- fact-level retrieval
- summary/keyword expansion
- dense semantic matching
- flatten points 后再聚合回 memory chunk

这和 UnifiedMem 的 flat/non graph-based retrieval 目标不一致。

### Online commit 可能误用 diagnostic reward

旧 commit hook 对 `rm_scores` 一整行求和。若 reward 函数返回 `score` 之外的 diagnostics，commit 决策会被诊断项污染。本次改为优先读取 scalar score channel，不再把 diagnostics 当 reward 相加。

## UnifiedMem Retrieval 对齐

参考 UnifiedMem 的 non graph-based retrieval，采用以下思想：

- retriever model 支持 `facebook/contriever` 等 dense encoder。
- memory index 不只包含 raw content，还包含 facts、summary、keywords。
- `flatten` 模式把 memory 拆成 fact/summary/keyword/content points 进行排序，再聚合回 memory chunk。

本仓库中新增 `DenseStructuredMemoryRetriever`：

```yaml
retriever:
  type: dense_structured
  embedding_model: contriever
  model_name: facebook/contriever
  retrieval_mode: flatten
  top_k_points: 24
  fields: ["facts", "summary", "keywords", "content"]
```

默认会尝试加载 HuggingFace `facebook/contriever`。如果当前环境没有 transformers/model cache，且 `require_model: false`，会退回 deterministic hashed dense retriever，便于单元测试和无网络环境 debug。正式实验建议提前下载 Contriever，并设置：

```yaml
retriever:
  require_model: true
  model_name: /mnt/local2/wxy/models/facebook-contriever
```

## Reward 新设计

新 reward 使用 gated design。核心思想是：compactness 只能在 completeness 之后生效。

### 当前问题 correctness

默认 `semantic_complete` 模式仍然使用 retrieved memory 判断 answer presence，但它不再足以拿高分。answer presence 只是第一层必要条件。

可选 `evaluation_aligned` 模式会在 reward worker 内调用：

- `RetrievedMemoryAnswerAgent`
- `AnswerEquivalenceJudge`

这样 reward 更接近 Evaluation，但每个 rollout 都需要 LLM call，训练成本明显更高。

### Completeness gate

新增语义/关系完整性检查：

- answer 是否出现在 retrieved evidence 中
- proposed chunk 是否保留 question/golden facts 中的 relation terms
- proposed chunk 是否覆盖 answer 之外的关键 fact terms
- 是否为 answer-only memory
- 是否 grounded in golden facts / selected old chunks
- 是否和未删除旧 chunk 冗余
- 是否大量 raw copy

如果 current answer 不正确，reward 上限为 `<= 0`。

如果 answer 正确但 completeness 不通过，reward 上限为 `<= 0.5`。

这会阻止 answer-only chunk 被 commit。

### Reward parts

默认权重：

```yaml
reward_config:
  mode: semantic_complete
  min_relation_overlap: 0.25
  min_fact_overlap: 0.25
  min_non_answer_tokens: 3
  weights:
    current_correct: 3.0
    current_wrong: -3.0
    completeness: 2.0
    completeness_missing: 2.0
    grounded: 1.0
    regression_accuracy: 1.25
    regression_failure: 2.0
    chunk_count: 0.2
    tokens_per_100: 0.15
    duplicate: 0.5
    answer_only: 1.5
    raw_copy: 0.5
```

这比旧版长度惩罚强很多，但不会先于 completeness 发生作用。

## Training / Evaluation 对齐

以下路径现在可以共享 retriever config：

- online training state preparation
- initial defense
- add/merge action routing
- GRPO reward evaluation
- evaluation memory construction sandbox
- target question evaluation

新增或更新的关键文件：

- `case_graph/retriever.py`
  - 新增 `DenseStructuredMemoryRetriever`
  - 新增 `build_memory_retriever`
  - 新增 `retriever_config_from_mapping`
  - `MemoryChunk.from_dict` 保留 top-level `facts/summary/keywords/source_ids`

- `case_graph/memory_evaluator.py`
  - 新增 shared evaluation helper
  - 新增 completeness / grounding / redundancy diagnostics

- `case_graph/grpo_adapter.py`
  - reward 不再使用旧 answer-presence-only `_presence_test`
  - prompt 要求输出 `facts/keywords/summary/source_ids`
  - reward payload 增加 completeness diagnostics

- `case_graph/refactoring.py`
  - reward weights 加强
  - reward gating
  - sandbox retrieval 支持 configured retriever

- `case_graph/pipeline.py`
  - `AlgorithmConfig` 增加 `top_k_points/retriever_config/reward_config`
  - initial defense 和 action router 使用 configured retriever

- `Evaluation/memory_construction.py`
  - construction sandbox 改用 shared evaluator
  - evaluation memory construction 可使用 same dense retriever + completeness reward

- `case_graph/evaluation.py`
  - target evaluation 支持 configured retriever

## Training 参数调整

`configs/online_grpo.yaml` 不再是 smoke-test 级别：

```yaml
num_epochs: 1
train_batch_size: 16
ppo_mini_batch_size: 8
rollout_n: 8
max_questions_per_case: 200
coverage_threshold: 0.98
critical_coverage_threshold: 1.0
training_probe_window: 12
top_k: 8
top_k_points: 32
regression_sample_size: 12
commit_threshold: 1.0
max_response_length: 1024
save_freq: 500
```

这些参数更适合正式训练：

- `rollout_n=8` 提高 GRPO group 内候选质量。
- `train_batch_size=16/ppo_mini_batch_size=8` 避免 4-sample smoke run 的高方差。
- `max_questions_per_case=200` 只是每个 case 的安全上限；case 达到 coverage 和 probe 条件后会提前退出 active pool。
- `num_epochs=1` 避免 300 个训练 case 被固定重复 8 轮；实际训练规模由 active-case completion 和数据量共同决定。
- `regression_sample_size=12` 降低 merge 时遗忘旧 memory 的概率。
- `top_k=8/top_k_points=32` 适配 fact-level dense retrieval。
- `commit_threshold=1.0` 避免 completeness 边缘通过的 rollout 被写入 `M_t`。
- `max_response_length=1024` 给 structured memory JSON 留足空间。

## CLI / Script 新增参数

`scripts/run_online_memory_grpo.sh` 支持：

```bash
TOP_K_POINTS=24
REGRESSION_SAMPLE_SIZE=8
RETRIEVER_TYPE=dense_structured
RETRIEVER_MODEL_NAME=/mnt/local2/wxy/models/facebook-contriever
RETRIEVER_EMBEDDING_MODEL=contriever
RETRIEVER_RETRIEVAL_MODE=flatten
RETRIEVER_REQUIRE_MODEL=true
REWARD_MODE=semantic_complete
```

`scripts/construct_defender_memory_for_eval.sh` 同样支持上述 retriever/reward 参数。

Target evaluation CLI 新增 dense retriever 参数：

```bash
python -m case_graph.evaluation_cli \
  --graphs outputs/case_graphs_test \
  --memory-dir outputs/eval_memory_construction/global_step_750/memory_states \
  --output outputs/eval_results/global_step_750.json \
  --retriever-type dense_structured \
  --retriever-model-name /mnt/local2/wxy/models/facebook-contriever \
  --retriever-embedding-model contriever \
  --retriever-retrieval-mode flatten \
  --top-k 8 \
  --top-k-points 24
```

## 推荐运行方式

正式训练前建议先准备 Contriever：

```bash
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('facebook/contriever'); AutoModel.from_pretrained('facebook/contriever')"
```

如果服务器不能联网，提前下载到本地：

```yaml
retriever:
  model_name: /mnt/local2/wxy/models/facebook-contriever
  require_model: true
```

训练仍然使用：

```bash
./scripts/run_online_memory_grpo.sh
```

高成本、更接近 Evaluation 的 reward 可这样开启：

```bash
REWARD_MODE=evaluation_aligned ./scripts/run_online_memory_grpo.sh
```

注意：`evaluation_aligned` 会在 reward worker 内调用 answer agent + judge，需要确保对应 OpenAI-compatible API 环境可用。

## Verification

## Adaptive Coverage And Certification

Training no longer treats a fixed question count as evidence of completeness. Each case now owns a `CaseCoverageTracker` built only from public graph relationships. The existing random-walk routing policy is unchanged, while a coverage-aware scheduler samples several random walks and prefers routes containing uncovered or previously failed units.

The GRPO reward now includes `coverage_gain` and `critical_coverage_gain`, but only when the retrieved answer and structured completeness gate both pass. Before commit, the winning rollout is re-evaluated against the latest live `M_t`, preventing stale same-batch Merge proposals from overwriting newer facts.

Evaluation memory construction uses an adaptive state machine:

```text
cover -> repair -> certify -> done
```

A case enters certification after weighted coverage reaches 0.98 and critical coverage reaches 1.0. It stops only after the configured number of consecutive fresh probes pass; otherwise it continues until `episodes_per_case`, now interpreted as the maximum question budget. Coverage state is written to `coverage_states/<case_id>.json`.

Recommended formal split and invocation:

```bash
VAL_GRAPHS_DIR=outputs/case_graphs_val \
MAX_QUESTIONS_PER_CASE=200 \
./scripts/run_online_memory_grpo.sh

MIN_QUESTIONS_PER_CASE=20 \
COVERAGE_THRESHOLD=0.98 \
CRITICAL_COVERAGE_THRESHOLD=1.0 \
CERTIFICATION_QUESTIONS=60 \
EPISODES_PER_CASE=250 \
./scripts/construct_defender_memory_for_eval.sh
```

已通过相关单元测试：

```bash
python3 -m unittest discover -s tests -p 'test_retriever.py'
python3 -m unittest discover -s tests -p 'test_grpo_adapter.py'
python3 -m unittest discover -s tests -p 'test_refactoring.py'
python3 -m unittest discover -s tests -p 'test_online_memory_dataset.py'
python3 -m unittest discover -s tests -p 'test_evaluation.py'
python3 -m unittest discover -s tests -p 'test_memory_construction.py'
```

全量 `python3 -m unittest discover -s tests` 仍有若干 pre-existing import errors，缺失模块包括 `case_graph.attacker_server_manager`、`case_graph.attacker_grpo_adapter`、`case_graph.online_attacker_dataset` 和 `baselines.*`。这些错误与本次 memory reward/retrieval 优化无关。
