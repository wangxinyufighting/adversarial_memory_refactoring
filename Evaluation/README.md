# Evaluation 方案

本文档定义 memory refactoring 与 baseline 的统一评测协议。核心原则是：只比较 memory 本身，不让 retrieval、answer backbone、judge 或 target 数据变化污染结论。

## 评测目标

我们评测的是不同 memory 构建策略对最终问答能力的影响：

- 我们方法：online GRPO 后的 `checkpoint_*/memory_states`。
- UnifiedMem baseline：来自 `/Users/ganning/Documents/project_python/大模型记忆/UnifiedMem` 的 flat memory / expansion memory 思路。
- Raw session baseline：每个原始 haystack session 作为一个 memory chunk。
- Oracle source baseline：只放入 `answer_source_ids` 对应 session，作为上界诊断，不作为公平方法。

所有方法必须转换为同一个 `MemoryStore` 格式，然后使用同一个评测链路：

```text
CaseGraph target question
-> FrozenBM25Retriever(top_k, min_score)
-> RetrievedMemoryAnswerAgent(answer backbone)
-> AnswerEquivalenceJudge
-> QA accuracy + retrieval source metrics + memory cost
```

因此，UnifiedMem 在这里不是使用自己的 retriever 或 answer pipeline；它只贡献 memory 内容。这样可以保证对比的是“不同 memory 表示”的效果。

## 数据划分建议

建议至少保留三份 graph 目录：

- `outputs/case_graphs_train`: co-training 使用。
- `outputs/case_graphs_dev`: 调参、选择 checkpoint、检查 prompt。
- `outputs/case_graphs_test`: 只做最终报告。

如果当前只有一份 LongMemEval graph，先按 `case_id` 做固定随机切分，并保存 split manifest。不要在 test 上选择 checkpoint 或调参。

## 指标

### 1. End-to-end QA

主指标：

- `accuracy`: answer agent 输出是否等价于 gold answer。

判定方式：

- 先使用 normalized string match。
- string match 不通过时，可启用 LLM judge。
- 所有 baseline 必须共用同一个 judge 设置。

### 2. Retrieval source metrics

如果 graph 的 `target.answer_source_ids` 存在，则计算：

- `source_recall`: top-k retrieved memories 覆盖了多少 gold source id。
- `source_hit`: top-k 是否命中任一 gold source id。
- `source_mrr`: 第一个命中 source 的 reciprocal rank。
- `source_ndcg`: source-aware nDCG。

对于压缩 memory，如果 memory chunk 的 metadata 包含 `source_ids` / `all_ids`，评测会用这些字段追溯原始 session id。Raw session 和 UnifiedMem adapter 都会写入 `source_ids`。

### 3. Memory cost

同时报告：

- `memory_chunks`: memory chunk 数。
- `memory_chars`: memory 总字符数。
- `approx_memory_tokens`: 近似 token 数，按 `chars / 4` 粗估。

如果两个方法 QA accuracy 接近，应优先看 retrieval 指标与 memory cost。更少 memory token 达到同等 accuracy，说明 memory 更有效。

## Baseline 设计

### Raw session

每个 CaseGraph `chunks[]` 直接作为一条 memory。它是强基线，因为包含完整历史信息，但 memory 成本高。

### UnifiedMem

`baselines/unifiedmem.py` 将 CaseGraph session 转成 UnifiedMem flat-memory 风格：

- 无 expansion cache 时：回退为 raw session 内容。
- 有 expansion cache 时：使用 user facts / keyphrases / summaries 作为 memory 内容。
- `join_mode=merge`: 只用 expansion 内容，缺失 expansion 时回退 raw。
- `join_mode=merge_raw`: raw + expansion 一起作为 memory。
- `join_mode=separate`: 每种 expansion 单独成为一个 memory chunk。

评测时仍然使用本仓库的 `FrozenBM25Retriever` 与 `RetrievedMemoryAnswerAgent`。

### Oracle source

只把 `answer_source_ids` 对应 session 放入 memory。它使用 target metadata，因此不公平，只用于检查 answer backbone 和 judge 是否能在“证据一定存在”时工作。

## 推荐运行方式

评测 raw session、UnifiedMem、以及某个训练 checkpoint：

```bash
./scripts/run_evaluation_suite.sh \
  --graphs outputs/case_graphs_test \
  --memory-dir ours=outputs/adversarial_cotrain/defender_ckpt_t000/checkpoint_latest/memory_states \
  --baseline raw_session \
  --baseline unifiedmem \
  --baseline oracle_source \
  --output outputs/evaluation/test_suite.json \
  --top-k 5
```

如果有 UnifiedMem expansion cache：

```bash
./scripts/run_evaluation_suite.sh \
  --graphs outputs/case_graphs_test \
  --baseline unifiedmem \
  --unifiedmem-expansion-cache /path/to/session-userfact.json \
  --unifiedmem-expansion-cache /path/to/session-keyphrase_.json \
  --unifiedmem-expansion-cache /path/to/session-summ_.json \
  --unifiedmem-join-mode merge \
  --output outputs/evaluation/unifiedmem.json
```

如果还没有 cache，可以先在 UnifiedMem 仓库中生成。下面命令会调用 UnifiedMem 原始脚本，产出本评测适配器可读取的三个 JSON cache：

```bash
cd /Users/ganning/Documents/project_python/大模型记忆/UnifiedMem
export LME_IN_FILE=/Users/ganning/Documents/project_python/大模型记忆/adversarial_memory_refactoring/data/longmemeval/longmemeval_s_cleaned.json
export LME_CACHE_FOLDER=/tmp/unifiedmem_expansions
python data_preprocessing/lme_extract_userfact.py
python data_preprocessing/lme_extract_keyphrase.py
python data_preprocessing/lme_extract_summ.py
```

随后把 `/tmp/unifiedmem_expansions/session-userfact.json`、`session-keyphrase_.json`、`session-summ_.json` 传给 `--unifiedmem-expansion-cache`。

快速跑 evaluation 相关单测：

```bash
./scripts/test_evaluation_suite.sh
```

## 报告格式

输出 JSON 包含：

- `evaluation_protocol`: 固定的 retriever、answer backbone、judge、top-k。
- `baselines[]`: 每个方法的 summary 与逐 case 结果。
- `leaderboard`: 按 accuracy 和 source recall 排序的简表。

正式报告建议至少列：

```text
method | accuracy | source_recall@k | source_mrr@k | memory_chunks | approx_memory_tokens
```
