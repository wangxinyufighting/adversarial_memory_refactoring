# Online GRPO 代码流程说明

本文档说明 `./scripts/run_online_memory_grpo.sh` 启动后的主要算法流程、关键代码文件，以及新增的 target-question evaluation 入口。

## 1. 启动入口

### `scripts/run_online_memory_grpo.sh`

脚本负责收集环境变量并启动 Python CLI：

- `GRAPHS_DIR`：训练用 CaseGraph 目录，默认 `outputs/case_graphs_train`
- `MODEL_PATH`：被 GRPO 训练的 policy/actor 模型
- `OUTPUT_DIR`：训练输出目录，包含 verl checkpoint 和 online memory checkpoint
- `CONFIG_FILE`：默认 `configs/online_grpo.yaml`
- `ATTACKER_LLM` / `ATTACKER_API_BASE` / `ATTACKER_API_KEY`：在线生成 attack 的冻结 LLM，默认指向本地 vLLM `http://localhost:8003/v1`

脚本最终执行：

```bash
python3 -m case_graph.online_memory_cli ...
```

## 2. 配置加载与合并

### `case_graph/online_memory_cli.py`

核心函数：

- `load_config()`：读取 YAML 配置
- `merge_config()`：命令行参数覆盖 YAML，包括 rollout、batch、tau、attacker、本地 vLLM、项目名等
- `list_graph_files()`：收集 `*.case_graph.json`
- `main()`：创建 `OnlineMemoryTrainer` 并调用 `trainer.train()`

注意：`attacker_llm` 和 `attacker_api_base` 已经从配置/命令行进入 `OnlineMemoryDataset`，不会再被忽略。

## 3. Online Dataset 与 Environment

### `case_graph/online_memory_dataset.py`

这是 online GRPO 的核心环境层。

#### `OnlineMemoryDataset`

verl 会把它当作 PyTorch Dataset 使用，每次 `__getitem__()` 动态产生一个训练状态：

1. 轮询选择一个 case graph
2. 为该 case 生成本轮 episode seed
3. 调用 `OnlineMemoryEnvironment.generate_episode()`
4. 如果初始 defense 已能答对，则跳过并继续采样
5. 如果需要重构，则转成 verl row

关键字段：

- `self.env`：保存所有 case 的在线 memory 状态
- `self.case_rotation`：轮询 case
- `self.episodes_per_case`：虚拟数据集长度控制
- `self.commit_threshold`：best rollout reward 大于该值才 commit
- `self.output_dir`：保存 online memory checkpoint

#### `build_attacker_from_config()`

根据 config 构建冻结 attacker：

- 如果设置了 `attacker_llm` 或 `attacker_api_base`，直接创建 `OpenAIChatClient`
- 默认使用 `attacker_api_base=http://localhost:8003/v1`
- 否则退回原来的 `OpenAIChatClient.from_env()`

这修复了本地 attacker vLLM 没有被调用的问题。

#### `OnlineMemoryEnvironment`

为每个 case 维护独立状态：

- `memory_store`：当前压缩记忆 `M_t`
- `success_pool`：历史已成功回答的问题，用于回归测试
- `high_priority_buffer`：被 rollback 或失败的问题
- `episode_count`：该 case 已采样 episode 数

`generate_episode()` 的流程：

1. `RandomWalkRoutingPolicy.select_route(graph, seed=seed)` 选择攻击路径
2. `FrozenLLMAttacker.generate()` 用冻结 LLM 生成 adversarial question/answer
3. `prepare_refactor_state()` 做初始 defense、Add/Merge 决策、回归问题采样
4. 返回状态 `S_t = (M_t, Q, A, F, action, regression_set)`

`commit_memory_update()` 会把 winning proposal 应用到 `M_t`，并更新 `success_pool`。

#### `on_batch_end()`

这是 online 的关键闭环。verl 每个 policy update 完成后调用：

1. 从 batch/TransferQueue 读取所有 rollout 的 response、reward、ground_truth
2. 按 `extra_info.uid` 聚合成同一个 GRPO group
3. 选择 reward 最高的 proposal
4. `best_reward > commit_threshold`：解析 JSON proposal 并 commit 到 `M_t`
5. 否则写入 `high_priority_buffer`
6. 保存 `checkpoint_latest/memory_states`，并按 `memory_checkpoint_interval` 保存 step checkpoint

## 4. 训练 Orchestrator

### `case_graph/online_memory_trainer.py`

`OnlineMemoryTrainer._run_verl_training()` 将 online dataset 注册给 verl：

- `data.custom_cls.path=pkg://case_graph.online_memory_dataset`
- `data.custom_cls.name=OnlineMemoryDataset`
- `reward.custom_reward_function.path=pkg://case_graph.grpo_adapter`
- `reward.custom_reward_function.name=compute_score`

重要设置：

- `data.dataloader_num_workers=0`：避免多个 dataloader worker 各自复制一份 memory state
- `data.shuffle=False`：保持 online 环境采样顺序可控
- `trainer.val_before_train=False`：避免 validation 在训练前提前消耗/变异 online 状态
- `trainer.default_local_dir=${OUTPUT_DIR}/verl_checkpoints`

训练输出包含两类内容：

- verl 模型 checkpoint：`OUTPUT_DIR/verl_checkpoints`
- online memory checkpoint：`OUTPUT_DIR/checkpoint_latest/memory_states` 和 `checkpoint_final/memory_states`

## 5. verl Hook

### `verl/verl/trainer/ppo/v1/trainer_base.py`

V1 trainer 在每步训练后、清理 TransferQueue 前调用：

```python
self.train_dataset.on_batch_end(batch=batch, tokenizer=self.tokenizer)
```

训练结束时调用：

```python
self.train_dataset.on_train_end()
```

### `verl/verl/trainer/ppo/ray_trainer.py`

旧版 trainer 也加入同样 hook，保证 `use_v1=true/false` 都能更新 online memory。

## 6. Reward 与训练样本格式

### `case_graph/grpo_adapter.py`

`build_verl_row_online()` 把在线状态写入 verl row：

- `prompt`：system + user prompt，要求 policy 输出 JSON chunks
- `reward_model.ground_truth`：保存 reward 需要的最小状态
- `extra_info.uid`：online commit 聚合使用的稳定 UID

`ground_truth` 现在包含：

- `current_memory`
- `question`
- `answer`
- `golden_facts`
- `action`
- `selected_memory_ids`
- `regression_questions`
- `top_k`
- `uid`

`compute_score()` 的流程：

1. 解析 policy 输出 JSON
2. 构造 `RefactorProposal`
3. 用 `build_sandbox_memory()` 在沙盒里应用 proposal，得到 `M_temp`
4. 用 `FrozenBM25Retriever` 对当前问题和 regression questions 检索
5. 用 answer-presence 规则判断答案是否可由 retrieved memories 支撑
6. `compute_reward()` 计算当前题正确性、回归正确率、遗忘惩罚、chunk 数和长度惩罚

## 7. 新增 Evaluation

### `case_graph/evaluation.py`

用于全新 case 的 target-question evaluation：

1. 读取 CaseGraph 的 `target.question` 和 `target.answer`
2. 读取 compressed memory
3. 使用已有 `FrozenBM25Retriever` 检索 top-k memory chunks
4. 使用 `RetrievedMemoryAnswerAgent` 调 LLM 回答 target question
5. 使用 `AnswerEquivalenceJudge` 判分
6. 输出 per-case 结果和整体 accuracy

### `case_graph/evaluation_cli.py`

命令行入口：

```bash
python3 -m case_graph.evaluation_cli \
  --graphs outputs/case_graphs_test \
  --memory-dir outputs/online_grpo/checkpoint_final/memory_states \
  --output outputs/eval_target_questions.json \
  --top-k 5
```

如果所有 case 共用一个 memory 文件，也可以：

```bash
python3 -m case_graph.evaluation_cli \
  --graphs outputs/case_graphs_test \
  --memory outputs/final_memory.json \
  --output outputs/eval_target_questions.json
```

### `scripts/evaluate_target_questions.sh`

Shell 包装入口：

```bash
./scripts/evaluate_target_questions.sh \
  --graphs outputs/case_graphs_test \
  --memory-dir outputs/online_grpo/checkpoint_final/memory_states \
  --output outputs/eval_target_questions.json
```

LLM backend 仍使用项目原有 `OpenAIChatClient.from_env()`，可通过：

```bash
export CASE_GRAPH_PROVIDER=local
export LOCAL_API_BASE_URL=http://localhost:8003/v1
export LOCAL_MODEL=/mnt/local2/wxy/models/Qwen3-0.6B
```

## 8. 端到端算法总览

```text
CaseGraph files
  -> OnlineMemoryDataset.__getitem__
  -> RandomWalkRoutingPolicy selects route
  -> FrozenLLMAttacker generates Q/A
  -> prepare_refactor_state builds S_t
  -> verl rollout generates N JSON proposals
  -> grpo_adapter.compute_score evaluates each proposal in sandbox
  -> verl GRPO updates policy
  -> OnlineMemoryDataset.on_batch_end picks best rollout
  -> commit proposal to per-case M_t or rollback
  -> save compressed memory checkpoint
  -> evaluation_cli retrieves from compressed memory and answers target question
```
