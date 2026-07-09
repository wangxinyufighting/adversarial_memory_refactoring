# Adversarial Attacker/Defender Co-Training

第一版实现目标：

- Graph Routing Policy 固定，不训练。
- attacker 和 defender 分开训练、分开保存 checkpoint。
- attacker 不读取 defender logits、hidden states、gradients、advantages 或训练 batch。
- attacker 只看 route evidence、当前 compressed memory view、retrieval observation、历史 mistake examples。
- attacker reward 的目标是构造更有助于 memory refactoring 的问题，而不是单纯为难 defender。

## 启动

```bash
./scripts/run_adversarial_memory_cotrain.sh
```

常用覆盖：

```bash
GRAPHS_DIR=outputs/case_graphs_train \
ATTACKER_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
DEFENDER_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
OUTPUT_DIR=outputs/adversarial_cotrain \
COTRAIN_ROUNDS=2 \
./scripts/run_adversarial_memory_cotrain.sh
```

配置文件：

```text
configs/adversarial_cotrain.yaml
```

## 输出结构

每一轮都会分开保存：

```text
outputs/adversarial_cotrain/
  attacker_ckpt_t000/
    verl_checkpoints/
  defender_ckpt_t000/
    verl_checkpoints/
    checkpoint_latest/memory_states/
    checkpoint_step*/memory_states/
  cotrain_manifest.json
```

下一轮 attacker 通过 `trainer.resume_from_path` 接续自己的 attacker checkpoint；
defender 通过 `trainer.resume_from_path` 接续自己的 defender checkpoint。

## Attacker Reward

入口：

```text
case_graph/attacker_grpo_adapter.py
```

reward 组件：

- `format_valid`: 输出必须是 JSON，并包含 question/answer。
- `grounded`: answer 必须能从 route evidence 或 golden facts 支持。
- `oracle_solvable`: grounded 即可由 oracle 验证为可解。
- `memory_gap`: 当前 compressed memory 的 retrieval top-k 不能直接支持 answer。
- `question_relevance`: question 必须和 route evidence 有词面相关性。
- `novelty`: 避免重复 mistake book 中已有问题。
- `route_complexity`: 鼓励合理多实体/多关系问题。
- penalties: answer 泄漏在 question 中、过短、过长。

因此 attacker 的高分样本是：

```text
有根据、可回答、当前 memory 不容易答出、能推动 defender 补 memory 的问题
```

而不是：

```text
幻觉问题、不可验证问题、歧义问题、单纯让 defender 失败的问题
```

## Defender 训练

defender 继续使用现有 online memory GRPO：

```text
case_graph/online_memory_dataset.py
case_graph/grpo_adapter.py
case_graph/online_memory_trainer.py
```

co-training 编排入口：

```text
case_graph/adversarial_cotrain_trainer.py
```

defender phase 会使用当前 attacker phase 后的 attacker model name/path 作为
`attacker_llm` 传入本地 OpenAI-compatible attacker API。

默认配置会自动管理 attacker server：

```text
train attacker
-> 找到 attacker_ckpt_t*/verl_checkpoints/global_step_*/actor
-> 用 verl.model_merger merge 成 HuggingFace 格式
-> 重启 vLLM OpenAI API server
-> 等待 http://localhost:8003/v1/models 返回 attacker-current
-> defender phase 使用 attacker_llm=attacker-current
```

相关配置：

```yaml
manage_attacker_server: true
attacker_api_base: "http://localhost:8003/v1"
attacker_served_model: "attacker-current"
attacker_server:
  enabled: true
  port: 8003
  checkpoint_backend: "fsdp"
  checkpoint_subdir: "actor"
  dtype: bfloat16
```

如果你想手动管理 vLLM，可以设置：

```bash
MANAGE_ATTACKER_SERVER=false ./scripts/run_adversarial_memory_cotrain.sh
```
