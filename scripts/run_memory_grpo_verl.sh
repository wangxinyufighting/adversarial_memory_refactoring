#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-outputs/memory_grpo/train.parquet}
VAL_FILE=${VAL_FILE:-${TRAIN_FILE}}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
FILTER_OVERLONG_PROMPTS=${FILTER_OVERLONG_PROMPTS:-True}
DATA_TRUNCATION=${DATA_TRUNCATION:-error}
TRAINER_USE_V1=${TRAINER_USE_V1:-False}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS}" \
  data.truncation="${DATA_TRUNCATION}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  reward.custom_reward_function.path="${ROOT_DIR}/case_graph/grpo_adapter.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  trainer.project_name="${PROJECT_NAME:-memory_refactor_grpo}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-minimal_verl_grpo}" \
  trainer.logger="${TRAINER_LOGGER:-[\"console\"]}" \
  trainer.use_v1="${TRAINER_USE_V1}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE:-1}" \
  trainer.nnodes="${NNODES:-1}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.save_freq="${SAVE_FREQ:-10}" \
  trainer.test_freq="${TEST_FREQ:-10}" \
  "$@"
