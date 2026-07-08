"""Trainer wrapper for the trainable memory attacker policy."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .online_attacker_dataset import OnlineAttackerDataset

logger = logging.getLogger(__name__)


class AttackerGRPOTrainer:
    """Run verl GRPO for the attacker policy in its own checkpoint tree."""

    def __init__(
        self,
        config: Dict[str, Any],
        model_path: str,
        graph_files: List[str],
        output_dir: str,
        initial_memory_dir: Optional[str] = None,
    ):
        self.config = dict(config)
        self.model_path = model_path
        self.graph_files = graph_files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.initial_memory_dir = initial_memory_dir
        self.config.setdefault("output_dir", str(self.output_dir))
        logger.info("Initializing attacker dataset...")
        self.dataset = OnlineAttackerDataset(
            graph_files=graph_files,
            config=self.config,
            initial_memory_dir=initial_memory_dir,
        )
        logger.info("Attacker dataset initialized with %d graphs", len(self.dataset.graphs))

    def train(self) -> None:
        logger.info("=" * 60)
        logger.info("Starting Attacker GRPO Training")
        logger.info("=" * 60)
        self._run_verl_training()

    def _run_verl_training(self) -> None:
        dataset_config_file = self.output_dir / "attacker_dataset_config.json"
        graph_file_paths = []
        for graph in self.dataset.graphs:
            case_id = graph.get("case_id", "unknown")
            temp_graph_file = self.output_dir / f"attacker_temp_graph_{case_id}.json"
            temp_graph_file.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
            graph_file_paths.append(str(temp_graph_file))

        dataset_runtime_config = dict(self.config)
        dataset_runtime_config["output_dir"] = str(self.output_dir)
        dataset_config = {
            "graph_files": graph_file_paths,
            "config": dataset_runtime_config,
            "initial_memory_dir": self.initial_memory_dir,
        }
        dataset_config_file.write_text(
            json.dumps(dataset_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        model_dtype = self.config.get("model_dtype", "bfloat16")
        rollout_dtype = self.config.get("rollout_dtype", model_dtype)
        attn_implementation = self.config.get("attn_implementation", "flash_attention_2")
        trainer_logger = _hydra_list(self.config.get("trainer_logger", ["console"]))
        save_freq = self.config.get("attacker_save_freq", self.config.get("save_freq", 100))
        resume_from_path = self.config.get("resume_from_path")
        train_batch_size = self.config.get("attacker_train_batch_size", self.config.get("train_batch_size", 4))
        ppo_mini_batch_size = self.config.get(
            "attacker_ppo_mini_batch_size",
            self.config.get("ppo_mini_batch_size", 2),
        )
        rollout_n = self.config.get("attacker_rollout_n", self.config.get("rollout_n", 4))
        total_epochs = self.config.get("attacker_num_epochs", self.config.get("num_epochs", 1))

        verl_args = [
            sys.executable,
            "-m",
            "verl.trainer.main_ppo",
            "algorithm.adv_estimator=grpo",
            "algorithm.use_kl_in_reward=False",
            f"data.train_files={str(dataset_config_file.resolve())}",
            f"data.val_files={str(dataset_config_file.resolve())}",
            f"data.train_batch_size={train_batch_size}",
            f"data.val_batch_size={train_batch_size}",
            f"data.max_prompt_length={self.config.get('attacker_max_prompt_length', self.config.get('max_prompt_length', 8192))}",
            f"data.max_response_length={self.config.get('attacker_max_response_length', self.config.get('max_response_length', 512))}",
            "data.dataloader_num_workers=0",
            "data.shuffle=False",
            "data.filter_overlong_prompts=False",
            "data.custom_cls.path=pkg://case_graph.online_attacker_dataset",
            "data.custom_cls.name=OnlineAttackerDataset",
            f"actor_rollout_ref.model.path={self.model_path}",
            f"+actor_rollout_ref.model.override_config.attn_implementation={attn_implementation}",
            f"actor_rollout_ref.actor.fsdp_config.model_dtype={model_dtype}",
            f"actor_rollout_ref.ref.fsdp_config.model_dtype={model_dtype}",
            f"actor_rollout_ref.actor.optim.lr={self.config.get('attacker_lr', self.config.get('actor_lr', 1e-6))}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={self.config.get('ppo_micro_batch_size_per_gpu', 1)}",
            f"actor_rollout_ref.rollout.name={self.config.get('infer_backend', 'vllm')}",
            f"actor_rollout_ref.rollout.n={rollout_n}",
            f"actor_rollout_ref.rollout.dtype={rollout_dtype}",
            "actor_rollout_ref.rollout.val_kwargs.n=1",
            f"actor_rollout_ref.rollout.temperature={self.config.get('attacker_temperature', self.config.get('temperature', 1.0))}",
            f"actor_rollout_ref.rollout.prompt_length={self.config.get('attacker_max_prompt_length', self.config.get('max_prompt_length', 8192))}",
            f"actor_rollout_ref.rollout.response_length={self.config.get('attacker_max_response_length', self.config.get('max_response_length', 512))}",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.config.get('rollout_tp', 1)}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={self.config.get('rollout_gpu_memory_utilization', 0.6)}",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={self.config.get('log_prob_micro_batch_size_per_gpu', 1)}",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={self.config.get('log_prob_micro_batch_size_per_gpu', 1)}",
            "reward.custom_reward_function.path=pkg://case_graph.attacker_grpo_adapter",
            "reward.custom_reward_function.name=compute_attacker_score",
            f"reward.reward_manager.name={self.config.get('reward_manager', 'naive')}",
            f"reward.num_workers={self.config.get('reward_num_workers', 4)}",
            f"trainer.project_name={self.config.get('project_name', 'memory_attacker_grpo')}",
            f"trainer.experiment_name={self.config.get('experiment_name', 'attacker_online')}",
            f"trainer.logger={trainer_logger}",
            f"trainer.use_v1={self.config.get('use_v1', True)}",
            f"trainer.n_gpus_per_node={self.config.get('n_gpus_per_node', 1)}",
            f"trainer.nnodes={self.config.get('nnodes', 1)}",
            f"trainer.total_epochs={total_epochs}",
            f"trainer.save_freq={save_freq}",
            f"trainer.test_freq={self.config.get('test_freq', -1)}",
            "trainer.val_before_train=False",
            f"trainer.default_local_dir={str((self.output_dir / 'verl_checkpoints').resolve())}",
        ]
        if resume_from_path:
            verl_args.extend(
                [
                    "trainer.resume_mode=resume_path",
                    f"trainer.resume_from_path={resume_from_path}",
                ]
            )
        logger.info("Running attacker verl wrapper with %d arguments", len(verl_args))
        env = os.environ.copy()
        env.setdefault("HYDRA_FULL_ERROR", "1")
        subprocess.run(verl_args, cwd=str(Path.cwd()), env=env, check=True)


def _hydra_list(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)
