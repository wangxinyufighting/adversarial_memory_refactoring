"""Attacker trainer that wraps verl PPO with custom dataset and frozen defender."""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OnlineAttackerTrainer:
    """Trains attacker policy with GRPO using frozen defender memory states."""

    def __init__(
        self,
        config: Dict[str, Any],
        attacker_model_path: str,
        defender_memory_dir: Optional[str],
        graph_files: List[str],
        output_dir: str,
    ):
        self.config = config
        self.attacker_model_path = attacker_model_path
        self.defender_memory_dir = defender_memory_dir
        self.graph_files = graph_files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing attacker trainer")
        logger.info(f"  Model: {attacker_model_path}")
        logger.info(f"  Graphs: {len(graph_files)} files")
        logger.info(f"  Output: {output_dir}")

    def train(self):
        """Run GRPO training for attacker using verl."""
        logger.info("=" * 60)
        logger.info("Starting Attacker GRPO Training")
        logger.info("=" * 60)

        num_epochs = self.config.get("num_epochs", 1)
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size: {self.config.get('train_batch_size', 16)}")
        logger.info(f"Rollout N: {self.config.get('rollout_n', 8)}")

        # Build dataset config
        dataset_config = self._build_dataset_config()
        dataset_config_path = self.output_dir / "attacker_dataset_config.json"
        with open(dataset_config_path, "w") as f:
            json.dump(dataset_config, f, indent=2)

        logger.info(f"Dataset config: {dataset_config_path}")

        # Run verl training
        self._run_verl_training(str(dataset_config_path))

        logger.info("Attacker training complete!")

    def _build_dataset_config(self) -> Dict[str, Any]:
        """Build config for OnlineAttackerDataset."""
        return {
            "graph_files": self.graph_files,
            "defender_memory_dir": self.defender_memory_dir,
            "config": {
                "episodes_per_case": self.config.get("episodes_per_case", 100),
                "routing_policy": self.config.get("routing_policy", "random_walk"),
                "routing_attempts": self.config.get("routing_attempts", 12),
                "routing_min_nodes": self.config.get("routing_min_nodes", 1),
                "routing_max_steps": self.config.get("routing_max_steps", 4),
            },
        }

    def _run_verl_training(self, dataset_config_path: str):
        """Launch verl PPO trainer with custom attacker dataset."""
        model_dtype = self.config.get("model_dtype", "bfloat16")
        rollout_dtype = self.config.get("rollout_dtype", model_dtype)
        attn_implementation = self.config.get("attn_implementation", "flash_attention_2")

        verl_args = [
            sys.executable, "-m", "verl.trainer.main_ppo",
            # Algorithm
            f"algorithm.adv_estimator={self.config.get('adv_estimator', 'grpo')}",
            f"algorithm.use_kl_in_reward=False",
            # Data - custom attacker dataset
            f"data.train_files={dataset_config_path}",
            f"data.val_files={dataset_config_path}",
            f"data.train_batch_size={self.config.get('train_batch_size', 16)}",
            f"data.val_batch_size={self.config.get('train_batch_size', 16)}",
            f"data.max_prompt_length={self.config.get('max_prompt_length', 4096)}",
            f"data.max_response_length={self.config.get('max_response_length', 512)}",
            "data.dataloader_num_workers=0",
            "data.shuffle=False",
            "data.filter_overlong_prompts=False",
            "data.custom_cls.path=pkg://case_graph.online_attacker_dataset",
            "data.custom_cls.name=OnlineAttackerDataset",
            # Model
            f"actor_rollout_ref.model.path={self.attacker_model_path}",
            f"+actor_rollout_ref.model.override_config.attn_implementation={attn_implementation}",
            f"actor_rollout_ref.actor.fsdp_config.model_dtype={model_dtype}",
            f"actor_rollout_ref.ref.fsdp_config.model_dtype={model_dtype}",
            f"actor_rollout_ref.actor.optim.lr={self.config.get('actor_lr', 1e-6)}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={self.config.get('ppo_mini_batch_size', 8)}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={self.config.get('ppo_micro_batch_size_per_gpu', 1)}",
            # Rollout
            f"actor_rollout_ref.rollout.name={self.config.get('infer_backend', 'vllm')}",
            f"actor_rollout_ref.rollout.n={self.config.get('rollout_n', 8)}",
            f"actor_rollout_ref.rollout.dtype={rollout_dtype}",
            f"actor_rollout_ref.rollout.temperature={self.config.get('temperature', 1.0)}",
            f"actor_rollout_ref.rollout.prompt_length={self.config.get('max_prompt_length', 4096)}",
            f"actor_rollout_ref.rollout.response_length={self.config.get('max_response_length', 512)}",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.config.get('rollout_tp', 1)}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={self.config.get('rollout_gpu_memory_utilization', 0.6)}",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={self.config.get('log_prob_micro_batch_size_per_gpu', 1)}",
            # Reference model
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={self.config.get('log_prob_micro_batch_size_per_gpu', 1)}",
            # Custom reward function
            "reward.custom_reward_function.path=pkg://case_graph.attacker_grpo_adapter",
            "reward.custom_reward_function.name=compute_attacker_score",
            f"reward.reward_manager.name={self.config.get('reward_manager', 'naive')}",
            f"reward.num_workers={self.config.get('reward_num_workers', 4)}",
            # Trainer
            f"trainer.project_name={self.config.get('project_name', 'cotrain_v2')}",
            f"trainer.experiment_name={self.config.get('experiment_name', 'attacker_training')}",
            f"trainer.logger={self._hydra_list(self.config.get('trainer_logger', ['console']))}",
            f"trainer.use_v1={self.config.get('use_v1', True)}",
            f"trainer.n_gpus_per_node={self.config.get('n_gpus_per_node', 1)}",
            f"trainer.nnodes={self.config.get('nnodes', 1)}",
            f"trainer.total_epochs={self.config.get('num_epochs', 1)}",
            f"trainer.save_freq={self.config.get('save_freq', 100)}",
            f"trainer.test_freq={self.config.get('test_freq', 500)}",
            "trainer.val_before_train=False",
            f"trainer.default_local_dir={str((self.output_dir / 'verl_checkpoints').resolve())}",
        ]

        logger.info(f"Launching verl with {len(verl_args)} arguments")

        try:
            result = subprocess.run(verl_args, check=True, cwd=str(Path.cwd()))
            logger.info("verl training completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"verl training failed with exit code {e.returncode}")
            raise

    @staticmethod
    def _hydra_list(value: Any) -> str:
        """Convert list to hydra format."""
        if isinstance(value, (list, tuple)):
            return "[" + ",".join(str(item) for item in value) + "]"
        return str(value)
