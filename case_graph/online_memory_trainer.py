"""Online memory trainer that wraps verl PPO trainer with custom dataset and commit logic.

This module provides the integration layer between the online memory environment
and verl's GRPO training infrastructure.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .grpo_adapter import compute_score
from .online_memory_dataset import OnlineMemoryDataset
from .refactoring import RefactorProposal
from .retriever import MemoryChunk

logger = logging.getLogger(__name__)


class OnlineMemoryTrainer:
    """Orchestrates online GRPO training with memory environment.

    Responsibilities:
    1. Initialize custom dataset
    2. Setup verl PPO trainer
    3. Implement training loop with post-batch commit
    4. Handle checkpointing of model + memory states
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_path: str,
        graph_files: List[str],
        output_dir: str,
        initial_memory_dir: Optional[str] = None,
    ):
        self.config = config
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dataset
        logger.info("Initializing online memory dataset...")
        self.dataset = OnlineMemoryDataset(
            graph_files=graph_files,
            config=config,
            initial_memory_dir=initial_memory_dir,
        )
        logger.info(f"Dataset initialized with {len(self.dataset.graphs)} graphs")

        # verl trainer (will be initialized in train())
        self.verl_trainer = None
        self.commit_threshold = config.get("commit_threshold", 0.0)

        # Training state
        self.global_step = 0
        self.epoch = 0

    def train(self):
        """Main training loop using verl's PPO trainer."""
        num_epochs = self.config.get("num_epochs", 3)

        logger.info("=" * 60)
        logger.info("Starting Online GRPO Training")
        logger.info("=" * 60)
        logger.info(f"Total epochs: {num_epochs}")
        logger.info(f"Batch size: {self.config.get('train_batch_size', 4)}")
        logger.info(f"Rollout N: {self.config.get('rollout_n', 4)}")
        logger.info(f"Total graphs: {len(self.dataset.graphs)}")
        logger.info("=" * 60)

        # Run verl training
        self._run_verl_training()

        logger.info("\nTraining complete!")

    def _run_verl_training(self):
        """Run verl PPO training with our custom dataset."""
        try:
            import subprocess
            import sys

            logger.info("Launching verl trainer with custom dataset...")

            # Write dataset config to a temp file that our custom dataset will read
            dataset_config_file = self.output_dir / "dataset_config.json"

            # Get graph file paths from the dataset's environment
            graph_file_paths = []
            for graph in self.dataset.graphs:
                # The graphs are already loaded dicts, we need to save them temporarily
                # or reference the original files if available
                case_id = graph.get("case_id", "unknown")
                # Save each graph to temp location
                temp_graph_file = self.output_dir / f"temp_graph_{case_id}.json"
                with open(temp_graph_file, "w") as f:
                    json.dump(graph, f)
                graph_file_paths.append(str(temp_graph_file))

            dataset_config = {
                "graph_files": graph_file_paths,
                "config": self.config,
                "initial_memory_dir": None,
                "dataset_config_file": str(dataset_config_file),
            }
            with open(dataset_config_file, "w") as f:
                json.dump(dataset_config, f, indent=2)

            logger.info(f"Dataset config saved to {dataset_config_file}")

            # Convert to absolute path string for verl
            dataset_config_path = str(dataset_config_file.resolve())
            logger.info(f"Dataset config absolute path: {dataset_config_path}")

            # Build command-line arguments for verl
            verl_args = [
                sys.executable, "-m", "verl.trainer.main_ppo",
                # Algorithm
                "algorithm.adv_estimator=grpo",
                "algorithm.use_kl_in_reward=False",
                # Data - use custom dataset class
                f"data.train_files={dataset_config_path}",
                f"data.val_files={dataset_config_path}",
                f"data.train_batch_size={self.config.get('train_batch_size', 4)}",
                "data.custom_cls.path=pkg://case_graph.online_memory_dataset",
                "data.custom_cls.name=OnlineMemoryDataset",
                # Model
                f"actor_rollout_ref.model.path={self.model_path}",
                f"actor_rollout_ref.actor.optim.lr={self.config.get('actor_lr', 1e-6)}",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={self.config.get('ppo_mini_batch_size', 2)}",
                f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={self.config.get('ppo_micro_batch_size_per_gpu', 1)}",
                # Rollout
                f"actor_rollout_ref.rollout.name={self.config.get('infer_backend', 'vllm')}",
                f"actor_rollout_ref.rollout.n={self.config.get('rollout_n', 4)}",
                f"actor_rollout_ref.rollout.temperature={self.config.get('temperature', 1.0)}",
                f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.config.get('rollout_tp', 1)}",
                f"actor_rollout_ref.rollout.gpu_memory_utilization={self.config.get('rollout_gpu_memory_utilization', 0.6)}",
                f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={self.config.get('log_prob_micro_batch_size_per_gpu', 1)}",
                # Reference model
                f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={self.config.get('log_prob_micro_batch_size_per_gpu', 1)}",
                # Custom reward function
                "reward.custom_reward_function.path=pkg://case_graph.grpo_adapter",
                "reward.custom_reward_function.name=compute_score",
                "reward.reward_manager.name=naive",
                # Trainer
                f"trainer.project_name={self.config.get('project_name', 'memory_refactor_grpo_online')}",
                f"trainer.experiment_name={self.config.get('experiment_name', 'online_training')}",
                "trainer.logger=[console]",
                f"trainer.use_v1={self.config.get('use_v1', True)}",
                f"trainer.n_gpus_per_node={self.config.get('n_gpus_per_node', 1)}",
                f"trainer.nnodes={self.config.get('nnodes', 1)}",
                f"trainer.total_epochs={self.config.get('num_epochs', 3)}",
                f"trainer.save_freq={self.config.get('checkpoint_interval', 100)}",
            ]

            logger.info(f"Running verl wrapper with {len(verl_args)} arguments")

            # Run verl training via wrapper
            result = subprocess.run(
                verl_args,
                cwd=str(Path.cwd()),
                check=True,
            )

            logger.info("verl training completed")

            # Cleanup
            if dataset_config_file.exists():
                dataset_config_file.unlink()

        except subprocess.CalledProcessError as e:
            logger.error(f"verl training process failed with exit code {e.returncode}")
            raise
        except Exception as e:
            logger.error(f"verl training failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _train_epoch_placeholder(self, batch_size: int, checkpoint_interval: int):
        """Placeholder for epoch training loop (fallback when verl not available)."""
        num_batches = len(self.dataset) // batch_size

        logger.info("\n" + "=" * 60)
        logger.info("PLACEHOLDER MODE: verl not available")
        logger.info("Running dataset sampling test only")
        logger.info("=" * 60)

        for batch_idx in range(min(num_batches, 10)):  # Limit for demo
            # Simulate batch sampling
            batch_indices = [
                (batch_idx * batch_size + i) % len(self.dataset)
                for i in range(batch_size)
            ]

            try:
                batch_samples = [self.dataset[idx] for idx in batch_indices]
                logger.info(f"  Batch {batch_idx + 1}/{num_batches}: Generated {len(batch_samples)} states")

                # Extract UIDs for tracking
                uids = [sample.get("extra_info", {}).get("uid", "unknown") for sample in batch_samples]
                logger.info(f"    UIDs: {uids}")

            except Exception as e:
                logger.warning(f"  Batch {batch_idx + 1} failed: {e}")
                continue

            self.global_step += 1

            # Checkpointing
            if self.global_step % checkpoint_interval == 0:
                self.save_checkpoint()

    def post_batch_commit(self, batch_results: List[Dict[str, Any]]):
        """Commit best proposals after GRPO update.

        This method should be called by verl trainer after policy update.

        Args:
            batch_results: List of dicts with:
                - uid: Unique ID for grouping
                - rollouts: List of policy outputs (JSON strings)
                - rewards: List of rewards for each rollout
                - state: Original state dict with question, answer, etc.
        """
        for result in batch_results:
            uid = result["uid"]
            rollouts = result["rollouts"]
            rewards = result["rewards"]
            state = result.get("state", {})

            if not rollouts or not rewards:
                logger.warning(f"Skipping commit for {uid}: no rollouts or rewards")
                continue

            # Select best proposal
            best_idx = int(np.argmax(rewards))
            best_reward = rewards[best_idx]

            logger.info(f"  {uid}: best reward = {best_reward:.3f}")

            if best_reward > self.commit_threshold:
                try:
                    # Parse proposal from rollout
                    proposal = self._parse_proposal_from_rollout(
                        rollouts[best_idx], state
                    )

                    # Commit to environment
                    current_question = state.get("question", "")
                    self.dataset.commit_memory_update(uid, proposal, current_question)

                    logger.info(f"  ✓ Committed {len(proposal.new_chunks)} chunks for {uid}")

                except Exception as e:
                    logger.warning(f"  ✗ Failed to commit for {uid}: {e}")
            else:
                logger.info(f"  ✗ Rolled back {uid} (reward ≤ threshold)")

    def _parse_proposal_from_rollout(
        self, rollout_text: str, state: Dict[str, Any]
    ) -> RefactorProposal:
        """Parse RefactorProposal from policy output JSON."""
        try:
            output = json.loads(rollout_text)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            match = re.search(r'\{.*\}', rollout_text, re.DOTALL)
            if match:
                output = json.loads(match.group())
            else:
                raise ValueError("Cannot parse JSON from rollout")

        chunks = [
            MemoryChunk.from_dict(item, fallback_id=f"{state.get('action', 'add')}_{i}")
            for i, item in enumerate(output.get("chunks", []))
        ]

        if not chunks:
            raise ValueError("No chunks in rollout output")

        action = state.get("action", "add")
        if action == "add":
            chunks = chunks[:1]
            remove_ids = []
        else:  # merge
            selected_ids = state.get("selected_memory_ids", [])
            chunks = chunks[: max(1, len(selected_ids))]
            remove_ids = selected_ids

        return RefactorProposal(
            action=action,
            new_chunks=chunks,
            remove_memory_ids=remove_ids,
            metadata={"source": "online_grpo"},
        )

    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint and memory states."""
        suffix = "final" if final else f"step{self.global_step}"
        checkpoint_dir = self.output_dir / f"checkpoint_{suffix}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save memory states
        memory_dir = checkpoint_dir / "memory_states"
        self.dataset.save_memory_states(str(memory_dir))

        # verl handles model checkpointing automatically

        logger.info(f"  Checkpoint saved to {checkpoint_dir}")

        # Save training metadata
        metadata = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "num_graphs": len(self.dataset.graphs),
            "case_episodes": {
                case_id: state.episode_count
                for case_id, state in self.dataset.env.case_states.items()
            },
        }
        metadata_path = checkpoint_dir / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_checkpoint(self, checkpoint_dir: str):
        """Load checkpoint and resume training."""
        checkpoint_path = Path(checkpoint_dir)

        # Load training metadata
        metadata_path = checkpoint_path / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.global_step = metadata.get("global_step", 0)
                self.epoch = metadata.get("epoch", 0)
            logger.info(f"Resuming from step {self.global_step}, epoch {self.epoch}")

        # verl handles model loading automatically

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

