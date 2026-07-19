"""Co-Training V2 Orchestrator for alternating attacker/defender training."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .online_attacker_trainer import OnlineAttackerTrainer
from .online_memory_trainer import OnlineMemoryTrainer

logger = logging.getLogger(__name__)


class CoTrainV2Orchestrator:
    """Manages alternating attacker and defender training rounds."""

    def __init__(
        self,
        config: Dict[str, Any],
        attacker_init_path: str,
        defender_init_path: str,
        train_graphs: List[str],
        val_graphs: List[str],
        output_dir: str,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_graphs = train_graphs
        self.val_graphs = val_graphs

        # Initial model checkpoints
        self.attacker_checkpoint = attacker_init_path
        self.defender_checkpoint = defender_init_path

        # Training state
        self.current_round = 0
        self.eval_history = []

        logger.info("=" * 70)
        logger.info("Co-Training V2 Orchestrator Initialized")
        logger.info("=" * 70)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Training graphs: {len(train_graphs)}")
        logger.info(f"Validation graphs: {len(val_graphs)}")
        logger.info(f"Attacker init: {attacker_init_path}")
        logger.info(f"Defender init: {defender_init_path}")

    def run_cotrain(self):
        """Execute full co-training loop."""
        num_rounds = self.config.get("cotrain_rounds", 10)

        logger.info(f"\nStarting {num_rounds} co-training rounds")

        for round_num in range(num_rounds):
            self.current_round = round_num
            logger.info("\n" + "=" * 70)
            logger.info(f"Co-Training Round {round_num + 1}/{num_rounds}")
            logger.info("=" * 70)

            # Phase 1: Train Attacker
            logger.info(f"\n[Phase 1/3] Training Attacker...")
            attacker_output = self.output_dir / f"attacker_round_{round_num + 1}"
            self._train_attacker(attacker_output)

            # Update attacker checkpoint
            self.attacker_checkpoint = self._get_latest_checkpoint(attacker_output)
            logger.info(f"Attacker checkpoint: {self.attacker_checkpoint}")

            # Phase 2: Train Defender
            logger.info(f"\n[Phase 2/3] Training Defender...")
            defender_output = self.output_dir / f"defender_round_{round_num + 1}"
            self._train_defender(defender_output)

            # Update defender checkpoint
            self.defender_checkpoint = self._get_latest_checkpoint(defender_output)
            logger.info(f"Defender checkpoint: {self.defender_checkpoint}")

            # Phase 3: Evaluation
            logger.info(f"\n[Phase 3/3] Evaluation...")
            eval_results = self._evaluate_round(round_num + 1)
            self.eval_history.append(eval_results)

            # Save round manifest
            self._save_round_manifest(round_num + 1, eval_results)

            # Check convergence
            if self._check_convergence(eval_results):
                logger.info(f"\nConverged at round {round_num + 1}!")
                break

        logger.info("\n" + "=" * 70)
        logger.info("Co-Training Complete!")
        logger.info("=" * 70)

        # Save final results
        self._save_final_results()

    def _train_attacker(self, output_dir: Path):
        """Train attacker with frozen defender memory states."""
        # Get defender memory directory from previous round
        if self.current_round == 0:
            defender_memory_dir = None  # No memory for first round
        else:
            prev_defender_dir = self.output_dir / f"defender_round_{self.current_round}"
            defender_memory_dir = str(prev_defender_dir / "checkpoint_final" / "memory_states")

        attacker_config = self.config.get("attacker", {})

        trainer = OnlineAttackerTrainer(
            config=attacker_config,
            attacker_model_path=self.attacker_checkpoint,
            defender_memory_dir=defender_memory_dir,
            graph_files=self.train_graphs,
            output_dir=str(output_dir),
        )

        trainer.train()

    def _train_defender(self, output_dir: Path):
        """Train defender with updated attacker."""
        # Load memory states from previous round
        if self.current_round == 0:
            initial_memory_dir = None
        else:
            prev_defender_dir = self.output_dir / f"defender_round_{self.current_round}"
            initial_memory_dir = str(prev_defender_dir / "checkpoint_final" / "memory_states")

        defender_config = self.config.get("defender", {})

        # Configure defender to use trained attacker
        # In full implementation, would deploy attacker as vLLM server
        # For now, defender uses its own question generation

        trainer = OnlineMemoryTrainer(
            config=defender_config,
            model_path=self.defender_checkpoint,
            graph_files=self.train_graphs,
            output_dir=str(output_dir),
            val_graph_files=self.val_graphs,
            initial_memory_dir=initial_memory_dir,
        )

        trainer.train()

    def _evaluate_round(self, round_num: int) -> Dict[str, Any]:
        """Evaluate both models on validation set."""
        # Placeholder evaluation
        # In full implementation, would:
        # 1. Evaluate defender accuracy on target questions
        # 2. Measure memory compression
        # 3. Analyze attacker question quality

        results = {
            "round": round_num,
            "attacker": {
                "checkpoint": str(self.attacker_checkpoint),
            },
            "defender": {
                "checkpoint": str(self.defender_checkpoint),
                "accuracy": 0.0,  # Placeholder
                "compression_ratio": 0.0,  # Placeholder
            },
        }

        logger.info(f"\nRound {round_num} Results:")
        logger.info(f"  Attacker: {results['attacker']['checkpoint']}")
        logger.info(f"  Defender: {results['defender']['checkpoint']}")

        return results

    def _check_convergence(self, results: Dict[str, Any]) -> bool:
        """Check if training has converged."""
        # Simple convergence check
        # In full implementation, would check:
        # - Defender accuracy threshold
        # - Memory compression target
        # - Stability across recent rounds

        target_accuracy = self.config.get("evaluation", {}).get("target_accuracy_threshold", 0.85)
        current_accuracy = results["defender"]["accuracy"]

        return current_accuracy >= target_accuracy

    def _save_round_manifest(self, round_num: int, eval_results: Dict):
        """Save manifest for this round."""
        manifest = {
            "round": round_num,
            "attacker_checkpoint": str(self.attacker_checkpoint),
            "defender_checkpoint": str(self.defender_checkpoint),
            "evaluation": eval_results,
        }

        manifest_path = self.output_dir / f"round_{round_num}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved round manifest: {manifest_path}")

    def _save_final_results(self):
        """Save final training results."""
        final_results = {
            "total_rounds": self.current_round + 1,
            "final_attacker": str(self.attacker_checkpoint),
            "final_defender": str(self.defender_checkpoint),
            "eval_history": self.eval_history,
        }

        results_path = self.output_dir / "final_results.json"
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"Saved final results: {results_path}")

    def _get_latest_checkpoint(self, output_dir: Path) -> str:
        """Get latest checkpoint from training output."""
        checkpoint_dir = output_dir / "verl_checkpoints"

        if not checkpoint_dir.exists():
            logger.warning(f"No checkpoints found in {checkpoint_dir}")
            return str(output_dir)

        # Find latest global_step directory
        step_dirs = sorted(
            checkpoint_dir.glob("global_step_*"),
            key=lambda p: int(p.name.split("_")[-1])
        )

        if not step_dirs:
            return str(output_dir)

        latest = step_dirs[-1] / "actor"
        return str(latest) if latest.exists() else str(output_dir)
