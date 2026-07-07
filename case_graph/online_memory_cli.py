"""CLI entry point for online GRPO memory refactoring training."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import yaml

from .online_memory_dataset import OnlineMemoryDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def list_graph_files(graphs_dir: str) -> List[str]:
    """List all case graph JSON files in directory."""
    graph_dir = Path(graphs_dir)

    if graph_dir.is_file():
        return [str(graph_dir)]

    if not graph_dir.is_dir():
        raise ValueError(f"Graphs directory not found: {graphs_dir}")

    graph_files = sorted(graph_dir.glob("*.case_graph.json"))

    if not graph_files:
        raise ValueError(f"No case graph files found in {graphs_dir}")

    logger.info(f"Found {len(graph_files)} case graph files in {graphs_dir}")
    return [str(f) for f in graph_files]


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def merge_config(base_config: dict, args: argparse.Namespace) -> dict:
    """Merge command-line args into base config."""
    # Command-line args override config file
    if args.rollout_n is not None:
        base_config["rollout_n"] = args.rollout_n
    if args.train_batch_size is not None:
        base_config["train_batch_size"] = args.train_batch_size
    if args.ppo_mini_batch_size is not None:
        base_config["ppo_mini_batch_size"] = args.ppo_mini_batch_size
    if args.total_epochs is not None:
        base_config["num_epochs"] = args.total_epochs
    if args.tau is not None:
        base_config["tau"] = args.tau
    if args.top_k is not None:
        base_config["top_k"] = args.top_k
    if args.episodes_per_case is not None:
        base_config["episodes_per_case"] = args.episodes_per_case
    if args.seed is not None:
        base_config["seed"] = args.seed
    if args.max_prompt_length is not None:
        base_config["max_prompt_length"] = args.max_prompt_length
    if args.max_response_length is not None:
        base_config["max_response_length"] = args.max_response_length

    return base_config


def main():
    parser = argparse.ArgumentParser(
        description="Online GRPO training for memory refactoring"
    )

    # Required arguments
    parser.add_argument(
        "--graphs",
        required=True,
        help="Directory containing case graph JSON files or single graph file"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to base model (e.g., Qwen/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for checkpoints and memory states"
    )

    # Optional arguments
    parser.add_argument(
        "--config",
        default="configs/online_grpo.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--initial-memory-dir",
        default=None,
        help="Directory with initial memory states (optional)"
    )

    # Training hyperparameters
    parser.add_argument("--rollout-n", type=int, help="Number of rollouts per state")
    parser.add_argument("--train-batch-size", type=int, help="Training batch size")
    parser.add_argument("--ppo-mini-batch-size", type=int, help="PPO mini-batch size")
    parser.add_argument("--total-epochs", type=int, help="Total training epochs")

    # Environment parameters
    parser.add_argument("--tau", type=float, help="Add/merge similarity threshold")
    parser.add_argument("--top-k", type=int, help="Top-K retrieval")
    parser.add_argument("--episodes-per-case", type=int, help="Episodes per case graph")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Model parameters
    parser.add_argument("--max-prompt-length", type=int, help="Max prompt tokens")
    parser.add_argument("--max-response-length", type=int, help="Max response tokens")

    # verl backend
    parser.add_argument("--infer-backend", default="vllm", help="Inference backend")
    parser.add_argument("--rollout-tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--rollout-gpu-memory-utilization",
        type=float,
        default=0.6,
        help="GPU memory utilization"
    )

    # Logging
    parser.add_argument("--project-name", help="Wandb project name")
    parser.add_argument("--experiment-name", help="Experiment name")

    args = parser.parse_args()

    # Load and merge config
    base_config = load_config(args.config)
    config = merge_config(base_config, args)

    # List graph files
    graph_files = list_graph_files(args.graphs)

    logger.info("=" * 60)
    logger.info("Online GRPO Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Number of graphs: {len(graph_files)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Rollout N: {config.get('rollout_n', 4)}")
    logger.info(f"Batch size: {config.get('train_batch_size', 4)}")
    logger.info(f"Episodes per case: {config.get('episodes_per_case', 100)}")
    logger.info(f"Seed: {config.get('seed', 42)}")
    logger.info("=" * 60)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save merged config
    config_output = output_path / "training_config.yaml"
    with open(config_output, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved training config to {config_output}")

    # Initialize dataset
    logger.info("Initializing online memory dataset...")
    try:
        dataset = OnlineMemoryDataset(
            graph_files=graph_files,
            config=config,
            initial_memory_dir=args.initial_memory_dir,
        )
        logger.info(f"Dataset initialized with {len(dataset.graphs)} valid graphs")
    except Exception as e:
        logger.error(f"Failed to initialize dataset: {e}")
        sys.exit(1)

    # Save merged config
    config_output = output_path / "training_config.yaml"
    with open(config_output, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved training config to {config_output}")

    # Initialize trainer
    logger.info("\n" + "=" * 60)
    logger.info("Initializing Online Memory Trainer")
    logger.info("=" * 60)

    from .online_memory_trainer import OnlineMemoryTrainer

    try:
        trainer = OnlineMemoryTrainer(
            config=config,
            model_path=args.model_path,
            graph_files=graph_files,
            output_dir=args.output_dir,
            initial_memory_dir=args.initial_memory_dir,
        )
        logger.info("Trainer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run training
    logger.info("\n" + "=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)

    try:
        trainer.train()
        logger.info("\n" + "=" * 60)
        logger.info("Training Completed Successfully!")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
