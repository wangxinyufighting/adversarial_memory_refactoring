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
    return normalize_verl_config(config or {})


def normalize_verl_config(config: dict) -> dict:
    """Mirror nested verl YAML defaults into the flat keys used by the wrapper."""
    actor_rollout_ref = config.get("verl", {}).get("actor_rollout_ref", {})
    algorithm = config.get("verl", {}).get("algorithm", {})
    reward = config.get("verl", {}).get("reward", {})
    trainer = config.get("verl", {}).get("trainer", {})

    actor = actor_rollout_ref.get("actor", {})
    model = actor_rollout_ref.get("model", {})
    rollout = actor_rollout_ref.get("rollout", {})
    ref = actor_rollout_ref.get("ref", {})

    actor_fsdp = actor.get("fsdp_config", {})
    ref_fsdp = ref.get("fsdp_config", {})

    _set_default(config, "adv_estimator", algorithm.get("adv_estimator"))
    _set_default(config, "use_kl_in_reward", algorithm.get("use_kl_in_reward"))
    _set_default(config, "model_dtype", actor_fsdp.get("model_dtype"))
    _set_default(config, "model_dtype", ref_fsdp.get("model_dtype"))
    _set_default(config, "model_dtype", model.get("model_dtype"))
    _set_default(config, "attn_implementation", model.get("override_config", {}).get("attn_implementation"))
    _set_default(config, "actor_lr", actor.get("optim", {}).get("lr"))
    _set_default(config, "ppo_mini_batch_size", actor.get("ppo_mini_batch_size"))
    _set_default(config, "ppo_mini_batch_size", model.get("ppo_mini_batch_size"))
    _set_default(config, "ppo_micro_batch_size_per_gpu", actor.get("ppo_micro_batch_size_per_gpu"))
    _set_default(config, "ppo_micro_batch_size_per_gpu", model.get("ppo_micro_batch_size_per_gpu"))
    _set_default(config, "infer_backend", rollout.get("name"))
    _set_default(config, "rollout_n", rollout.get("n"))
    _set_default(config, "rollout_dtype", rollout.get("dtype"))
    _set_default(config, "temperature", rollout.get("temperature"))
    _set_default(config, "rollout_tp", rollout.get("tensor_model_parallel_size"))
    _set_default(config, "rollout_gpu_memory_utilization", rollout.get("gpu_memory_utilization"))
    _set_default(config, "log_prob_micro_batch_size_per_gpu", rollout.get("log_prob_micro_batch_size_per_gpu"))
    _set_default(config, "log_prob_micro_batch_size_per_gpu", ref.get("log_prob_micro_batch_size_per_gpu"))
    _set_default(config, "reward_manager", reward.get("reward_manager", {}).get("name"))
    _set_default(config, "project_name", trainer.get("project_name"))
    _set_default(config, "experiment_name", trainer.get("experiment_name"))
    _set_default(config, "trainer_logger", trainer.get("logger"))
    _set_default(config, "n_gpus_per_node", trainer.get("n_gpus_per_node"))
    _set_default(config, "nnodes", trainer.get("nnodes"))
    _set_default(config, "num_epochs", trainer.get("total_epochs"))
    _set_default(config, "save_freq", trainer.get("save_freq"))
    _set_default(config, "checkpoint_interval", trainer.get("save_freq"))
    _set_default(config, "test_freq", trainer.get("test_freq"))
    return config


def _set_default(config: dict, key: str, value) -> None:
    if value is not None and key not in config:
        config[key] = value


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
    if args.save_freq is not None:
        base_config["save_freq"] = args.save_freq
    if args.tau is not None:
        base_config["tau"] = args.tau
    if args.top_k is not None:
        base_config["top_k"] = args.top_k
    if args.top_k_points is not None:
        base_config["top_k_points"] = args.top_k_points
        base_config["retriever_top_k_points"] = args.top_k_points
    if args.regression_sample_size is not None:
        base_config["regression_sample_size"] = args.regression_sample_size
    if args.episodes_per_case is not None:
        base_config["episodes_per_case"] = args.episodes_per_case
    if args.seed is not None:
        base_config["seed"] = args.seed
    if args.max_prompt_length is not None:
        base_config["max_prompt_length"] = args.max_prompt_length
    if args.max_response_length is not None:
        base_config["max_response_length"] = args.max_response_length
    if args.model_dtype is not None:
        base_config["model_dtype"] = args.model_dtype
    if args.rollout_dtype is not None:
        base_config["rollout_dtype"] = args.rollout_dtype
    if args.attn_implementation is not None:
        base_config["attn_implementation"] = args.attn_implementation
    if args.infer_backend is not None:
        base_config["infer_backend"] = args.infer_backend
    if args.rollout_tp is not None:
        base_config["rollout_tp"] = args.rollout_tp
    if args.rollout_gpu_memory_utilization is not None:
        base_config["rollout_gpu_memory_utilization"] = args.rollout_gpu_memory_utilization
    if args.project_name is not None:
        base_config["project_name"] = args.project_name
    if args.experiment_name is not None:
        base_config["experiment_name"] = args.experiment_name
    if args.commit_threshold is not None:
        base_config["commit_threshold"] = args.commit_threshold
    if args.retriever_type is not None:
        base_config["retriever_type"] = args.retriever_type
    if args.retriever_model_name is not None:
        base_config["retriever_model_name"] = args.retriever_model_name
    if args.retriever_embedding_model is not None:
        base_config["retriever_embedding_model"] = args.retriever_embedding_model
    if args.retriever_retrieval_mode is not None:
        base_config["retriever_retrieval_mode"] = args.retriever_retrieval_mode
    if args.retriever_device is not None:
        base_config["retriever_device"] = args.retriever_device
    if args.retriever_cache_dir is not None:
        base_config["retriever_cache_dir"] = args.retriever_cache_dir
    if args.retriever_require_model:
        base_config["retriever_require_model"] = True
    if args.reward_mode is not None:
        base_config["reward_config"] = dict(base_config.get("reward_config", {}))
        base_config["reward_config"]["mode"] = args.reward_mode
    if args.attacker_llm is not None:
        base_config["attacker_llm"] = args.attacker_llm
    if args.attacker_api_base is not None:
        base_config["attacker_api_base"] = args.attacker_api_base
    if args.attacker_api_key is not None:
        base_config["attacker_api_key"] = args.attacker_api_key
    if args.memory_trajectory_dir is not None:
        base_config["memory_trajectory_dir"] = args.memory_trajectory_dir
    if args.disable_memory_trajectory:
        base_config["memory_trajectory_enabled"] = False

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
    parser.add_argument("--save-freq", type=int, help="verl model checkpoint save frequency")

    # Environment parameters
    parser.add_argument("--tau", type=float, help="Add/merge similarity threshold")
    parser.add_argument("--top-k", type=int, help="Top-K retrieval")
    parser.add_argument("--top-k-points", type=int, help="Top-K dense retrieval points before chunk aggregation")
    parser.add_argument("--regression-sample-size", type=int, help="Regression questions sampled for ADD actions")
    parser.add_argument("--episodes-per-case", type=int, help="Episodes per case graph")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--commit-threshold", type=float, help="Minimum reward required to commit")
    parser.add_argument("--retriever-type", help="Retriever type, e.g. dense_structured or bm25")
    parser.add_argument("--retriever-model-name", help="Dense retriever HF model name or local path")
    parser.add_argument("--retriever-embedding-model", help="Dense retriever family, e.g. contriever or hash")
    parser.add_argument("--retriever-retrieval-mode", help="Structured retrieval mode: flatten, merge, or separate")
    parser.add_argument("--retriever-device", help="Retriever device, e.g. cpu or cuda:0")
    parser.add_argument("--retriever-cache-dir", help="Retriever model cache directory")
    parser.add_argument("--retriever-require-model", action="store_true", help="Fail instead of falling back to hash retrieval")
    parser.add_argument("--reward-mode", choices=["semantic_complete", "evaluation_aligned"], help="Reward evaluator mode")
    parser.add_argument("--memory-trajectory-dir", help="Directory name under output-dir for per-step M_t snapshots")
    parser.add_argument(
        "--disable-memory-trajectory",
        action="store_true",
        help="Disable per-commit/rollback memory trajectory snapshots",
    )

    # Attack generation
    parser.add_argument("--attacker-llm", help="OpenAI-compatible attacker model name/path")
    parser.add_argument("--attacker-api-base", help="OpenAI-compatible attacker API base URL")
    parser.add_argument("--attacker-api-key", help="Optional attacker API key")

    # Model parameters
    parser.add_argument("--max-prompt-length", type=int, help="Max prompt tokens")
    parser.add_argument("--max-response-length", type=int, help="Max response tokens")
    parser.add_argument("--model-dtype", help="FSDP model load dtype, e.g. bfloat16 or float16")
    parser.add_argument("--rollout-dtype", help="Rollout/vLLM dtype, e.g. bfloat16 or float16")
    parser.add_argument(
        "--attn-implementation",
        help="Transformers attention implementation, e.g. flash_attention_2",
    )

    # verl backend
    parser.add_argument("--infer-backend", help="Inference backend")
    parser.add_argument("--rollout-tp", type=int, help="Tensor parallel size")
    parser.add_argument(
        "--rollout-gpu-memory-utilization",
        type=float,
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
    logger.info(f"Retriever: {config.get('retriever_type', config.get('retriever', {}).get('type', 'bm25'))}")
    logger.info(f"Reward mode: {config.get('reward_config', {}).get('mode', 'semantic_complete')}")
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
