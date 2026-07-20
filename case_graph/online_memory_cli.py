"""CLI entry point for online GRPO memory refactoring training."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import yaml

from .online_memory_dataset import OnlineMemoryDataset
from .retriever import MemoryChunk, MemoryStore, build_memory_retriever, retriever_config_from_mapping

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


def load_config(config_path: str, section: Optional[str] = None) -> dict:
    """Load a flat config or one phase from a co-training config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = config or {}

    selected_section = section
    if selected_section is None and "cotrain_rounds" in config:
        defender_config = config.get("defender")
        if isinstance(defender_config, dict):
            selected_section = "defender"

    if selected_section is not None:
        section_config = config.get(selected_section)
        if not isinstance(section_config, dict):
            available_sections = sorted(
                key for key, value in config.items() if isinstance(value, dict)
            )
            raise ValueError(
                f"Config section {selected_section!r} was not found in {config_path}; "
                f"available sections: {available_sections}"
            )
        logger.info(
            "Using config section %s from %s",
            selected_section,
            config_path,
        )
        config = dict(section_config)

    return normalize_verl_config(config)


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
    if args.test_freq is not None:
        base_config["test_freq"] = args.test_freq
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
    if args.max_questions_per_case is not None:
        base_config["max_questions_per_case"] = args.max_questions_per_case
        base_config["episodes_per_case"] = args.max_questions_per_case
    if args.coverage_threshold is not None:
        base_config["coverage_threshold"] = args.coverage_threshold
    if args.critical_coverage_threshold is not None:
        base_config["critical_coverage_threshold"] = args.critical_coverage_threshold
    if args.training_probe_window is not None:
        base_config["training_probe_window"] = args.training_probe_window
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
    if args.n_gpus_per_node is not None:
        base_config["n_gpus_per_node"] = args.n_gpus_per_node
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
    if args.retriever_max_length is not None:
        base_config["retriever_max_length"] = args.retriever_max_length
    if args.reward_mode is not None:
        base_config["reward_config"] = dict(base_config.get("reward_config", {}))
        base_config["reward_config"]["mode"] = args.reward_mode
    if args.initial_defense_use_llm:
        base_config["initial_defense_use_llm"] = True
    if args.initial_defense_judge_use_llm:
        base_config["initial_defense_judge_use_llm"] = True
    if args.skip_retriever_preflight:
        base_config["retriever_preflight"] = False
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


def preflight_retriever(config: dict) -> None:
    """Load the configured retriever once before Ray starts reward workers."""
    if config.get("retriever_preflight", True) is False:
        logger.info("Skipping retriever preflight.")
        return
    retriever_config = retriever_config_from_mapping(config)
    retriever_type = str(retriever_config.get("type", "bm25")).casefold()
    if retriever_type in {"bm25", "frozen_bm25", "frozen-bm25"}:
        return

    logger.info(
        "Preflighting retriever: type=%s model=%s device=%s require_model=%s",
        retriever_config.get("type"),
        retriever_config.get("model_name"),
        retriever_config.get("device"),
        retriever_config.get("require_model"),
    )
    store = MemoryStore(
        [
            MemoryChunk(
                "preflight",
                "The user admired Jose Altuve after watching the Astros.",
                metadata={
                    "facts": [
                        "The user admired Jose Altuve after watching the Astros. "
                        + " ".join(["context"] * 700)
                    ],
                    "keywords": ["Jose Altuve", "Astros"],
                    "summary": "Baseball preference memory.",
                },
            )
        ]
    )
    try:
        hits = build_memory_retriever(retriever_config, store).retrieve(
            "Which Astros player did the user admire?",
            top_k=1,
        )
    except Exception as exc:
        raise SystemExit(
            "Retriever preflight failed before starting Ray. "
            "This usually means RETRIEVER_MODEL_NAME does not point to a complete "
            "HuggingFace checkpoint, or the reward workers cannot load that model. "
            f"Config: {retriever_config}. Error: {exc}"
        ) from exc
    if not hits:
        raise SystemExit(f"Retriever preflight returned no hits. Config: {retriever_config}")
    logger.info("Retriever preflight succeeded: top_hit=%s score=%.4f", hits[0].memory_id, hits[0].score)


def validate_training_scale(config: dict, num_graphs: int) -> None:
    """Warn when the active config has fallen back to smoke-test scale."""
    train_batch_size = _as_int(config.get("train_batch_size"), 16)
    rollout_n = _as_int(config.get("rollout_n"), 8)
    ppo_mini_batch_size = _as_int(config.get("ppo_mini_batch_size"), 8)
    num_epochs = _as_int(config.get("num_epochs"), 8)
    episodes_per_case = _as_int(
        config.get("max_questions_per_case", config.get("episodes_per_case")), 200
    )
    regression_sample_size = _as_int(config.get("regression_sample_size"), 12)
    max_response_length = _as_int(config.get("max_response_length"), 1024)
    top_k = _as_int(config.get("top_k"), 8)
    retriever_config = retriever_config_from_mapping(config)
    top_k_points = _as_int(
        config.get("top_k_points", retriever_config.get("top_k_points")),
        32,
    )
    save_freq = _as_int(config.get("save_freq"), 500)
    total_examples = max(0, num_graphs) * max(0, episodes_per_case)
    steps_per_epoch = (
        (total_examples + train_batch_size - 1) // train_batch_size
        if train_batch_size > 0
        else 0
    )
    total_optimizer_steps = steps_per_epoch * max(0, num_epochs)
    sampled_rollouts_per_step = train_batch_size * rollout_n

    logger.info(
        "Training scale: examples=%s steps_per_epoch=%s total_steps=%s sampled_rollouts_per_step=%s",
        total_examples,
        steps_per_epoch,
        total_optimizer_steps,
        sampled_rollouts_per_step,
    )

    warnings = []
    if num_epochs < 1:
        warnings.append("num_epochs must be at least 1.")
    if train_batch_size < 8:
        warnings.append(f"train_batch_size={train_batch_size} is small; use >=8, preferably 16 on A6000/Qwen3-0.6B.")
    if ppo_mini_batch_size < 4:
        warnings.append(f"ppo_mini_batch_size={ppo_mini_batch_size} is small; use >=4, preferably 8.")
    if rollout_n < 8:
        warnings.append(f"rollout_n={rollout_n} weakens GRPO selection; use >=8.")
    if episodes_per_case < 50:
        warnings.append(
            f"max_questions_per_case={episodes_per_case} is a shallow adaptive budget; use 100-250."
        )
    if regression_sample_size < 8:
        warnings.append(f"regression_sample_size={regression_sample_size} may miss forgetting; use >=8.")
    if top_k < 8:
        warnings.append(f"top_k={top_k} may under-retrieve facts; use >=8 with dense_structured retrieval.")
    if top_k_points < 24:
        warnings.append(f"top_k_points={top_k_points} is low for flattened fact retrieval; use >=24.")
    if max_response_length < 768:
        warnings.append(f"max_response_length={max_response_length} may truncate structured JSON; use >=768.")
    if save_freq > 0 and save_freq < 250:
        warnings.append(f"save_freq={save_freq} is very frequent for formal training; use >=250 unless debugging.")
    retriever_type = str(retriever_config.get("type", "bm25")).casefold()
    if retriever_type in {"bm25", "frozen_bm25", "frozen-bm25"}:
        warnings.append("retriever is BM25; formal runs should use dense_structured/contriever.")
    if retriever_type not in {"bm25", "frozen_bm25", "frozen-bm25"} and not retriever_config.get("require_model"):
        warnings.append("retriever.require_model=false allows hash fallback; set RETRIEVER_REQUIRE_MODEL=true for formal runs.")
    if total_optimizer_steps < 1000:
        warnings.append(
            f"total_optimizer_steps={total_optimizer_steps} is short; increase graph count or max_questions_per_case."
        )
    if num_graphs < train_batch_size:
        warnings.append(
            f"num_graphs={num_graphs} is smaller than train_batch_size={train_batch_size}; "
            "multiple stale states from the same case can enter one batch. Use more training cases for formal runs."
        )
    if num_graphs < 20:
        warnings.append(
            f"num_graphs={num_graphs} is too small to establish policy generalization; "
            "formal training should use a substantially larger case split and held-out validation cases."
        )
    reward_config = dict(config.get("reward_config", {}) or {})
    if float(reward_config.get("min_relation_overlap", 0.0)) < 0.5:
        warnings.append("reward min_relation_overlap is permissive; use >=0.5 to reject answer-word memories.")
    if float(reward_config.get("min_fact_overlap", 0.0)) < 0.5:
        warnings.append("reward min_fact_overlap is permissive; use >=0.5 for fact completeness.")
    if retriever_config.get("device") == "cuda" and _as_int(config.get("n_gpus_per_node"), 1) == 1:
        warnings.append(
            "retriever.device=cuda shares the only GPU with actor/vLLM and may load one model per reward worker; use cpu."
        )

    for message in warnings:
        logger.warning("Training scale check: %s", message)


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
        "--val-graphs",
        help="Held-out validation CaseGraph directory or file",
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
        "--config-section",
        help=(
            "Top-level YAML section to use, such as defender in "
            "configs/cotrain_v2.yaml. Co-training configs select defender "
            "automatically."
        ),
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
    parser.add_argument("--test-freq", type=int, help="verl validation frequency")

    # Environment parameters
    parser.add_argument("--tau", type=float, help="Add/merge similarity threshold")
    parser.add_argument("--top-k", type=int, help="Top-K retrieval")
    parser.add_argument("--top-k-points", type=int, help="Top-K dense retrieval points before chunk aggregation")
    parser.add_argument("--regression-sample-size", type=int, help="Regression questions sampled for ADD actions")
    parser.add_argument("--episodes-per-case", type=int, help="Episodes per case graph")
    parser.add_argument("--max-questions-per-case", type=int, help="Maximum adaptive construction questions per training case")
    parser.add_argument("--coverage-threshold", type=float, help="Weighted fact coverage needed before training probes")
    parser.add_argument("--critical-coverage-threshold", type=float, help="Critical fact coverage needed before training probes")
    parser.add_argument("--training-probe-window", type=int, help="Consecutive successful probes needed to resolve a training case")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--commit-threshold", type=float, help="Minimum reward required to commit")
    parser.add_argument("--retriever-type", help="Retriever type, e.g. dense_structured or bm25")
    parser.add_argument("--retriever-model-name", help="Dense retriever HF model name or local path")
    parser.add_argument("--retriever-embedding-model", help="Dense retriever family, e.g. contriever or hash")
    parser.add_argument("--retriever-retrieval-mode", help="Structured retrieval mode: flatten, merge, or separate")
    parser.add_argument("--retriever-device", help="Retriever device, e.g. cpu or cuda:0")
    parser.add_argument("--retriever-cache-dir", help="Retriever model cache directory")
    parser.add_argument("--retriever-require-model", action="store_true", help="Fail instead of falling back to hash retrieval")
    parser.add_argument("--retriever-max-length", type=int, help="Max token length for dense retriever encoding")
    parser.add_argument("--skip-retriever-preflight", action="store_true", help="Skip loading the configured retriever before Ray starts")
    parser.add_argument("--reward-mode", choices=["semantic_complete", "evaluation_aligned"], help="Reward evaluator mode")
    parser.add_argument(
        "--initial-defense-use-llm",
        action="store_true",
        help="Allow initial-defense retrieved-memory answering to call the configured LLM API",
    )
    parser.add_argument(
        "--initial-defense-judge-use-llm",
        action="store_true",
        help="Allow initial-defense answer equivalence judging to call the configured LLM API",
    )
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
    parser.add_argument(
        "--n-gpus-per-node",
        type=int,
        help="Number of visible training GPUs used by verl on each node",
    )

    # Logging
    parser.add_argument("--project-name", help="Wandb project name")
    parser.add_argument("--experiment-name", help="Experiment name")

    args = parser.parse_args()

    # Load and merge config
    base_config = load_config(args.config, section=args.config_section)
    config = merge_config(base_config, args)

    # List graph files
    graph_files = list_graph_files(args.graphs)
    val_graph_files = list_graph_files(args.val_graphs) if args.val_graphs else None

    logger.info("=" * 60)
    logger.info("Online GRPO Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Number of graphs: {len(graph_files)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Rollout N: {config.get('rollout_n', 8)}")
    logger.info(f"Batch size: {config.get('train_batch_size', 16)}")
    logger.info(
        "Max questions per case: %s",
        config.get("max_questions_per_case", config.get("episodes_per_case", 200)),
    )
    logger.info(f"Retriever: {config.get('retriever_type', config.get('retriever', {}).get('type', 'bm25'))}")
    logger.info(f"Reward mode: {config.get('reward_config', {}).get('mode', 'semantic_complete')}")
    logger.info(f"Seed: {config.get('seed', 42)}")
    logger.info("=" * 60)

    validate_training_scale(config, len(graph_files))
    preflight_retriever(config)

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
            val_graph_files=val_graph_files,
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
