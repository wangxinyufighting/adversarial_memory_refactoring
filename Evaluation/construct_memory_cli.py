"""CLI for building evaluation memories with a trained defender checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import yaml

from case_graph.attacker import FrozenLLMAttacker
from case_graph.baseline import AnswerEquivalenceJudge
from case_graph.defense import RetrievedMemoryAnswerAgent
from case_graph.evaluation import graph_paths, load_graph
from case_graph.retriever import retriever_config_from_mapping

from .defender_server_manager import DefenderServerManager
from .memory_construction import (
    CoverageGraphAttacker,
    DefenderCheckpointPolicy,
    EvaluationMemoryConstructor,
    MemoryConstructionConfig,
    build_openai_client,
    construct_memories_for_graphs,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct target-free evaluation memories with a trained defender checkpoint."
    )
    parser.add_argument("--graphs", required=True, help="CaseGraph JSON file or directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for memory_states and traces.")
    parser.add_argument("--initial-memory-dir", help="Optional per-case memory directory to warm start from.")
    parser.add_argument(
        "--training-config",
        default="configs/online_grpo.yaml",
        help="Training YAML whose retrieval, routing, threshold, and reward settings are reused.",
    )

    parser.add_argument("--defender-api-base", default="http://localhost:8004/v1")
    parser.add_argument("--defender-served-model", default="defender-current")
    parser.add_argument("--defender-api-key", default="dummy-key")
    parser.add_argument("--defender-timeout", type=int, default=120)
    parser.add_argument("--defender-max-output-tokens", type=int, default=4096)
    parser.add_argument("--defender-proposal-retries", type=int, default=2)
    parser.add_argument("--defender-checkpoint", help="Optional HF model dir or verl global_step checkpoint to serve.")
    parser.add_argument("--manage-defender-server", action="store_true")
    parser.add_argument("--defender-server-port", type=int, default=8004)
    parser.add_argument("--defender-server-host", default="0.0.0.0")
    parser.add_argument("--defender-server-dtype", default="bfloat16")
    parser.add_argument("--defender-server-tp", type=int, default=1)
    parser.add_argument("--defender-server-gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--defender-server-max-model-len", type=int)
    parser.add_argument("--defender-server-api-key", default="")
    parser.add_argument("--defender-server-startup-timeout", type=float, default=600)
    parser.add_argument("--defender-checkpoint-backend", default="fsdp", choices=["fsdp", "megatron"])
    parser.add_argument("--defender-checkpoint-subdir", default="actor")

    parser.add_argument("--attacker-api-base", help="Optional OpenAI-compatible API for construction probes.")
    parser.add_argument("--attacker-model", help="Optional model name for construction probe generation.")
    parser.add_argument("--attacker-api-key", default="dummy-key")
    parser.add_argument("--attacker-timeout", type=int, default=120)
    parser.add_argument("--attacker-max-output-tokens", type=int, default=700)
    parser.add_argument(
        "--attacker-mode",
        choices=["llm", "coverage"],
        default="coverage",
        help="Use training-consistent LLM probes or high-coverage deterministic probes.",
    )

    parser.add_argument("--answer-api-base", help="Optional answer backbone API base.")
    parser.add_argument("--answer-model", help="Optional answer backbone model name.")
    parser.add_argument("--answer-api-key", default="dummy-key")
    parser.add_argument("--answer-timeout", type=int, default=120)
    parser.add_argument("--answer-max-output-tokens", type=int, default=200)
    parser.add_argument("--judge-api-base", help="Optional judge API base.")
    parser.add_argument("--judge-model", help="Optional judge model name.")
    parser.add_argument("--judge-api-key", default="dummy-key")
    parser.add_argument("--judge-timeout", type=int, default=120)
    parser.add_argument("--judge-max-output-tokens", type=int, default=200)
    parser.add_argument("--skip-llm-judge", action="store_true")

    parser.add_argument(
        "--episodes-per-case",
        type=int,
        default=250,
        help="Question-budget floor; dynamic coverage can raise it up to the hard cap.",
    )
    parser.add_argument("--disable-dynamic-question-budget", action="store_true")
    parser.add_argument("--questions-per-unit", type=float, default=3.0)
    parser.add_argument("--hard-max-questions-per-case", type=int, default=2000)
    parser.add_argument("--min-questions-per-case", type=int, default=20)
    parser.add_argument("--coverage-threshold", type=float)
    parser.add_argument("--critical-coverage-threshold", type=float)
    parser.add_argument("--certification-questions", type=int, default=60)
    parser.add_argument("--disable-adaptive-stopping", action="store_true")
    parser.add_argument(
        "--force-add",
        action="store_true",
        help="Debug-only: disable Merge during construction",
    )
    parser.add_argument("--proposal-count", type=int, default=1)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-k-points", type=int)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--regression-sample-size", type=int)
    parser.add_argument("--commit-threshold", type=float)
    parser.add_argument("--retriever-type")
    parser.add_argument("--retriever-model-name")
    parser.add_argument("--retriever-embedding-model")
    parser.add_argument("--retriever-retrieval-mode")
    parser.add_argument("--retriever-device")
    parser.add_argument("--retriever-cache-dir")
    parser.add_argument("--retriever-max-length", type=int)
    parser.add_argument("--retriever-require-model", action="store_true", default=None)
    parser.add_argument("--reward-mode", choices=["semantic_complete", "evaluation_aligned"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--progress-log-interval", type=int, default=1)
    parser.add_argument("--routing-max-steps", type=int)
    parser.add_argument("--routing-min-nodes", type=int)
    parser.add_argument("--routing-attempts", type=int)
    parser.add_argument("--max-attack-failures", type=int, default=20)
    parser.add_argument("--max-retries-per-unit", type=int, default=3)
    parser.add_argument("--disable-compositional-probes", action="store_true")
    parser.add_argument("--trace-detail", choices=["compact", "full"], default="compact")
    parser.add_argument("--case-workers", type=int, default=1)
    parser.add_argument("--exp-name", default="eval_memory_construction")
    parser.add_argument("--no-traces", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    defender_server = None
    defender_api_base = args.defender_api_base
    defender_served_model = args.defender_served_model
    if (
        args.manage_defender_server
        and args.defender_server_api_key
        and args.defender_api_key == "dummy-key"
    ):
        args.defender_api_key = args.defender_server_api_key
    if args.manage_defender_server:
        if not args.defender_checkpoint:
            raise SystemExit("--manage-defender-server requires --defender-checkpoint.")
        defender_server = _start_defender_server(args, output_dir)
        handle = defender_server.serve(args.defender_checkpoint)
        defender_api_base = handle.api_base
        defender_served_model = handle.served_model_name

    try:
        training_config = _load_training_config(args.training_config)
        reward_config = dict(training_config.get("reward_config", {}) or {})
        reward_config["mode"] = args.reward_mode or reward_config.get(
            "mode", "semantic_complete"
        )
        tau = _setting(args.tau, training_config, "tau", 0.55)
        top_k = _setting(args.top_k, training_config, "top_k", 8)
        top_k_points = _setting(args.top_k_points, training_config, "top_k_points", 32)
        min_score = _setting(args.min_score, training_config, "min_score", 0.0)
        regression_sample_size = _setting(
            args.regression_sample_size,
            training_config,
            "regression_sample_size",
            12,
        )
        commit_threshold = _setting(
            args.commit_threshold,
            training_config,
            "commit_threshold",
            1.0,
        )
        coverage_threshold = _setting(
            args.coverage_threshold,
            training_config,
            "coverage_threshold",
            0.98,
        )
        critical_coverage_threshold = _setting(
            args.critical_coverage_threshold,
            training_config,
            "critical_coverage_threshold",
            1.0,
        )
        seed = _setting(args.seed, training_config, "seed", 42)
        routing_max_steps = _setting(
            args.routing_max_steps,
            training_config,
            "routing_max_steps",
            4,
        )
        routing_min_nodes = _setting(
            args.routing_min_nodes,
            training_config,
            "routing_min_nodes",
            1,
        )
        routing_attempts = _setting(
            args.routing_attempts,
            training_config,
            "routing_attempts",
            12,
        )
        retriever_mapping = {"retriever": dict(training_config.get("retriever", {}) or {})}
        retriever_overrides = {
            "retriever_type": args.retriever_type,
            "retriever_model_name": args.retriever_model_name,
            "retriever_embedding_model": args.retriever_embedding_model,
            "retriever_retrieval_mode": args.retriever_retrieval_mode,
            "retriever_top_k_points": top_k_points,
            "retriever_device": args.retriever_device,
            "retriever_cache_dir": args.retriever_cache_dir,
            "retriever_require_model": args.retriever_require_model,
            "retriever_max_length": args.retriever_max_length,
        }
        retriever_mapping.update(
            {key: value for key, value in retriever_overrides.items() if value is not None}
        )
        retriever_config = retriever_config_from_mapping(retriever_mapping)
        config = MemoryConstructionConfig(
            tau=tau,
            top_k=top_k,
            top_k_points=top_k_points,
            min_score=min_score,
            regression_sample_size=regression_sample_size,
            episodes_per_case=args.episodes_per_case,
            dynamic_question_budget=not args.disable_dynamic_question_budget,
            questions_per_unit=args.questions_per_unit,
            hard_max_questions_per_case=args.hard_max_questions_per_case,
            min_questions_per_case=args.min_questions_per_case,
            coverage_threshold=coverage_threshold,
            critical_coverage_threshold=critical_coverage_threshold,
            certification_questions=args.certification_questions,
            adaptive_stopping=not args.disable_adaptive_stopping,
            proposal_count=args.proposal_count,
            commit_threshold=commit_threshold,
            seed=seed,
            force_add=args.force_add,
            progress_log_interval=args.progress_log_interval,
            routing_max_steps=routing_max_steps,
            routing_min_nodes=routing_min_nodes,
            routing_attempts=routing_attempts,
            max_attack_failures=args.max_attack_failures,
            defender_max_output_tokens=args.defender_max_output_tokens,
            defender_proposal_retries=args.defender_proposal_retries,
            attacker_max_output_tokens=args.attacker_max_output_tokens,
            max_retries_per_unit=args.max_retries_per_unit,
            compositional_probes=not args.disable_compositional_probes,
            trace_detail=args.trace_detail,
            case_workers=max(1, args.case_workers),
            retriever_config=retriever_config,
            reward_config=reward_config,
            exp_name=args.exp_name,
        )
        defender_policy = DefenderCheckpointPolicy(
            client=build_openai_client(
                model=defender_served_model,
                api_base=defender_api_base,
                api_key=args.defender_api_key,
                timeout=args.defender_timeout,
            ),
            max_output_tokens=args.defender_max_output_tokens,
            retries=args.defender_proposal_retries,
        )
        attacker = _build_attacker(args)
        answer_client = _build_answer_client(args, defender_api_base, defender_served_model)
        judge_client = _build_judge_client(args, defender_api_base, defender_served_model)
        use_llm_judge = _use_llm_judge(args, judge_client)
        answer_agent = RetrievedMemoryAnswerAgent(
            client=answer_client,
            max_output_tokens=args.answer_max_output_tokens,
        )
        judge = AnswerEquivalenceJudge(
            client=judge_client,
            max_output_tokens=args.judge_max_output_tokens,
            use_llm=use_llm_judge,
        )
        constructor = EvaluationMemoryConstructor(
            config=config,
            defender_policy=defender_policy,
            attacker=attacker,
            answer_agent=answer_agent,
            judge=judge,
        )
        _write_run_config(
            args,
            output_dir,
            defender_api_base,
            defender_served_model,
            config,
        )
        graphs = [load_graph(path) for path in graph_paths(args.graphs)]
        summary = construct_memories_for_graphs(
            graphs=graphs,
            constructor=constructor,
            output_dir=output_dir,
            initial_memory_dir=args.initial_memory_dir,
            save_traces=not args.no_traces,
        )
        print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))
    finally:
        if defender_server is not None:
            defender_server.stop()


def _optional_client(
    model: str | None,
    api_base: str | None,
    api_key: str,
    timeout: int,
):
    if not model and not api_base:
        return None
    return build_openai_client(
        model=model or "local-model",
        api_base=api_base or "http://localhost:8000/v1",
        api_key=api_key,
        timeout=timeout,
    )


def _load_training_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        logger.warning("Training config %s not found; using Evaluation defaults.", path)
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return dict(payload) if isinstance(payload, dict) else {}


def _setting(cli_value, training_config: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    value = training_config.get(key, default)
    return default if value is None else value


def _build_attacker(args: argparse.Namespace):
    if args.attacker_mode == "coverage":
        logger.info("Using deterministic coverage attacker for evaluation memory construction.")
        return CoverageGraphAttacker(
            compositional_probes=not args.disable_compositional_probes,
        )

    client = _optional_client(
        model=args.attacker_model,
        api_base=args.attacker_api_base,
        api_key=args.attacker_api_key,
        timeout=args.attacker_timeout,
    )
    if client is None and not _env_llm_configured():
        raise SystemExit(
            "ATTACKER_MODE=llm requires ATTACKER_API_BASE/ATTACKER_MODEL or a global LLM env "
            "such as CASE_GRAPH_PROVIDER=local with LOCAL_API_BASE_URL. Use ATTACKER_MODE=coverage "
            "when no attacker server is needed."
        )

    logger.info("Using LLM attacker for training-consistent construction probes.")
    return FrozenLLMAttacker(
        client=client,
        max_output_tokens=args.attacker_max_output_tokens,
    )


def _build_answer_client(
    args: argparse.Namespace,
    defender_api_base: str,
    defender_served_model: str,
):
    client = _optional_client(
        model=args.answer_model,
        api_base=args.answer_api_base,
        api_key=args.answer_api_key,
        timeout=args.answer_timeout,
    )
    if client is not None or _env_llm_configured():
        return client

    logger.warning(
        "No answer API or global LLM env configured; reusing the defender endpoint for "
        "retrieved-memory answering. Set ANSWER_API_BASE/ANSWER_MODEL to use a separate answer model."
    )
    return build_openai_client(
        model=defender_served_model,
        api_base=defender_api_base,
        api_key=args.defender_api_key,
        timeout=args.defender_timeout,
    )


def _use_llm_judge(args: argparse.Namespace, judge_client) -> bool:
    if args.skip_llm_judge:
        return False
    return judge_client is not None or _env_llm_configured()


def _build_judge_client(
    args: argparse.Namespace,
    defender_api_base: str,
    defender_served_model: str,
):
    if args.skip_llm_judge:
        return None
    client = _optional_client(
        model=args.judge_model,
        api_base=args.judge_api_base,
        api_key=args.judge_api_key,
        timeout=args.judge_timeout,
    )
    if client is not None or _env_llm_configured():
        return client

    logger.warning(
        "No judge API or global LLM env configured; reusing the defender endpoint for "
        "semantic answer judging. Set SKIP_LLM_JUDGE=true to force string-match judging."
    )
    return build_openai_client(
        model=defender_served_model,
        api_base=defender_api_base,
        api_key=args.defender_api_key,
        timeout=args.defender_timeout,
    )


def _env_llm_configured() -> bool:
    return any(
        os.environ.get(name)
        for name in (
            "CASE_GRAPH_PROVIDER",
            "LOCAL_API_BASE_URL",
            "LOCAL_BASE_URL",
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
        )
    )


def _start_defender_server(args: argparse.Namespace, output_dir: Path) -> DefenderServerManager:
    parsed = urlparse(args.defender_api_base)
    request_host = parsed.hostname or "localhost"
    port = parsed.port or args.defender_server_port
    api_base = f"http://{request_host}:{port}/v1"
    return DefenderServerManager(
        output_dir=output_dir,
        served_model_name=args.defender_served_model,
        api_base=api_base,
        host=args.defender_server_host,
        port=port,
        dtype=args.defender_server_dtype,
        tensor_parallel_size=args.defender_server_tp,
        gpu_memory_utilization=args.defender_server_gpu_memory_utilization,
        max_model_len=args.defender_server_max_model_len,
        api_key=args.defender_server_api_key,
        startup_timeout=args.defender_server_startup_timeout,
        checkpoint_backend=args.defender_checkpoint_backend,
        checkpoint_subdir=args.defender_checkpoint_subdir,
    )


def _write_run_config(
    args: argparse.Namespace,
    output_dir: Path,
    defender_api_base: str,
    defender_served_model: str,
    config: MemoryConstructionConfig,
) -> None:
    payload = vars(args).copy()
    for key in list(payload):
        if "api_key" in key and payload.get(key):
            payload[key] = "<redacted>"
    payload["resolved_defender_api_base"] = defender_api_base
    payload["resolved_defender_served_model"] = defender_served_model
    payload["resolved_memory_construction"] = asdict(config)
    (output_dir / "construction_config.yaml").write_text(
        yaml.dump(payload, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
