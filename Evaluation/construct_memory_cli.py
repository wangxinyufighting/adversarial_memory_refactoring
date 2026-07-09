"""CLI for building evaluation memories with a trained defender checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

import yaml

from case_graph.attacker import FrozenLLMAttacker
from case_graph.baseline import AnswerEquivalenceJudge
from case_graph.defense import RetrievedMemoryAnswerAgent
from case_graph.evaluation import graph_paths, load_graph

from .defender_server_manager import DefenderServerManager
from .memory_construction import (
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

    parser.add_argument("--defender-api-base", default="http://localhost:8004/v1")
    parser.add_argument("--defender-served-model", default="defender-current")
    parser.add_argument("--defender-api-key", default="dummy-key")
    parser.add_argument("--defender-timeout", type=int, default=120)
    parser.add_argument("--defender-max-output-tokens", type=int, default=800)
    parser.add_argument("--defender-checkpoint", help="Optional HF model dir or verl global_step checkpoint to serve.")
    parser.add_argument("--manage-defender-server", action="store_true")
    parser.add_argument("--defender-server-port", type=int, default=8004)
    parser.add_argument("--defender-server-host", default="0.0.0.0")
    parser.add_argument("--defender-server-dtype", default="bfloat16")
    parser.add_argument("--defender-server-tp", type=int, default=1)
    parser.add_argument("--defender-server-gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--defender-server-startup-timeout", type=float, default=600)
    parser.add_argument("--defender-checkpoint-backend", default="fsdp", choices=["fsdp", "megatron"])
    parser.add_argument("--defender-checkpoint-subdir", default="actor")

    parser.add_argument("--attacker-api-base", help="Optional OpenAI-compatible API for construction probes.")
    parser.add_argument("--attacker-model", help="Optional model name for construction probe generation.")
    parser.add_argument("--attacker-api-key", default="dummy-key")
    parser.add_argument("--attacker-timeout", type=int, default=120)
    parser.add_argument("--attacker-max-output-tokens", type=int, default=700)

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

    parser.add_argument("--episodes-per-case", type=int, default=100)
    parser.add_argument("--proposal-count", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--regression-sample-size", type=int, default=3)
    parser.add_argument("--commit-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--routing-max-steps", type=int, default=3)
    parser.add_argument("--routing-min-nodes", type=int, default=1)
    parser.add_argument("--routing-attempts", type=int, default=8)
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
    if args.manage_defender_server:
        if not args.defender_checkpoint:
            raise SystemExit("--manage-defender-server requires --defender-checkpoint.")
        defender_server = _start_defender_server(args, output_dir)
        handle = defender_server.serve(args.defender_checkpoint)
        defender_api_base = handle.api_base
        defender_served_model = handle.served_model_name

    try:
        config = MemoryConstructionConfig(
            tau=args.tau,
            top_k=args.top_k,
            min_score=args.min_score,
            regression_sample_size=args.regression_sample_size,
            episodes_per_case=args.episodes_per_case,
            proposal_count=args.proposal_count,
            commit_threshold=args.commit_threshold,
            seed=args.seed,
            routing_max_steps=args.routing_max_steps,
            routing_min_nodes=args.routing_min_nodes,
            routing_attempts=args.routing_attempts,
            defender_max_output_tokens=args.defender_max_output_tokens,
            attacker_max_output_tokens=args.attacker_max_output_tokens,
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
        )
        attacker = FrozenLLMAttacker(
            client=_optional_client(
                model=args.attacker_model,
                api_base=args.attacker_api_base,
                api_key=args.attacker_api_key,
                timeout=args.attacker_timeout,
            ),
            max_output_tokens=args.attacker_max_output_tokens,
        )
        answer_agent = RetrievedMemoryAnswerAgent(
            client=_optional_client(
                model=args.answer_model,
                api_base=args.answer_api_base,
                api_key=args.answer_api_key,
                timeout=args.answer_timeout,
            ),
            max_output_tokens=args.answer_max_output_tokens,
        )
        judge = AnswerEquivalenceJudge(
            client=_optional_client(
                model=args.judge_model,
                api_base=args.judge_api_base,
                api_key=args.judge_api_key,
                timeout=args.judge_timeout,
            ),
            max_output_tokens=args.judge_max_output_tokens,
            use_llm=not args.skip_llm_judge,
        )
        constructor = EvaluationMemoryConstructor(
            config=config,
            defender_policy=defender_policy,
            attacker=attacker,
            answer_agent=answer_agent,
            judge=judge,
        )
        graphs = [load_graph(path) for path in graph_paths(args.graphs)]
        summary = construct_memories_for_graphs(
            graphs=graphs,
            constructor=constructor,
            output_dir=output_dir,
            initial_memory_dir=args.initial_memory_dir,
            save_traces=not args.no_traces,
        )
        _write_run_config(args, output_dir, defender_api_base, defender_served_model)
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
        startup_timeout=args.defender_server_startup_timeout,
        checkpoint_backend=args.defender_checkpoint_backend,
        checkpoint_subdir=args.defender_checkpoint_subdir,
    )


def _write_run_config(
    args: argparse.Namespace,
    output_dir: Path,
    defender_api_base: str,
    defender_served_model: str,
) -> None:
    payload = vars(args).copy()
    payload["resolved_defender_api_base"] = defender_api_base
    payload["resolved_defender_served_model"] = defender_served_model
    (output_dir / "construction_config.yaml").write_text(
        yaml.dump(payload, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
