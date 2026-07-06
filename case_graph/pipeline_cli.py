import argparse
import json
from pathlib import Path

from .attacker import FrozenLLMAttacker
from .baseline import AnswerEquivalenceJudge, GoldenFactAnswerAgent
from .defense import RetrievedMemoryAnswerAgent, SuccessPool
from .llm import OpenAIChatClient
from .pipeline import (
    AlgorithmConfig,
    MemoryRefactoringPipeline,
    load_attacks,
    load_graphs,
    prepare_attacks_from_graphs,
)
from .route_cli import POLICY_NAMES, build_policies, parse_policy_names
from .refactoring import HighPriorityBuffer, PromptMemoryRefactoringPolicy
from .retriever import MemoryStore
from .verifier import GoldenFactVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full online memory-refactoring algorithm.")
    parser.add_argument("--memory", help="Initial memory store M0 JSON.")
    parser.add_argument("--empty-memory", action="store_true", help="Start from an empty M0.")
    parser.add_argument("--attacks", help="Attack JSON list or {'attacks': [...]} file.")
    parser.add_argument("--graphs", help="CaseGraph JSON file or directory for stage-one attack generation.")
    parser.add_argument("--max-graphs", type=int, default=-1, help="Limit graphs used from --graphs; -1 means all.")
    parser.add_argument(
        "--routes-per-graph",
        type=int,
        default=1,
        help="Repeat route sampling this many times per graph and policy.",
    )
    parser.add_argument("--memory-output", required=True, help="Path for final memory store.")
    parser.add_argument("--trace-output", required=True, help="Path for per-step trace JSON.")
    parser.add_argument("--attack-trace-output", help="Write generated routes, attacks, and oracle discards.")
    parser.add_argument("--success-pool", help="Load and update success pool JSON.")
    parser.add_argument("--success-pool-output", help="Write final success pool JSON.")
    parser.add_argument("--high-priority-buffer", help="Load high-priority buffer JSON.")
    parser.add_argument("--high-priority-buffer-output", help="Write final high-priority buffer JSON.")
    parser.add_argument("--memory-archive-dir", help="Directory for pre-commit Mt snapshots.")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--tau", required=True, type=float)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--proposal-count", type=int, default=1)
    parser.add_argument("--regression-sample-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--commit-threshold", type=float, default=0.0)
    parser.add_argument("--answer-max-output-tokens", type=int, default=200)
    parser.add_argument("--judge-max-output-tokens", type=int, default=200)
    parser.add_argument("--policy-max-output-tokens", type=int, default=800)
    parser.add_argument(
        "--policies",
        default="random_walk",
        help=f"Comma-separated routing policies for --graphs. Available: {', '.join(POLICY_NAMES)}.",
    )
    parser.add_argument("--seed-routes", type=int, default=0)
    parser.add_argument("--random-walk-steps", type=int, default=3)
    parser.add_argument("--random-walk-min-nodes", type=int, default=1)
    parser.add_argument("--random-walk-attempts", type=int, default=8)
    parser.add_argument("--route-top-k", type=int, default=5)
    parser.add_argument("--route-max-output-tokens", type=int, default=300)
    parser.add_argument("--attack-max-output-tokens", type=int, default=700)
    parser.add_argument("--verify-max-output-tokens", type=int, default=300)
    parser.add_argument("--use-llm-rerank", action="store_true")
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--keep-failed-verification", action="store_true")
    parser.add_argument("--keep-failed-oracle", action="store_true")
    parser.add_argument("--derive-missing-answer", action="store_true")
    parser.add_argument("--skip-llm-judge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.memory and not args.empty_memory:
        raise ValueError("Provide --memory or set --empty-memory.")
    if not args.attacks and not args.graphs:
        raise ValueError("Provide either --attacks or --graphs.")
    client = OpenAIChatClient.from_env()
    attack_preparation = None
    if args.graphs:
        attack_preparation = _prepare_attacks(args, client)
        attacks = attack_preparation.attacks
    else:
        attacks = load_attacks(args.attacks)
    if args.derive_missing_answer:
        attacks = _fill_missing_answers(
            attacks,
            GoldenFactAnswerAgent(client=client, max_output_tokens=args.answer_max_output_tokens),
        )

    pipeline = MemoryRefactoringPipeline(
        config=AlgorithmConfig(
            tau=args.tau,
            top_k=args.top_k,
            min_score=args.min_score,
            regression_sample_size=args.regression_sample_size,
            proposal_count=args.proposal_count,
            seed=args.seed,
            commit_threshold=args.commit_threshold,
            memory_archive_dir=args.memory_archive_dir,
            exp_name=args.exp_name,
        ),
        answer_agent=RetrievedMemoryAnswerAgent(
            client=client,
            max_output_tokens=args.answer_max_output_tokens,
        ),
        judge=AnswerEquivalenceJudge(
            client=client,
            max_output_tokens=args.judge_max_output_tokens,
            use_llm=not args.skip_llm_judge,
        ),
        policy=PromptMemoryRefactoringPolicy(
            client=client,
            max_output_tokens=args.policy_max_output_tokens,
        ),
    )

    result = pipeline.run_stream(
        attacks=attacks,
        memory_store=MemoryStore() if args.empty_memory else MemoryStore.load(args.memory),
        success_pool=SuccessPool.load(args.success_pool) if args.success_pool else SuccessPool(),
        high_priority_buffer=(
            HighPriorityBuffer.load(args.high_priority_buffer)
            if args.high_priority_buffer
            else HighPriorityBuffer()
        ),
    )

    result.memory_store.save(args.memory_output)
    if args.success_pool_output or args.success_pool:
        result.success_pool.save(args.success_pool_output or args.success_pool)
    if args.high_priority_buffer_output or args.high_priority_buffer:
        result.high_priority_buffer.save(
            args.high_priority_buffer_output or args.high_priority_buffer
        )
    _write_json(args.trace_output, result.to_dict())
    if args.attack_trace_output and attack_preparation is not None:
        _write_json(args.attack_trace_output, attack_preparation.to_dict())
    print(json.dumps(_summary(result.to_dict(), attack_preparation), ensure_ascii=False, indent=2))


def _fill_missing_answers(attacks, answer_agent):
    filled = []
    for attack in attacks:
        item = dict(attack)
        if not item.get("answer") and item.get("golden_facts"):
            item["answer"] = answer_agent.answer(
                str(item.get("question", "")),
                item.get("golden_facts", []),
            )["answer"]
        filled.append(item)
    return filled


def _prepare_attacks(args, client):
    graphs = load_graphs(args.graphs)
    if args.max_graphs >= 0:
        graphs = graphs[: args.max_graphs]

    routes = []
    attacks = []
    oracle_discarded = []
    policy_names = parse_policy_names(args.policies)
    for route_round in range(max(1, args.routes_per_graph)):
        round_args = _route_args(args, seed_offset=route_round)
        prepared = prepare_attacks_from_graphs(
            graphs=graphs,
            policies=build_policies(round_args, client=client if args.use_llm_rerank else None),
            policy_names=policy_names,
            attacker=FrozenLLMAttacker(
                client=client,
                max_output_tokens=args.attack_max_output_tokens,
            ),
            verifier=(
                None
                if args.skip_verification
                else GoldenFactVerifier(client=client, max_output_tokens=args.verify_max_output_tokens)
            ),
            oracle_answer_agent=GoldenFactAnswerAgent(
                client=client,
                max_output_tokens=args.answer_max_output_tokens,
            ),
            oracle_judge=AnswerEquivalenceJudge(
                client=client,
                max_output_tokens=args.judge_max_output_tokens,
                use_llm=not args.skip_llm_judge,
            ),
            keep_failed_verification=args.keep_failed_verification,
            keep_failed_oracle=args.keep_failed_oracle,
        )
        routes.extend(prepared.routes)
        attacks.extend(prepared.attacks)
        oracle_discarded.extend(prepared.oracle_discarded)
    return type(prepared)(routes=routes, attacks=attacks, oracle_discarded=oracle_discarded)


def _summary(payload, attack_preparation=None):
    steps = payload["steps"]
    summary = {
        "steps": len(steps),
        "initial_defense_success": sum(1 for item in steps if item["status"] == "initial_defense_success"),
        "refactor_committed": sum(1 for item in steps if item["status"] == "refactor_committed"),
        "refactor_rolled_back": sum(1 for item in steps if item["status"] == "refactor_rolled_back"),
        "final_memory_size": len(payload["final_memory"]["memories"]),
        "success_pool_size": len(payload["success_pool"]["successes"]),
        "high_priority_buffer_size": len(payload["high_priority_buffer"]["attacks"]),
    }
    if attack_preparation is not None:
        summary.update(
            {
                "generated_routes": len(attack_preparation.routes),
                "oracle_passed_attacks": len(attack_preparation.attacks),
                "oracle_discarded_attacks": len(attack_preparation.oracle_discarded),
            }
        )
    return summary


def _route_args(args, seed_offset=0):
    return argparse.Namespace(
        seed=args.seed_routes + seed_offset,
        random_walk_steps=args.random_walk_steps,
        random_walk_min_nodes=args.random_walk_min_nodes,
        random_walk_attempts=args.random_walk_attempts,
        route_top_k=args.route_top_k,
        route_max_output_tokens=args.route_max_output_tokens,
    )


def _write_json(path: str, payload) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
