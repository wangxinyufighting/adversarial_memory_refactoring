import argparse
import json
from pathlib import Path
from typing import Dict, List

from .attacker import FrozenLLMAttacker
from .llm import OpenAIChatClient
from .routing import (
    FeatureScoredLLMRerankRoutingPolicy,
    HeuristicRoutingPolicy,
    RandomWalkRoutingPolicy,
)
from .verifier import GoldenFactVerifier


POLICY_NAMES = ("random_walk", "heuristic", "feature_scored_llm_rerank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adversarial Q/A pairs from CaseGraph JSON files.")
    parser.add_argument("--input", required=True, help="A case_graph JSON file or a directory containing them.")
    parser.add_argument("--output", required=True, help="Path to write attack JSON output.")
    parser.add_argument(
        "--policies",
        default=",".join(POLICY_NAMES),
        help=f"Comma-separated routing policies. Available: {', '.join(POLICY_NAMES)}.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for random walk routing.")
    parser.add_argument("--random-walk-steps", type=int, default=3, help="Maximum random walk steps.")
    parser.add_argument("--random-walk-min-nodes", type=int, default=1, help="Preferred minimum random-walk nodes.")
    parser.add_argument("--random-walk-attempts", type=int, default=8, help="Attempts used to satisfy min nodes.")
    parser.add_argument("--route-top-k", type=int, default=5, help="Top feature-scored routes sent to reranker.")
    parser.add_argument("--route-max-output-tokens", type=int, default=300)
    parser.add_argument("--attack-max-output-tokens", type=int, default=700)
    parser.add_argument("--verify-max-output-tokens", type=int, default=300)
    parser.add_argument("--skip-verification", action="store_true", help="Do not verify answers against golden facts.")
    parser.add_argument(
        "--keep-failed-verification",
        action="store_true",
        help="Keep attacks that fail answer support or ambiguity verification.",
    )
    return parser.parse_args()


def graph_paths(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.case_graph.json"))
    return [input_path]


def build_policies(args: argparse.Namespace, client: OpenAIChatClient) -> Dict[str, object]:
    return {
        "random_walk": RandomWalkRoutingPolicy(
            seed=args.seed,
            max_steps=args.random_walk_steps,
            min_nodes=args.random_walk_min_nodes,
            attempts=args.random_walk_attempts,
        ),
        "heuristic": HeuristicRoutingPolicy(),
        "feature_scored_llm_rerank": FeatureScoredLLMRerankRoutingPolicy(
            client=client,
            top_k=args.route_top_k,
            max_output_tokens=args.route_max_output_tokens,
        ),
    }


def main() -> None:
    args = parse_args()
    client = OpenAIChatClient.from_env()
    attacker = FrozenLLMAttacker(client=client, max_output_tokens=args.attack_max_output_tokens)
    verifier = None
    if not args.skip_verification:
        verifier = GoldenFactVerifier(client=client, max_output_tokens=args.verify_max_output_tokens)
    policies = build_policies(args, client)
    selected_policy_names = [name.strip() for name in args.policies.split(",") if name.strip()]

    attacks = []
    for path in graph_paths(Path(args.input)):
        graph = json.loads(path.read_text(encoding="utf-8"))
        for policy_name in selected_policy_names:
            route = policies[policy_name].select_route(graph)
            attack = attacker.generate(graph, route).to_dict()
            if verifier is not None:
                attack["verification"] = verifier.verify(
                    question=attack["question"],
                    answer=attack["answer"],
                    golden_facts=attack["golden_facts"],
                    route=attack["route"],
                )
                if not args.keep_failed_verification and not attack["verification"]["supported"]:
                    continue
            attacks.append(attack)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"attacks": attacks}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(attacks)} attacks to {output_path}")


if __name__ == "__main__":
    main()
