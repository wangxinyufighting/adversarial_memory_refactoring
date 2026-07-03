import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm import OpenAIChatClient
from .routing import (
    FeatureScoredLLMRerankRoutingPolicy,
    HeuristicRoutingPolicy,
    RandomWalkRoutingPolicy,
)


POLICY_NAMES = ("random_walk", "heuristic", "feature_scored_llm_rerank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate attack routes from CaseGraph JSON files.")
    parser.add_argument("--input", required=True, help="A case_graph JSON file or a directory containing them.")
    parser.add_argument("--output", required=True, help="Path to write route JSON output.")
    parser.add_argument(
        "--policies",
        default=",".join(POLICY_NAMES),
        help=f"Comma-separated routing policies. Available: {', '.join(POLICY_NAMES)}.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for random walk routing.")
    parser.add_argument("--random-walk-steps", type=int, default=3, help="Maximum random walk steps.")
    parser.add_argument("--route-top-k", type=int, default=5, help="Top feature-scored routes considered.")
    parser.add_argument("--route-max-output-tokens", type=int, default=300)
    parser.add_argument(
        "--use-llm-rerank",
        action="store_true",
        help="Use the configured LLM for feature_scored_llm_rerank. Off by default to inspect routes cheaply.",
    )
    return parser.parse_args()


def graph_paths(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.case_graph.json"))
    return [input_path]


def parse_policy_names(policy_text: str) -> List[str]:
    names = [name.strip() for name in policy_text.split(",") if name.strip()]
    unknown = sorted(set(names) - set(POLICY_NAMES))
    if unknown:
        raise ValueError(f"Unknown routing policies: {', '.join(unknown)}")
    return names


def build_policies(args: argparse.Namespace, client: Optional[OpenAIChatClient] = None) -> Dict[str, object]:
    return {
        "random_walk": RandomWalkRoutingPolicy(seed=args.seed, max_steps=args.random_walk_steps),
        "heuristic": HeuristicRoutingPolicy(),
        "feature_scored_llm_rerank": FeatureScoredLLMRerankRoutingPolicy(
            client=client,
            top_k=args.route_top_k,
            max_output_tokens=args.route_max_output_tokens,
        ),
    }


def generate_routes(
    graphs: List[Dict[str, Any]],
    policies: Dict[str, object],
    policy_names: List[str],
) -> List[Dict[str, Any]]:
    routes = []
    for graph in graphs:
        for policy_name in policy_names:
            route = policies[policy_name].select_route(graph)
            routes.append(
                {
                    "case_id": graph.get("case_id"),
                    "route": route.to_evidence_dict(),
                }
            )
    return routes


def main() -> None:
    args = parse_args()
    policy_names = parse_policy_names(args.policies)
    client = OpenAIChatClient.from_env() if args.use_llm_rerank else None
    policies = build_policies(args, client=client)
    graphs = [json.loads(path.read_text(encoding="utf-8")) for path in graph_paths(Path(args.input))]
    routes = generate_routes(graphs, policies, policy_names)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"routes": routes}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(routes)} routes to {output_path}")


if __name__ == "__main__":
    main()
