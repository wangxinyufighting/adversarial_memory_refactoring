import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .attacker import FrozenLLMAttacker
from .llm import OpenAIChatClient
from .verifier import GoldenFactVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adversarial Q/A pairs from route JSON.")
    parser.add_argument("--input", required=True, help="Route JSON produced by case_graph.route_cli.")
    parser.add_argument("--output", required=True, help="Path to write attack JSON output.")
    parser.add_argument("--attack-max-output-tokens", type=int, default=700)
    parser.add_argument("--verify-max-output-tokens", type=int, default=300)
    parser.add_argument("--skip-verification", action="store_true", help="Do not verify answers against golden facts.")
    parser.add_argument(
        "--keep-failed-verification",
        action="store_true",
        help="Keep attacks that fail answer support or ambiguity verification.",
    )
    return parser.parse_args()


def load_route_items(input_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return payload.get("routes", [])


def generate_attacks(
    route_items: List[Dict[str, Any]],
    attacker: FrozenLLMAttacker,
    verifier: Optional[GoldenFactVerifier] = None,
    keep_failed_verification: bool = False,
) -> List[Dict[str, Any]]:
    attacks = []
    for item in route_items:
        attack = attacker.generate_from_route(
            case_id=str(item.get("case_id", "")),
            route=item.get("route", {}),
            golden_facts=item.get("golden_facts", []),
        ).to_dict()
        if verifier is not None:
            attack["verification"] = verifier.verify(
                question=attack["question"],
                answer=attack["answer"],
                golden_facts=attack["golden_facts"],
                route=attack["route"],
            )
            if not keep_failed_verification and not attack["verification"]["supported"]:
                continue
        attacks.append(attack)
    return attacks


def main() -> None:
    args = parse_args()
    client = OpenAIChatClient.from_env()
    attacker = FrozenLLMAttacker(
        client=client,
        max_output_tokens=args.attack_max_output_tokens,
    )
    verifier = None
    if not args.skip_verification:
        verifier = GoldenFactVerifier(client=client, max_output_tokens=args.verify_max_output_tokens)
    attacks = generate_attacks(
        load_route_items(Path(args.input)),
        attacker,
        verifier=verifier,
        keep_failed_verification=args.keep_failed_verification,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"attacks": attacks}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(attacks)} attacks to {output_path}")


if __name__ == "__main__":
    main()
