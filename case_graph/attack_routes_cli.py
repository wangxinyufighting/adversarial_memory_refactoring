import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .attacker import FrozenLLMAttacker
from .llm import OpenAIChatClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adversarial questions from route JSON.")
    parser.add_argument("--input", required=True, help="Route JSON produced by case_graph.route_cli.")
    parser.add_argument("--output", required=True, help="Path to write attack JSON output.")
    parser.add_argument("--attack-max-output-tokens", type=int, default=700)
    return parser.parse_args()


def load_route_items(input_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return payload.get("routes", [])


def generate_attacks(route_items: List[Dict[str, Any]], attacker: FrozenLLMAttacker) -> List[Dict[str, Any]]:
    return [
        attacker.generate_from_route(
            case_id=str(item.get("case_id", "")),
            route=item.get("route", {}),
        ).to_dict()
        for item in route_items
    ]


def main() -> None:
    args = parse_args()
    attacker = FrozenLLMAttacker(
        client=OpenAIChatClient.from_env(),
        max_output_tokens=args.attack_max_output_tokens,
    )
    attacks = generate_attacks(load_route_items(Path(args.input)), attacker)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"attacks": attacks}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(attacks)} attacks to {output_path}")


if __name__ == "__main__":
    main()
