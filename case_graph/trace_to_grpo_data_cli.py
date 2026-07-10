import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .grpo_adapter import write_verl_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract minimal GRPO states from an algorithm trace.")
    parser.add_argument("--trace", required=True, help="Trace JSON produced by run_memory_algorithm.sh.")
    parser.add_argument("--states-output", required=True, help="Output train_states.json path.")
    parser.add_argument("--parquet-output", help="Optional output train.parquet path for verl.")
    parser.add_argument("--attack-trace", help="Optional fallback source for answer/golden_facts.")
    parser.add_argument("--include-committed", action="store_true", help="Include committed refactor steps.")
    parser.add_argument("--include-rolled-back", action="store_true", help="Include rolled-back refactor steps.")
    parser.add_argument("--top-k", type=int, default=5, help="Stored in each state for reward retrieval.")
    args = parser.parse_args()

    trace = _load_json(args.trace)
    attack_lookup = _attack_lookup(_load_json(args.attack_trace)) if args.attack_trace else {}
    states = trace_to_grpo_states(
        trace=trace,
        attack_lookup=attack_lookup,
        include_committed=args.include_committed,
        include_rolled_back=args.include_rolled_back,
        top_k=args.top_k,
    )
    _write_json(args.states_output, states)
    if args.parquet_output:
        write_verl_parquet(states, args.parquet_output)
    print(f"Wrote {len(states)} GRPO states to {args.states_output}")


def trace_to_grpo_states(
    trace: Dict[str, Any],
    attack_lookup: Dict[tuple, Dict[str, Any]] | None = None,
    include_committed: bool = False,
    include_rolled_back: bool = False,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Build S_t = (M_t, Q, F, action, regression_set) records for offline GRPO."""

    attack_lookup = attack_lookup or {}
    states = []
    allowed_statuses = set()
    if include_committed:
        allowed_statuses.add("refactor_committed")
    if include_rolled_back:
        allowed_statuses.add("refactor_rolled_back")
    if not allowed_statuses:
        allowed_statuses = {"refactor_committed", "refactor_rolled_back"}

    for step in trace.get("steps", []):
        if step.get("status") not in allowed_statuses:
            continue
        decision = step.get("decision") or {}
        current_memory = step.get("current_memory")
        if not decision or current_memory is None:
            raise ValueError(
                "Trace step is missing decision/current_memory. Regenerate trace with the updated pipeline."
            )

        key = (str(step.get("case_id", "")), str(step.get("question", "")))
        attack = attack_lookup.get(key, {})
        answer = step.get("answer") or attack.get("answer") or attack.get("gold_answer")
        golden_facts = step.get("golden_facts") or attack.get("golden_facts", [])
        route_evidence = (
            step.get("route_evidence")
            or attack.get("route_evidence")
            or attack.get("route")
            or {}
        )
        if not answer:
            raise ValueError(f"Missing answer for GRPO state: case_id={key[0]} question={key[1]}")

        states.append(
            {
                "case_id": step.get("case_id", ""),
                "step": step.get("step", len(states)),
                "current_memory": current_memory,
                "question": step.get("question", ""),
                "answer": answer,
                "golden_facts": golden_facts,
                "route_evidence": route_evidence,
                "action": decision.get("action", ""),
                "selected_memory_ids": decision.get("selected_memory_ids", []),
                "regression_questions": step.get("regression_questions", []),
                "top_k": top_k,
            }
        )
    return states


def _attack_lookup(payload: Dict[str, Any]) -> Dict[tuple, Dict[str, Any]]:
    attacks = payload.get("attacks", payload.get("oracle_passed_attacks", []))
    return {
        (str(item.get("case_id", "")), str(item.get("question", ""))): item
        for item in attacks
    }


def _load_json(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str, payload: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
