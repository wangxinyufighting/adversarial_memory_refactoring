import argparse
import json
from pathlib import Path

from .baseline import AnswerEquivalenceJudge, GoldenFactAnswerAgent, run_baseline_sanity_test
from .llm import OpenAIChatClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline sanity test with golden facts as context.")
    parser.add_argument("--input", required=True, help="Attack JSON produced by attack generation.")
    parser.add_argument("--output", required=True, help="Path to write attacks that pass the baseline sanity test.")
    parser.add_argument("--answer-max-output-tokens", type=int, default=200)
    parser.add_argument("--judge-max-output-tokens", type=int, default=200)
    parser.add_argument("--skip-llm-judge", action="store_true", help="Use string matching only for answer correctness.")
    parser.add_argument("--keep-failed", action="store_true", help="Keep failed samples for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    attacks = payload.get("attacks", [])

    client = OpenAIChatClient.from_env()
    answer_agent = GoldenFactAnswerAgent(client=client, max_output_tokens=args.answer_max_output_tokens)
    judge = AnswerEquivalenceJudge(
        client=client,
        max_output_tokens=args.judge_max_output_tokens,
        use_llm=not args.skip_llm_judge,
    )
    passed, discarded = run_baseline_sanity_test(
        attacks,
        answer_agent=answer_agent,
        judge=judge,
        keep_failed=args.keep_failed,
    )

    output = {
        "attacks": passed,
        "summary": {
            "input": len(attacks),
            "passed": sum(1 for attack in passed if attack.get("baseline_sanity", {}).get("correct")),
            "discarded": len(discarded),
            "kept_failed": args.keep_failed,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Baseline sanity test: {output['summary']['passed']}/{len(attacks)} passed; "
        f"discarded {len(discarded)}."
    )
    print(f"Wrote {len(passed)} attacks to {output_path}")


if __name__ == "__main__":
    main()
