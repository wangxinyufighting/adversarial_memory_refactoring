import argparse
import json
from pathlib import Path

from .baseline import AnswerEquivalenceJudge
from .defense import RetrievedMemoryAnswerAgent, SuccessPool, run_initial_defense
from .llm import OpenAIChatClient
from .retriever import MemoryStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-three initial defense over old memories.")
    parser.add_argument("--memory", required=True, help="Current memory store Mt JSON.")
    parser.add_argument("--question", required=True, help="Attack question Q.")
    parser.add_argument("--answer", required=True, help="Gold answer for judging Answer Agent output.")
    parser.add_argument("--output", help="Optional path to write defense result JSON.")
    parser.add_argument("--memory-output", help="Optional path to write Mt after successful binding.")
    parser.add_argument("--success-pool", help="Optional path to load and update the global success pool.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--answer-max-output-tokens", type=int, default=200)
    parser.add_argument("--judge-max-output-tokens", type=int, default=200)
    parser.add_argument("--skip-llm-judge", action="store_true", help="Use string matching only for correctness.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    memory_store = MemoryStore.load(args.memory)
    success_pool = SuccessPool.load(args.success_pool) if args.success_pool else None
    client = OpenAIChatClient.from_env()
    outcome = run_initial_defense(
        question=args.question,
        gold_answer=args.answer,
        memory_store=memory_store,
        answer_agent=RetrievedMemoryAnswerAgent(
            client=client,
            max_output_tokens=args.answer_max_output_tokens,
        ),
        judge=AnswerEquivalenceJudge(
            client=client,
            max_output_tokens=args.judge_max_output_tokens,
            use_llm=not args.skip_llm_judge,
        ),
        success_pool=success_pool,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    if outcome.correct and args.memory_output:
        memory_store.save(args.memory_output)
    if outcome.correct and args.success_pool and success_pool is not None:
        success_pool.save(args.success_pool)

    payload = outcome.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
