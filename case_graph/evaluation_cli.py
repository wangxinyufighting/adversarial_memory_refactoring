"""CLI for evaluating compressed memories on held-out target questions."""

import argparse
import json
from pathlib import Path

from .baseline import AnswerEquivalenceJudge
from .defense import RetrievedMemoryAnswerAgent
from .evaluation import (
    TargetEvaluationResult,
    evaluate_case_target,
    graph_paths,
    load_graph,
    resolve_case_memory_path,
    summarize_results,
)
from .llm import OpenAIChatClient
from .retriever import MemoryStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate compressed memory on CaseGraph target questions."
    )
    parser.add_argument("--graphs", required=True, help="CaseGraph JSON file or directory.")
    parser.add_argument("--memory", help="Single compressed memory store for all cases.")
    parser.add_argument(
        "--memory-dir",
        help="Directory with per-case compressed memory JSON files, e.g. checkpoint_final/memory_states.",
    )
    parser.add_argument("--output", required=True, help="Path to write evaluation JSON.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--answer-max-output-tokens", type=int, default=200)
    parser.add_argument("--judge-max-output-tokens", type=int, default=200)
    parser.add_argument("--skip-llm-judge", action="store_true", help="Use string matching only for judging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.memory and not args.memory_dir:
        raise SystemExit("Provide --memory or --memory-dir.")

    client = OpenAIChatClient.from_env()
    answer_agent = RetrievedMemoryAnswerAgent(
        client=client,
        max_output_tokens=args.answer_max_output_tokens,
    )
    judge = AnswerEquivalenceJudge(
        client=client,
        max_output_tokens=args.judge_max_output_tokens,
        use_llm=not args.skip_llm_judge,
    )

    shared_memory = MemoryStore.load(args.memory) if args.memory else None
    results = []
    for graph_path in graph_paths(args.graphs):
        graph = load_graph(graph_path)
        case_id = str(graph.get("case_id", graph_path.stem.replace(".case_graph", "")))
        memory_path = args.memory or ""
        memory_store = shared_memory

        if memory_store is None:
            resolved = resolve_case_memory_path(case_id, args.memory_dir)
            if resolved is None:
                results.append(_missing_memory_result(graph, case_id))
                continue
            memory_path = str(resolved)
            memory_store = MemoryStore.load(resolved)

        results.append(
            evaluate_case_target(
                graph=graph,
                memory_store=memory_store,
                answer_agent=answer_agent,
                judge=judge,
                top_k=args.top_k,
                min_score=args.min_score,
                memory_path=memory_path,
            )
        )

    payload = {
        "summary": summarize_results(results),
        "results": [item.to_dict() for item in results],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def _missing_memory_result(graph: dict, case_id: str) -> TargetEvaluationResult:
    target = graph.get("target") or {}
    return TargetEvaluationResult(
        case_id=case_id,
        question=str(target.get("question", "")),
        gold_answer=str(target.get("answer", "")),
        candidate_answer="",
        correct=False,
        retrieved_memories=[],
        answer_result={"answer": "", "reason": "No compressed memory found for case."},
        judge={"correct": False, "method": "missing_memory", "reason": "No compressed memory found."},
        status="missing_memory",
    )


if __name__ == "__main__":
    main()
