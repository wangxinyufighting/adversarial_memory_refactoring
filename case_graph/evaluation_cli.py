"""CLI for evaluating compressed memories on held-out target questions."""

import argparse
import json
from pathlib import Path

import yaml

from .baseline import AnswerEquivalenceJudge
from .defense import RetrievedMemoryAnswerAgent
from .evaluation import (
    TargetEvaluationResult,
    evaluate_case_target,
    graph_paths,
    load_graph,
    resolve_case_coverage_path,
    resolve_case_memory_path,
    summarize_results,
)
from .retriever import MemoryStore, retriever_config_from_mapping


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
    parser.add_argument(
        "--coverage-dir",
        help="Per-case coverage states. Defaults to the sibling coverage_states directory.",
    )
    parser.add_argument(
        "--allow-incomplete-memory",
        action="store_true",
        help="Evaluate even when construction did not reach coverage certification.",
    )
    parser.add_argument(
        "--training-config",
        default="configs/online_grpo.yaml",
        help="Reuse the training retriever and top-k settings unless explicitly overridden.",
    )
    parser.add_argument("--output", required=True, help="Path to write evaluation JSON.")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-k-points", type=int)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--retriever-type")
    parser.add_argument("--retriever-model-name")
    parser.add_argument("--retriever-embedding-model")
    parser.add_argument("--retriever-retrieval-mode")
    parser.add_argument("--retriever-device")
    parser.add_argument("--retriever-cache-dir")
    parser.add_argument("--retriever-max-length", type=int)
    parser.add_argument("--retriever-require-model", action="store_true", default=None)
    parser.add_argument("--answer-max-output-tokens", type=int, default=200)
    parser.add_argument("--judge-max-output-tokens", type=int, default=200)
    parser.add_argument("--skip-llm-judge", action="store_true", help="Use string matching only for judging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.memory and not args.memory_dir:
        raise SystemExit("Provide --memory or --memory-dir.")

    answer_agent = RetrievedMemoryAnswerAgent(
        client=None,
        max_output_tokens=args.answer_max_output_tokens,
    )
    judge = AnswerEquivalenceJudge(
        client=None,
        max_output_tokens=args.judge_max_output_tokens,
        use_llm=not args.skip_llm_judge,
    )

    shared_memory = MemoryStore.load(args.memory) if args.memory else None
    coverage_dir = _resolve_coverage_dir(args.memory_dir, args.coverage_dir)
    training_config = _load_training_config(args.training_config)
    top_k = _setting(args.top_k, training_config, "top_k", 8)
    top_k_points = _setting(args.top_k_points, training_config, "top_k_points", 32)
    min_score = _setting(args.min_score, training_config, "min_score", 0.0)
    retriever_mapping = {"retriever": dict(training_config.get("retriever", {}) or {})}
    retriever_overrides = {
        "retriever_type": args.retriever_type,
        "retriever_model_name": args.retriever_model_name,
        "retriever_embedding_model": args.retriever_embedding_model,
        "retriever_retrieval_mode": args.retriever_retrieval_mode,
        "retriever_top_k_points": top_k_points,
        "retriever_device": args.retriever_device,
        "retriever_cache_dir": args.retriever_cache_dir,
        "retriever_require_model": args.retriever_require_model,
        "retriever_max_length": args.retriever_max_length,
    }
    retriever_mapping.update(
        {key: value for key, value in retriever_overrides.items() if value is not None}
    )
    retriever_config = retriever_config_from_mapping(retriever_mapping)
    results = []
    for graph_path in graph_paths(args.graphs):
        graph = load_graph(graph_path)
        case_id = str(graph.get("case_id", graph_path.stem.replace(".case_graph", "")))
        memory_path = args.memory or ""
        memory_store = shared_memory
        coverage_state = _load_coverage_state(case_id, coverage_dir)

        if memory_store is None:
            resolved = resolve_case_memory_path(case_id, args.memory_dir)
            if resolved is None:
                results.append(_missing_memory_result(graph, case_id))
                continue
            memory_path = str(resolved)
            memory_store = MemoryStore.load(resolved)

        if not args.allow_incomplete_memory and args.memory_dir:
            if not coverage_state:
                results.append(
                    _coverage_blocked_result(
                        graph,
                        case_id,
                        "missing_coverage_state",
                        "No coverage state found; memory completeness cannot be certified.",
                        {},
                    )
                )
                continue
            if coverage_state.get("completion_status") != "done":
                results.append(
                    _coverage_blocked_result(
                        graph,
                        case_id,
                        "incomplete_memory",
                        "Memory construction did not reach coverage certification.",
                        coverage_state,
                    )
                )
                continue

        results.append(
            evaluate_case_target(
                graph=graph,
                memory_store=memory_store,
                answer_agent=answer_agent,
                judge=judge,
                top_k=top_k,
                min_score=min_score,
                memory_path=memory_path,
                retriever_config=retriever_config,
                coverage_state=coverage_state,
            )
        )

    payload = {
        "summary": summarize_results(results),
        "retriever": retriever_config,
        "top_k": top_k,
        "top_k_points": top_k_points,
        "min_score": min_score,
        "results": [item.to_dict() for item in results],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def _load_training_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return dict(payload) if isinstance(payload, dict) else {}


def _setting(cli_value, training_config: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    value = training_config.get(key, default)
    return default if value is None else value


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


def _resolve_coverage_dir(memory_dir: str | None, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    if not memory_dir:
        return None
    candidate = Path(memory_dir).parent / "coverage_states"
    return candidate


def _load_coverage_state(case_id: str, coverage_dir: Path | None) -> dict:
    if coverage_dir is None:
        return {}
    path = resolve_case_coverage_path(case_id, coverage_dir)
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _coverage_blocked_result(
    graph: dict,
    case_id: str,
    status: str,
    reason: str,
    coverage_state: dict,
) -> TargetEvaluationResult:
    target = graph.get("target") or {}
    return TargetEvaluationResult(
        case_id=case_id,
        question=str(target.get("question", "")),
        gold_answer=str(target.get("answer", "")),
        candidate_answer="",
        correct=False,
        retrieved_memories=[],
        answer_result={"answer": "", "reason": reason},
        judge={"correct": False, "method": status, "reason": reason},
        status=status,
        coverage_state=coverage_state,
    )


if __name__ == "__main__":
    main()
