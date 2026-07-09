"""Core evaluation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from baselines.base import MemoryBaseline, MissingMemoryError
from case_graph.baseline import AnswerEquivalenceJudge
from case_graph.defense import RetrievedMemoryAnswerAgent
from case_graph.evaluation import TargetEvaluationResult, evaluate_case_target, graph_paths, load_graph

from .metrics import (
    answer_source_ids,
    compute_memory_stats,
    compute_retrieval_metrics,
    summarize_evaluation_items,
)


def evaluate_baseline(
    graphs: str | Path | Iterable[str | Path],
    baseline: MemoryBaseline,
    answer_agent: RetrievedMemoryAnswerAgent,
    judge: AnswerEquivalenceJudge,
    top_k: int = 5,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate one memory baseline with the shared retriever and answer agent."""
    items = []
    graph_file_list = _graph_file_list(graphs)
    for graph_path in graph_file_list:
        graph = load_graph(graph_path)
        case_id = str(graph.get("case_id", Path(graph_path).stem.replace(".case_graph", "")))
        try:
            memory_store = baseline.build_memory(graph)
        except MissingMemoryError as exc:
            items.append(_missing_memory_item(graph, baseline, str(exc)))
            continue

        result = evaluate_case_target(
            graph=graph,
            memory_store=memory_store,
            answer_agent=answer_agent,
            judge=judge,
            top_k=top_k,
            min_score=min_score,
            memory_path=baseline.name,
        )
        item = result.to_dict()
        gold_sources = answer_source_ids(graph)
        item.update(
            {
                "baseline": baseline.name,
                "baseline_metadata": baseline.metadata(),
                "graph_path": str(graph_path),
                "answer_source_ids": gold_sources,
                "retrieval_metrics": compute_retrieval_metrics(
                    item["retrieved_memories"],
                    gold_sources,
                ),
                "memory_stats": compute_memory_stats(memory_store),
            }
        )
        items.append(item)

    return {
        "baseline": baseline.name,
        "baseline_metadata": baseline.metadata(),
        "summary": summarize_evaluation_items(items),
        "results": items,
    }


def evaluate_suite(
    graphs: str | Path | Iterable[str | Path],
    baselines: Iterable[MemoryBaseline],
    answer_agent: RetrievedMemoryAnswerAgent,
    judge: AnswerEquivalenceJudge,
    top_k: int = 5,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate multiple baselines under identical retrieval and answer settings."""
    baseline_results = [
        evaluate_baseline(
            graphs=graphs,
            baseline=baseline,
            answer_agent=answer_agent,
            judge=judge,
            top_k=top_k,
            min_score=min_score,
        )
        for baseline in baselines
    ]
    return {
        "evaluation_protocol": {
            "retriever": "case_graph.retriever.FrozenBM25Retriever",
            "answer_backbone": "case_graph.defense.RetrievedMemoryAnswerAgent",
            "judge": "case_graph.baseline.AnswerEquivalenceJudge",
            "top_k": top_k,
            "min_score": min_score,
        },
        "baselines": baseline_results,
        "leaderboard": _leaderboard(baseline_results),
    }


def _leaderboard(baseline_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for result in baseline_results:
        summary = result["summary"]
        rows.append(
            {
                "baseline": result["baseline"],
                "accuracy": summary["accuracy"],
                "avg_source_recall": summary["avg_source_recall"],
                "avg_source_mrr": summary["avg_source_mrr"],
                "avg_memory_chunks": summary["avg_memory_chunks"],
                "avg_approx_memory_tokens": summary["avg_approx_memory_tokens"],
            }
        )
    return sorted(rows, key=lambda row: (row["accuracy"], row["avg_source_recall"]), reverse=True)


def _graph_file_list(graphs: str | Path | Iterable[str | Path]) -> List[Path]:
    if isinstance(graphs, (str, Path)):
        return graph_paths(graphs)
    paths: List[Path] = []
    for item in graphs:
        paths.extend(graph_paths(item))
    return paths


def _missing_memory_item(
    graph: Dict[str, Any],
    baseline: MemoryBaseline,
    reason: str,
) -> Dict[str, Any]:
    case_id = str(graph.get("case_id", "unknown"))
    target = graph.get("target") or {}
    result = TargetEvaluationResult(
        case_id=case_id,
        question=str(target.get("question", "")),
        gold_answer=str(target.get("answer", "")),
        candidate_answer="",
        correct=False,
        retrieved_memories=[],
        answer_result={"answer": "", "reason": reason},
        judge={"correct": False, "method": "missing_memory", "reason": reason},
        status="missing_memory",
    ).to_dict()
    result.update(
        {
            "baseline": baseline.name,
            "baseline_metadata": baseline.metadata(),
            "answer_source_ids": answer_source_ids(graph),
            "retrieval_metrics": compute_retrieval_metrics([], answer_source_ids(graph)),
            "memory_stats": {
                "memory_chunks": 0.0,
                "memory_chars": 0.0,
                "approx_memory_tokens": 0.0,
                "linked_questions": 0.0,
            },
        }
    )
    return result
