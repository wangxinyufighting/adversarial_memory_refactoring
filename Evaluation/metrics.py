"""Metrics for controlled memory evaluation."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Set

from case_graph.retriever import MemoryStore


def answer_source_ids(graph: Dict[str, Any]) -> List[str]:
    target = graph.get("target") or {}
    values = target.get("answer_source_ids") or graph.get("answer_source_ids") or []
    if isinstance(values, str):
        values = [values]
    return [str(item) for item in values if str(item)]


def compute_retrieval_metrics(
    retrieved_memories: Sequence[Dict[str, Any]],
    gold_source_ids: Iterable[str],
) -> Dict[str, float]:
    """Compute source-aware retrieval metrics from serialized RetrievalHit rows."""
    gold = {str(item) for item in gold_source_ids if str(item)}
    if not gold:
        return {
            "has_gold_source": 0.0,
            "source_recall": 0.0,
            "source_hit": 0.0,
            "source_mrr": 0.0,
            "source_ndcg": 0.0,
        }

    covered: Set[str] = set()
    first_rank = 0
    dcg = 0.0
    for index, hit in enumerate(retrieved_memories, start=1):
        matched = _hit_source_ids(hit) & gold
        if matched:
            covered.update(matched)
            if not first_rank:
                first_rank = index
            dcg += 1.0 / math.log2(index + 1)

    ideal_relevant = min(len(gold), len(retrieved_memories))
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_relevant + 1))
    return {
        "has_gold_source": 1.0,
        "source_recall": len(covered) / len(gold),
        "source_hit": float(bool(covered)),
        "source_mrr": (1.0 / first_rank) if first_rank else 0.0,
        "source_ndcg": (dcg / idcg) if idcg else 0.0,
    }


def compute_memory_stats(memory_store: MemoryStore) -> Dict[str, float]:
    chars = sum(len(chunk.content) for chunk in memory_store.chunks)
    linked_questions = sum(len(chunk.linked_questions) for chunk in memory_store.chunks)
    return {
        "memory_chunks": float(len(memory_store.chunks)),
        "memory_chars": float(chars),
        "approx_memory_tokens": float(chars / 4.0),
        "linked_questions": float(linked_questions),
    }


def summarize_evaluation_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [item for item in items if item.get("status") == "evaluated"]
    total = len(evaluated)
    correct = sum(1 for item in evaluated if item.get("correct"))
    return {
        "total": len(items),
        "evaluated": total,
        "missing_memory": sum(1 for item in items if item.get("status") == "missing_memory"),
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "avg_source_recall": _mean_metric(evaluated, "retrieval_metrics", "source_recall"),
        "avg_source_hit": _mean_metric(evaluated, "retrieval_metrics", "source_hit"),
        "avg_source_mrr": _mean_metric(evaluated, "retrieval_metrics", "source_mrr"),
        "avg_source_ndcg": _mean_metric(evaluated, "retrieval_metrics", "source_ndcg"),
        "avg_memory_chunks": _mean_metric(evaluated, "memory_stats", "memory_chunks"),
        "avg_memory_chars": _mean_metric(evaluated, "memory_stats", "memory_chars"),
        "avg_approx_memory_tokens": _mean_metric(evaluated, "memory_stats", "approx_memory_tokens"),
    }


def _hit_source_ids(hit: Dict[str, Any]) -> Set[str]:
    ids = {str(hit.get("memory_id", ""))}
    metadata = hit.get("metadata") or {}
    for key in ("source_id", "session_id", "chunk_id"):
        value = metadata.get(key)
        if value:
            ids.add(str(value))
    for key in ("source_ids", "all_ids", "session_ids", "chunk_ids"):
        values = metadata.get(key) or []
        if isinstance(values, str):
            values = [values]
        ids.update(str(item) for item in values if str(item))
    return {item for item in ids if item}


def _mean_metric(items: List[Dict[str, Any]], section: str, key: str) -> float:
    values = [
        float(item.get(section, {}).get(key, 0.0))
        for item in items
        if section in item
    ]
    return (sum(values) / len(values)) if values else 0.0
