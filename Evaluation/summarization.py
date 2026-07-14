"""Result summarization and comparison for memory QA evaluation."""

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


def summarize_memory_qa(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    evaluated = [item for item in results if item.get("status") == "evaluated"]
    status_counts = Counter(str(item.get("status") or "unknown") for item in results)
    answer_accuracy = _rate(sum(bool(item.get("correct")) for item in results), total)
    paper_metrics = _paper_metrics_summary(results, answer_accuracy)
    summary = {
        "total": total,
        "evaluated": len(evaluated),
        "correct": sum(bool(item.get("correct")) for item in results),
        "accuracy": answer_accuracy,
        "overall_score": answer_accuracy,
        "overall_score_definition": "end_to_end_answer_accuracy_over_all_selected_cases",
        "evaluated_accuracy": _rate(
            sum(bool(item.get("correct")) for item in evaluated), len(evaluated)
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "answer": _answer_summary(results, evaluated),
        "retrieval": _retrieval_summary(results),
        "memory": _memory_summary(results),
        "paper_metrics": paper_metrics,
        "by_question_type": {},
    }
    question_types = sorted({str(item.get("question_type") or "unknown") for item in results})
    for question_type in question_types:
        group = [item for item in results if str(item.get("question_type") or "unknown") == question_type]
        group_evaluated = [item for item in group if item.get("status") == "evaluated"]
        summary["by_question_type"][question_type] = {
            "total": len(group),
            "evaluated": len(group_evaluated),
            "correct": sum(bool(item.get("correct")) for item in group),
            "accuracy": _rate(sum(bool(item.get("correct")) for item in group), len(group)),
            "exact_match_accuracy": _mean_metric(group, "answer_metrics", "exact_match", missing=0.0),
            "mean_source_recall_at_k": _mean_metric(group, "retrieval_metrics", "source_recall"),
            "mean_memory_source_recall": _mean_metric(
                group, "retrieval_metrics", "memory_source_recall"
            ),
            "paper_metrics": _paper_metrics_summary(
                group,
                _rate(sum(bool(item.get("correct")) for item in group), len(group)),
            ),
        }
    return summary


def compare_memory_qa_results(
    method_results: Sequence[Dict[str, Any]],
    baseline_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build paired diagnostics for refactored memory versus CaseGraph memory."""

    method_by_id = {str(item.get("case_id") or ""): item for item in method_results}
    baseline_by_id = {str(item.get("case_id") or ""): item for item in baseline_results}
    shared_ids = sorted((set(method_by_id) & set(baseline_by_id)) - {""})
    method_summary = summarize_memory_qa(
        [method_by_id[case_id] for case_id in shared_ids]
    )
    baseline_summary = summarize_memory_qa(
        [baseline_by_id[case_id] for case_id in shared_ids]
    )

    paired = Counter()
    for case_id in shared_ids:
        method_correct = bool(method_by_id[case_id].get("correct"))
        baseline_correct = bool(baseline_by_id[case_id].get("correct"))
        if method_correct and baseline_correct:
            paired["both_correct"] += 1
        elif method_correct:
            paired["method_only_correct"] += 1
        elif baseline_correct:
            paired["baseline_only_correct"] += 1
        else:
            paired["both_incorrect"] += 1

    method_paper = method_summary.get("paper_metrics") or {}
    baseline_paper = baseline_summary.get("paper_metrics") or {}
    method_memory = method_summary.get("memory") or {}
    baseline_memory = baseline_summary.get("memory") or {}
    return {
        "method": "refactored_memory",
        "baseline": "case_graph",
        "shared_cases": len(shared_ids),
        "method_missing_cases": sorted(set(baseline_by_id) - set(method_by_id)),
        "baseline_missing_cases": sorted(set(method_by_id) - set(baseline_by_id)),
        "paired_outcomes": {
            key: int(paired.get(key, 0))
            for key in (
                "both_correct",
                "method_only_correct",
                "baseline_only_correct",
                "both_incorrect",
            )
        },
        "accuracy": {
            "method": method_summary.get("overall_score", 0.0),
            "baseline": baseline_summary.get("overall_score", 0.0),
            "delta_method_minus_baseline": (
                float(method_summary.get("overall_score", 0.0))
                - float(baseline_summary.get("overall_score", 0.0))
            ),
        },
        "retrieval_comparison_note": (
            "Accuracy and answer top-k are directly paired. Retrieval deltas are "
            "diagnostic because refactored chunks and graph entity/relationship values "
            "have different granularity."
        ),
        "paper_metric_deltas_method_minus_baseline": {
            key: _optional_delta(method_paper.get(key), baseline_paper.get(key))
            for key in ("R@5", "R@10", "N@5", "N@10")
        },
        "memory": {
            "method_mean_chunks": method_memory.get("mean_chunks_per_case", 0.0),
            "baseline_mean_chunks": baseline_memory.get("mean_chunks_per_case", 0.0),
            "method_mean_chars": method_memory.get("mean_chars_per_case", 0.0),
            "baseline_mean_chars": baseline_memory.get("mean_chars_per_case", 0.0),
            "char_ratio_method_over_baseline": _safe_ratio(
                method_memory.get("mean_chars_per_case"),
                baseline_memory.get("mean_chars_per_case"),
            ),
        },
    }


def build_paper_metric_report(
    method_results: Sequence[Dict[str, Any]],
    baseline_results: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return the compact paper-facing metric table with explicit method names."""

    method_summary = summarize_memory_qa(method_results)
    report: Dict[str, Any] = {
        "protocol": "UnifiedMem_LongMemEval",
        "metric_order": ["R@5", "R@10", "N@5", "N@10", "Answer Accuracy"],
        "OURS": _paper_metric_row(
            method_summary,
            method_name="Defender Refactored Memory",
            memory_source="memory_states/<case_id>.json",
            retrieval_unit="compressed_memory_value",
        ),
    }
    if baseline_results is None:
        return report

    baseline_summary = summarize_memory_qa(baseline_results)
    report["BASELINE"] = _paper_metric_row(
        baseline_summary,
        method_name="Original CaseGraph",
        memory_source="case_graphs/<case_id>.case_graph.json",
        retrieval_unit="case_graph_entity_or_relationship",
    )
    report["DELTA_OURS_MINUS_BASELINE"] = {
        metric: _optional_delta(
            report["OURS"].get(metric),
            report["BASELINE"].get(metric),
        )
        for metric in report["metric_order"]
    }
    report["comparison_note"] = (
        "Answer Accuracy is directly paired over the same cases. R@K and N@K use the "
        "same formulas but different retrieval units for the two memory representations."
    )
    return report


def _answer_summary(
    results: Sequence[Dict[str, Any]],
    evaluated: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "exact_match_accuracy": _mean_metric(
            results, "answer_metrics", "exact_match", missing=0.0
        ),
        "mean_token_f1": _mean_metric(results, "answer_metrics", "token_f1", missing=0.0),
        "unknown_rate_on_evaluated": _mean_metric(
            evaluated, "answer_metrics", "candidate_unknown", missing=0.0
        ),
    }


def _retrieval_summary(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "source_hit_rate_at_k": _mean_metric(results, "retrieval_metrics", "source_hit", missing=0.0),
        "mean_source_recall_at_k": _mean_metric(
            results, "retrieval_metrics", "source_recall", missing=0.0
        ),
        "all_sources_retrieved_rate_at_k": _mean_metric(
            results, "retrieval_metrics", "all_sources_retrieved", missing=0.0
        ),
        "mean_relevant_chunk_precision_at_k": _mean_metric(
            results, "retrieval_metrics", "relevant_chunk_precision", missing=0.0
        ),
        "mrr": _mean_metric(results, "retrieval_metrics", "mrr", missing=0.0),
        "memory_source_hit_rate": _mean_metric(
            results, "retrieval_metrics", "memory_source_hit", missing=0.0
        ),
        "mean_memory_source_recall": _mean_metric(
            results, "retrieval_metrics", "memory_source_recall", missing=0.0
        ),
        "mean_retrieval_recall_given_memory": _mean_metric(
            results,
            "retrieval_metrics",
            "retrieval_recall_given_memory",
            missing=0.0,
        ),
        "gold_answer_text_hit_rate_at_k": _mean_metric(
            results,
            "retrieval_metrics",
            "gold_answer_text_in_retrieved",
            missing=0.0,
        ),
        "gold_answer_text_in_memory_rate": _mean_metric(
            results,
            "retrieval_metrics",
            "gold_answer_text_in_memory",
            missing=0.0,
        ),
    }


def _memory_summary(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "mean_chunks_per_case": _mean_metric(results, "memory_metrics", "memory_chunks", missing=0.0),
        "mean_chars_per_case": _mean_metric(results, "memory_metrics", "memory_chars", missing=0.0),
        "mean_tokens_per_case_approx": _mean_metric(
            results, "memory_metrics", "memory_tokens_approx", missing=0.0
        ),
        "mean_memory_to_raw_char_ratio": _mean_metric(
            results, "memory_metrics", "memory_to_raw_char_ratio"
        ),
        "mean_char_reduction": _mean_metric(results, "memory_metrics", "char_reduction"),
    }


def _paper_metrics_summary(
    results: Sequence[Dict[str, Any]],
    answer_accuracy: float,
) -> Dict[str, Any]:
    eligible = [
        item
        for item in results
        if (item.get("longmemeval_retrieval_metrics") or {}).get("eligible") is True
    ]
    excluded_reasons = Counter()
    for item in results:
        metrics = item.get("longmemeval_retrieval_metrics") or {}
        if metrics.get("eligible") is True:
            continue
        reason = str(metrics.get("exclusion_reason") or "missing_metric_payload")
        excluded_reasons[reason] += 1

    k_values = sorted(
        {
            int(k)
            for item in eligible
            for k in (item.get("longmemeval_retrieval_metrics") or {}).get(
                "k_values", []
            )
            if int(k) > 0
        }
    )
    retrieval_metrics: Dict[str, float] = {}
    for k in k_values:
        for prefix in ("recall_any", "recall_all", "ndcg_any"):
            key = f"{prefix}@{k}"
            retrieval_metrics[key] = _mean_metric(
                eligible,
                "longmemeval_retrieval_metrics",
                key,
                missing=0.0,
            )

    return {
        "protocol": "UnifiedMem_LongMemEval",
        "retrieval_unit": "compressed_memory_value",
        "relevance": "answer_session_provenance_overlap",
        "R@5": retrieval_metrics.get("recall_all@5"),
        "R@10": retrieval_metrics.get("recall_all@10"),
        "N@5": retrieval_metrics.get("ndcg_any@5"),
        "N@10": retrieval_metrics.get("ndcg_any@10"),
        "answer_accuracy": answer_accuracy,
        "overall_score": answer_accuracy,
        "overall_score_definition": "end_to_end_answer_accuracy",
        "retrieval_evaluated_cases": len(eligible),
        "retrieval_excluded_cases": len(results) - len(eligible),
        "retrieval_exclusion_reasons": dict(sorted(excluded_reasons.items())),
        "all_retrieval_metrics": retrieval_metrics,
    }


def _paper_metric_row(
    summary: Dict[str, Any],
    *,
    method_name: str,
    memory_source: str,
    retrieval_unit: str,
) -> Dict[str, Any]:
    paper = summary.get("paper_metrics") or {}
    status_counts = summary.get("status_counts") or {}
    return {
        "method_name": method_name,
        "memory_source": memory_source,
        "retrieval_unit": retrieval_unit,
        "selected_cases": int(summary.get("total") or 0),
        "retrieval_evaluated_cases": int(paper.get("retrieval_evaluated_cases") or 0),
        "answer_error_cases": int(status_counts.get("answer_error") or 0),
        "R@5": paper.get("R@5"),
        "R@10": paper.get("R@10"),
        "N@5": paper.get("N@5"),
        "N@10": paper.get("N@10"),
        "Answer Accuracy": paper.get("answer_accuracy"),
    }


def _mean_metric(
    items: Sequence[Dict[str, Any]],
    section: str,
    key: str,
    *,
    missing: Optional[float] = None,
) -> float:
    values = []
    for item in items:
        value = (item.get(section) or {}).get(key)
        if value is None:
            if missing is None:
                continue
            value = missing
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _optional_delta(left: Any, right: Any) -> Optional[float]:
    if left is None or right is None:
        return None
    try:
        return float(left) - float(right)
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        denominator_value = float(denominator)
        if denominator_value == 0.0:
            return None
        return float(numerator) / denominator_value
    except (TypeError, ValueError):
        return None
