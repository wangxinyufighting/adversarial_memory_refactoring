"""Core evaluation logic for memory QA."""

from typing import Any, Dict, Optional, Sequence

from case_graph.retriever import MemoryStore, build_memory_retriever

from .agents import LongMemEvalAnswerJudge, LongMemEvalMemoryAnswerAgent
from .loaders import MemoryQuestion
from .metrics import (
    compute_answer_metrics,
    compute_longmemeval_retrieval_metrics,
    compute_retrieval_metrics,
    memory_size_metrics,
)


def evaluate_memory_question(
    question: MemoryQuestion,
    memory_store: MemoryStore,
    answer_agent: LongMemEvalMemoryAnswerAgent,
    judge: Any,
    *,
    retriever_config: Dict[str, Any],
    top_k: int,
    retrieval_k_values: Sequence[int] = (5, 10, 20, 30),
    min_score: float = 0.0,
    memory_path: str = "",
    coverage_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate one case without exposing gold answer metadata to retrieval or QA."""

    metric_ks = sorted({int(value) for value in retrieval_k_values if int(value) > 0})
    retrieval_limit = max([int(top_k), *metric_ks])
    retriever = build_memory_retriever(retriever_config, memory_store)
    ranked_hits = retriever.retrieve(
        question.question,
        top_k=retrieval_limit,
        min_score=min_score,
    )
    hits = ranked_hits[: int(top_k)]
    full_memory = [chunk.to_dict() for chunk in memory_store.chunks]
    hit_dicts = [hit.to_dict() for hit in hits]
    ranked_hit_dicts = [hit.to_dict() for hit in ranked_hits]
    retrieval_metrics = compute_retrieval_metrics(
        hit_dicts,
        question.answer_source_ids,
        all_memory_items=full_memory,
        gold_answer=question.gold_answer,
    )
    longmemeval_retrieval_metrics = compute_longmemeval_retrieval_metrics(
        ranked_hit_dicts,
        question.answer_source_ids,
        all_memory_items=full_memory,
        k_values=metric_ks,
        eligible=question.retrieval_eligible,
        exclusion_reason=question.retrieval_exclusion_reason,
    )

    try:
        answer_result = answer_agent.answer(
            question.question,
            hits,
            question_date=question.question_date,
            question_type=question.question_type,
        )
    except Exception as exc:
        reason = str(exc)
        return _evaluated_result(
            question,
            status="answer_error",
            correct=False,
            candidate="",
            answer_result={"answer": "", "reason": reason, "evidence_memory_ids": []},
            judge_result={"correct": False, "method": "answer_error", "reason": reason},
            full_memory=full_memory,
            hit_dicts=hit_dicts,
            ranked_hit_dicts=ranked_hit_dicts,
            retrieval_metrics=retrieval_metrics,
            longmemeval_retrieval_metrics=longmemeval_retrieval_metrics,
            memory_path=memory_path,
            coverage_state=coverage_state,
            error=reason,
        )

    candidate = str(answer_result.get("answer", ""))
    try:
        judge_method = getattr(judge, "judge_longmemeval", None)
        if callable(judge_method):
            judge_result = judge_method(
                case_id=question.case_id,
                question_type=question.question_type,
                question=question.question,
                gold_answer=str(question.gold_answer),
                candidate_answer=candidate,
            )
        else:
            judge_result = judge.judge(
                question=question.question,
                gold_answer=str(question.gold_answer),
                candidate_answer=candidate,
            )
    except Exception as exc:
        reason = str(exc)
        return _evaluated_result(
            question,
            status="judge_error",
            correct=False,
            candidate=candidate,
            answer_result=answer_result,
            judge_result={"correct": False, "method": "judge_error", "reason": reason},
            full_memory=full_memory,
            hit_dicts=hit_dicts,
            ranked_hit_dicts=ranked_hit_dicts,
            retrieval_metrics=retrieval_metrics,
            longmemeval_retrieval_metrics=longmemeval_retrieval_metrics,
            memory_path=memory_path,
            coverage_state=coverage_state,
            error=reason,
        )

    return _evaluated_result(
        question,
        status="evaluated",
        correct=bool(judge_result.get("correct", False)),
        candidate=candidate,
        answer_result=answer_result,
        judge_result=judge_result,
        full_memory=full_memory,
        hit_dicts=hit_dicts,
        ranked_hit_dicts=ranked_hit_dicts,
        retrieval_metrics=retrieval_metrics,
        longmemeval_retrieval_metrics=longmemeval_retrieval_metrics,
        memory_path=memory_path,
        coverage_state=coverage_state,
    )


def failed_result(
    question: MemoryQuestion,
    *,
    status: str,
    reason: str,
    memory_path: str = "",
    memory_store: Optional[MemoryStore] = None,
    coverage_state: Optional[Dict[str, Any]] = None,
    retrieval_k_values: Sequence[int] = (5, 10, 20, 30),
) -> Dict[str, Any]:
    full_memory = (
        [chunk.to_dict() for chunk in memory_store.chunks]
        if memory_store is not None
        else []
    )
    retrieval_metrics = compute_retrieval_metrics(
        [],
        question.answer_source_ids,
        all_memory_items=full_memory,
        gold_answer=question.gold_answer,
    )
    longmemeval_retrieval_metrics = compute_longmemeval_retrieval_metrics(
        [],
        question.answer_source_ids,
        all_memory_items=full_memory,
        k_values=retrieval_k_values,
        eligible=question.retrieval_eligible,
        exclusion_reason=question.retrieval_exclusion_reason,
    )
    return {
        "case_id": question.case_id,
        "question_type": question.question_type,
        "question": question.question,
        "question_date": question.question_date,
        "gold_answer": question.gold_answer,
        "answer_source_ids": list(question.answer_source_ids),
        "candidate_answer": "",
        "correct": False,
        "status": status,
        "error": reason,
        "memory_path": memory_path,
        "retrieved_memories": [],
        "ranked_retrieval_memories": [],
        "answer_result": {"answer": "", "reason": reason, "evidence_memory_ids": []},
        "judge": {"correct": False, "method": status, "reason": reason},
        "answer_metrics": compute_answer_metrics(question.gold_answer, ""),
        "retrieval_metrics": retrieval_metrics,
        "longmemeval_retrieval_metrics": longmemeval_retrieval_metrics,
        "evidence_metrics": {},
        "memory_metrics": memory_size_metrics(full_memory, question.raw_session_chars),
        "coverage_state": dict(coverage_state or {}),
    }


def _evaluated_result(
    question: MemoryQuestion,
    *,
    status: str,
    correct: bool,
    candidate: str,
    answer_result: Dict[str, Any],
    judge_result: Dict[str, Any],
    full_memory: list,
    hit_dicts: list,
    ranked_hit_dicts: list,
    retrieval_metrics: Dict[str, Any],
    longmemeval_retrieval_metrics: Dict[str, Any],
    memory_path: str,
    coverage_state: Optional[Dict[str, Any]],
    error: str = "",
) -> Dict[str, Any]:
    claimed_ids = {
        str(item) for item in answer_result.get("evidence_memory_ids", []) if str(item)
    }
    hit_ids = {str(item.get("memory_id", "")) for item in hit_dicts}
    claimed_hits = [item for item in hit_dicts if str(item.get("memory_id", "")) in claimed_ids]
    evidence_metrics = compute_retrieval_metrics(
        claimed_hits,
        question.answer_source_ids,
        gold_answer=question.gold_answer,
    )
    evidence_metrics.update(
        {
            "claimed_evidence_ids": sorted(claimed_ids),
            "valid_claimed_evidence_ids": sorted(claimed_ids & hit_ids),
            "evidence_id_precision": (
                len(claimed_ids & hit_ids) / len(claimed_ids) if claimed_ids else 0.0
            ),
        }
    )

    result = {
        "case_id": question.case_id,
        "question_type": question.question_type,
        "question": question.question,
        "question_date": question.question_date,
        "gold_answer": question.gold_answer,
        "answer_source_ids": list(question.answer_source_ids),
        "candidate_answer": candidate,
        "correct": bool(correct),
        "status": status,
        "memory_path": memory_path,
        "retrieved_memories": hit_dicts,
        "ranked_retrieval_memories": ranked_hit_dicts,
        "answer_result": answer_result,
        "judge": judge_result,
        "answer_metrics": compute_answer_metrics(question.gold_answer, candidate),
        "retrieval_metrics": retrieval_metrics,
        "longmemeval_retrieval_metrics": longmemeval_retrieval_metrics,
        "evidence_metrics": evidence_metrics,
        "memory_metrics": memory_size_metrics(full_memory, question.raw_session_chars),
        "coverage_state": dict(coverage_state or {}),
    }
    if error:
        result["error"] = error
    return result
