"""End-to-end QA evaluation over fixed, per-case compressed memories."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from case_graph.baseline import AnswerEquivalenceJudge
from case_graph.retriever import (
    MemoryStore,
    RetrievalHit,
    build_memory_retriever,
)

from .metrics import (
    compute_answer_metrics,
    compute_retrieval_metrics,
    memory_size_metrics,
)


class JsonClient(Protocol):
    def complete_json(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        ...


@dataclass(frozen=True)
class MemoryQuestion:
    case_id: str
    question_type: str
    question: str
    question_date: str
    gold_answer: Any
    answer_source_ids: List[str]
    raw_session_chars: int = 0

    @classmethod
    def from_longmemeval(cls, entry: Dict[str, Any]) -> "MemoryQuestion":
        case_id = str(
            entry.get("case_id")
            or entry.get("question_id")
            or entry.get("id")
            or ""
        )
        return cls(
            case_id=case_id,
            question_type=str(entry.get("question_type") or "unknown"),
            question=str(entry.get("question") or ""),
            question_date=str(entry.get("question_date") or ""),
            gold_answer=entry.get("answer", ""),
            answer_source_ids=[
                str(item) for item in entry.get("answer_session_ids", []) if str(item)
            ],
            raw_session_chars=_raw_session_chars(entry.get("haystack_sessions", [])),
        )

    @classmethod
    def from_graph(cls, graph: Dict[str, Any]) -> "MemoryQuestion":
        target = graph.get("target") if isinstance(graph.get("target"), dict) else {}
        return cls(
            case_id=str(graph.get("case_id") or ""),
            question_type=str(
                graph.get("question_type") or target.get("question_type") or "unknown"
            ),
            question=str(target.get("question") or graph.get("question") or ""),
            question_date=str(
                target.get("question_date") or graph.get("question_date") or ""
            ),
            gold_answer=(
                target.get("answer")
                if target.get("answer") is not None
                else graph.get("answer", "")
            ),
            answer_source_ids=[
                str(item)
                for item in (
                    target.get("answer_source_ids")
                    or graph.get("answer_source_ids")
                    or []
                )
                if str(item)
            ],
            raw_session_chars=sum(
                len(str(chunk.get("content") or ""))
                for chunk in graph.get("chunks", [])
                if isinstance(chunk, dict)
            ),
        )


class LongMemEvalMemoryAnswerAgent:
    """Answer a LongMemEval target using only Contriever-retrieved memories."""

    def __init__(self, client: JsonClient, max_output_tokens: int = 256):
        self.client = client
        self.max_output_tokens = int(max_output_tokens)

    def answer(
        self,
        question: str,
        retrieved_memories: Sequence[RetrievalHit],
        *,
        question_date: str = "",
        question_type: str = "",
    ) -> Dict[str, Any]:
        if not question.strip() or not retrieved_memories:
            return {
                "answer": "UNKNOWN",
                "reason": "Missing question or retrieved memories.",
                "evidence_memory_ids": [],
            }

        response = self.client.complete_json(
            system_prompt=(
                "You are a memory question-answering agent. Use only the retrieved "
                "memories, never outside knowledge. Combine evidence across memories "
                "when the question asks for a count, comparison, update, or temporal "
                "reasoning. Return one JSON object with answer, reason, and "
                "evidence_memory_ids."
            ),
            user_prompt=json.dumps(
                {
                    "question": question,
                    "question_date": question_date,
                    "question_type": question_type,
                    "retrieved_memories": [item.to_dict() for item in retrieved_memories],
                    "instruction": (
                        "Answer concisely from the memory content. Treat question_date "
                        "as the reference date for relative time expressions. For counts, "
                        "count distinct supported events rather than mentions. If the "
                        "memories do not support an answer, answer UNKNOWN. Cite only "
                        "retrieved memory_id values that directly support the answer. "
                        "Metadata IDs are provenance labels, not facts."
                    ),
                },
                ensure_ascii=False,
            ),
            max_tokens=self.max_output_tokens,
        )
        evidence_ids = response.get("evidence_memory_ids", [])
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
        return {
            "answer": str(response.get("answer", "")),
            "reason": str(response.get("reason", "")),
            "evidence_memory_ids": [str(item) for item in evidence_ids if str(item)],
        }


def load_longmemeval_questions(path: str | Path) -> Dict[str, MemoryQuestion]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LongMemEval input must be a JSON list.")
    questions: Dict[str, MemoryQuestion] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        question = MemoryQuestion.from_longmemeval(entry)
        if not question.case_id:
            continue
        if question.case_id in questions:
            raise ValueError(f"Duplicate LongMemEval case id: {question.case_id}")
        questions[question.case_id] = question
    return questions


def load_graph_questions(path: str | Path) -> Dict[str, MemoryQuestion]:
    graph_path = Path(path)
    paths = sorted(graph_path.glob("*.case_graph.json")) if graph_path.is_dir() else [graph_path]
    questions: Dict[str, MemoryQuestion] = {}
    for item in paths:
        graph = json.loads(item.read_text(encoding="utf-8"))
        question = MemoryQuestion.from_graph(graph)
        case_id = question.case_id or item.name.replace(".case_graph.json", "")
        if not question.case_id:
            question = MemoryQuestion(
                case_id=case_id,
                question_type=question.question_type,
                question=question.question,
                question_date=question.question_date,
                gold_answer=question.gold_answer,
                answer_source_ids=question.answer_source_ids,
                raw_session_chars=question.raw_session_chars,
            )
        questions[case_id] = question
    return questions


def merge_question_metadata(
    dataset_questions: Dict[str, MemoryQuestion],
    graph_questions: Dict[str, MemoryQuestion],
) -> Dict[str, MemoryQuestion]:
    """Use graph membership while retaining dataset-only type/date metadata."""

    if not graph_questions:
        return dict(dataset_questions)
    merged: Dict[str, MemoryQuestion] = {}
    for case_id, graph_question in graph_questions.items():
        dataset_question = dataset_questions.get(case_id)
        if dataset_question is None:
            merged[case_id] = graph_question
            continue
        if (
            graph_question.question
            and graph_question.question != dataset_question.question
        ):
            raise ValueError(f"Question mismatch between graph and dataset for case {case_id}.")
        if str(graph_question.gold_answer) not in {"", str(dataset_question.gold_answer)}:
            raise ValueError(f"Answer mismatch between graph and dataset for case {case_id}.")
        merged[case_id] = dataset_question
    return merged


def discover_memory_case_ids(memory_dir: str | Path) -> List[str]:
    directory = Path(memory_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Memory directory does not exist: {directory}")
    case_ids = []
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith("_pool.json") or path.name.endswith("_buffer.json"):
            continue
        if not _looks_like_memory_file(path):
            continue
        case_ids.append(path.stem)
    return case_ids


def evaluate_memory_question(
    question: MemoryQuestion,
    memory_store: MemoryStore,
    answer_agent: LongMemEvalMemoryAnswerAgent,
    judge: AnswerEquivalenceJudge,
    *,
    retriever_config: Dict[str, Any],
    top_k: int,
    min_score: float = 0.0,
    memory_path: str = "",
    coverage_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate one case without exposing gold answer metadata to retrieval or QA."""

    retriever = build_memory_retriever(retriever_config, memory_store)
    hits = retriever.retrieve(question.question, top_k=top_k, min_score=min_score)
    full_memory = [chunk.to_dict() for chunk in memory_store.chunks]
    hit_dicts = [hit.to_dict() for hit in hits]
    retrieval_metrics = compute_retrieval_metrics(
        hit_dicts,
        question.answer_source_ids,
        all_memory_items=full_memory,
        gold_answer=question.gold_answer,
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
            retrieval_metrics=retrieval_metrics,
            memory_path=memory_path,
            coverage_state=coverage_state,
            error=reason,
        )

    candidate = str(answer_result.get("answer", ""))
    try:
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
            retrieval_metrics=retrieval_metrics,
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
        retrieval_metrics=retrieval_metrics,
        memory_path=memory_path,
        coverage_state=coverage_state,
    )


def _evaluated_result(
    question: MemoryQuestion,
    *,
    status: str,
    correct: bool,
    candidate: str,
    answer_result: Dict[str, Any],
    judge_result: Dict[str, Any],
    full_memory: List[Dict[str, Any]],
    hit_dicts: List[Dict[str, Any]],
    retrieval_metrics: Dict[str, Any],
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
        "answer_result": answer_result,
        "judge": judge_result,
        "answer_metrics": compute_answer_metrics(question.gold_answer, candidate),
        "retrieval_metrics": retrieval_metrics,
        "evidence_metrics": evidence_metrics,
        "memory_metrics": memory_size_metrics(full_memory, question.raw_session_chars),
        "coverage_state": dict(coverage_state or {}),
    }
    if error:
        result["error"] = error
    return result


def failed_result(
    question: MemoryQuestion,
    *,
    status: str,
    reason: str,
    memory_path: str = "",
    memory_store: Optional[MemoryStore] = None,
    coverage_state: Optional[Dict[str, Any]] = None,
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
        "answer_result": {"answer": "", "reason": reason, "evidence_memory_ids": []},
        "judge": {"correct": False, "method": status, "reason": reason},
        "answer_metrics": compute_answer_metrics(question.gold_answer, ""),
        "retrieval_metrics": retrieval_metrics,
        "evidence_metrics": {},
        "memory_metrics": memory_size_metrics(full_memory, question.raw_session_chars),
        "coverage_state": dict(coverage_state or {}),
    }


def summarize_memory_qa(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    evaluated = [item for item in results if item.get("status") == "evaluated"]
    status_counts = Counter(str(item.get("status") or "unknown") for item in results)
    summary = {
        "total": total,
        "evaluated": len(evaluated),
        "correct": sum(bool(item.get("correct")) for item in results),
        "accuracy": _rate(sum(bool(item.get("correct")) for item in results), total),
        "evaluated_accuracy": _rate(
            sum(bool(item.get("correct")) for item in evaluated), len(evaluated)
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "answer": _answer_summary(results, evaluated),
        "retrieval": _retrieval_summary(results),
        "memory": _memory_summary(results),
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
        }
    return summary


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


def _raw_session_chars(sessions: Iterable[Any]) -> int:
    total = 0
    for session in sessions:
        if not isinstance(session, list):
            continue
        for turn in session:
            if isinstance(turn, dict):
                total += len(str(turn.get("content") or ""))
    return total


def _looks_like_memory_file(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if isinstance(payload, dict):
        return any(key in payload for key in ("memories", "chunks", "memory_chunks"))
    if not isinstance(payload, list):
        return False
    if not payload:
        return True
    first = payload[0]
    return isinstance(first, dict) and bool(
        {"memory_id", "chunk_id", "content", "text", "summary"} & set(first)
    )
