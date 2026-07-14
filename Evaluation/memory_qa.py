"""End-to-end QA evaluation over fixed, per-case compressed memories."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from case_graph.models import EVALUATOR_METADATA_SOURCE
from case_graph.retriever import (
    MemoryChunk,
    MemoryStore,
    RetrievalHit,
    build_memory_retriever,
)

from .metrics import (
    compute_answer_metrics,
    compute_longmemeval_retrieval_metrics,
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


class TextClient(Protocol):
    def complete_text(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
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
    retrieval_eligible: bool = True
    retrieval_exclusion_reason: str = ""

    @classmethod
    def from_longmemeval(cls, entry: Dict[str, Any]) -> "MemoryQuestion":
        case_id = str(
            entry.get("case_id")
            or entry.get("question_id")
            or entry.get("id")
            or ""
        )
        source_ids, retrieval_eligible, exclusion_reason = (
            _longmemeval_retrieval_targets(entry, case_id)
        )
        return cls(
            case_id=case_id,
            question_type=str(entry.get("question_type") or "unknown"),
            question=str(entry.get("question") or ""),
            question_date=str(entry.get("question_date") or ""),
            gold_answer=entry.get("answer", ""),
            answer_source_ids=source_ids,
            raw_session_chars=_raw_session_chars(entry.get("haystack_sessions", [])),
            retrieval_eligible=retrieval_eligible,
            retrieval_exclusion_reason=exclusion_reason,
        )

    @classmethod
    def from_graph(cls, graph: Dict[str, Any]) -> "MemoryQuestion":
        target = graph.get("target") if isinstance(graph.get("target"), dict) else {}
        source_ids = [
            str(item)
            for item in (
                target.get("answer_source_ids")
                or graph.get("answer_source_ids")
                or []
            )
            if str(item)
        ]
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
            answer_source_ids=source_ids,
            raw_session_chars=sum(
                len(str(chunk.get("content") or ""))
                for chunk in graph.get("chunks", [])
                if isinstance(chunk, dict)
            ),
            retrieval_eligible=bool(source_ids),
            retrieval_exclusion_reason=(
                "" if source_ids else "no_retrieval_target_session"
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


class LongMemEvalAnswerJudge:
    """Question-type-aware judge matching the released LongMemEval protocol."""

    def __init__(
        self,
        client: TextClient,
        max_output_tokens: int = 10,
        provider: str = "llm",
    ):
        self.client = client
        self.max_output_tokens = int(max_output_tokens)
        self.provider = str(provider or "llm")

    def judge_longmemeval(
        self,
        *,
        case_id: str,
        question_type: str,
        question: str,
        gold_answer: str,
        candidate_answer: str,
    ) -> Dict[str, Any]:
        prompt = _longmemeval_judge_prompt(
            case_id=case_id,
            question_type=question_type,
            question=question,
            gold_answer=gold_answer,
            candidate_answer=candidate_answer,
        )
        response = self.client.complete_text(
            system_prompt=None,
            user_prompt=prompt,
            max_tokens=self.max_output_tokens,
        ).strip()
        correct = "yes" in response.casefold()
        return {
            "correct": correct,
            "method": f"longmemeval_{self.provider}_judge",
            "reason": response,
            "question_type": question_type,
            "abstention": "_abs" in case_id,
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


def load_case_graphs(path: str | Path) -> Dict[str, Dict[str, Any]]:
    graph_path = Path(path)
    paths = sorted(graph_path.glob("*.case_graph.json")) if graph_path.is_dir() else [graph_path]
    graphs: Dict[str, Dict[str, Any]] = {}
    for item in paths:
        graph = json.loads(item.read_text(encoding="utf-8"))
        if not isinstance(graph, dict):
            raise ValueError(f"CaseGraph must be a JSON object: {item}")
        case_id = str(graph.get("case_id") or item.name.replace(".case_graph.json", ""))
        if not case_id:
            raise ValueError(f"CaseGraph has no case id: {item}")
        if case_id in graphs:
            raise ValueError(f"Duplicate CaseGraph case id: {case_id}")
        graphs[case_id] = graph
    return graphs


def load_graph_questions(path: str | Path) -> Dict[str, MemoryQuestion]:
    questions: Dict[str, MemoryQuestion] = {}
    for case_id, graph in load_case_graphs(path).items():
        question = MemoryQuestion.from_graph(graph)
        if not question.case_id:
            question = MemoryQuestion(
                case_id=case_id,
                question_type=question.question_type,
                question=question.question,
                question_date=question.question_date,
                gold_answer=question.gold_answer,
                answer_source_ids=question.answer_source_ids,
                raw_session_chars=question.raw_session_chars,
                retrieval_eligible=question.retrieval_eligible,
                retrieval_exclusion_reason=question.retrieval_exclusion_reason,
            )
        questions[case_id] = question
    return questions


def case_graph_to_memory_store(graph: Dict[str, Any]) -> MemoryStore:
    """Expose a target-safe CaseGraph as retrievable entity and edge values.

    This is the uncompressed graph baseline. It deliberately excludes raw
    sessions, evaluator target metadata, and safeguard nodes injected from the
    gold answer. The resulting values can therefore be evaluated with exactly
    the same retriever and answer agent as refactored memories without leaking
    the target through graph metadata.
    """

    case_id = str(graph.get("case_id") or "unknown")
    raw_entities = [
        item for item in graph.get("entities", []) if isinstance(item, dict)
    ]
    entity_map = {
        str(item.get("name") or ""): item
        for item in raw_entities
        if str(item.get("name") or "")
    }
    evaluator_entities = {
        name for name, item in entity_map.items() if _is_evaluator_graph_entity(item)
    }
    chunks: List[MemoryChunk] = []

    for index, entity in enumerate(raw_entities):
        name = str(entity.get("name") or "").strip()
        if not name or name == "USER" or name in evaluator_entities:
            continue
        entity_type = str(entity.get("entity_type") or entity.get("type") or "Other").strip()
        description = str(entity.get("description") or "").strip()
        content = (
            f"{name} ({entity_type}): {description}"
            if description
            else f"{name} is an entity of type {entity_type}."
        )
        source_ids = _graph_source_ids(entity)
        chunks.append(
            MemoryChunk(
                memory_id=f"case_graph:{case_id}:entity:{index}",
                content=content,
                metadata={
                    "source": "case_graph_baseline",
                    "case_id": case_id,
                    "graph_unit_type": "entity",
                    "source_ids": source_ids,
                    "facts": [description or content],
                    "summary": content,
                    "keywords": _unique_texts([name, entity_type]),
                },
            )
        )

    raw_relationships = [
        item for item in graph.get("relationships", []) if isinstance(item, dict)
    ]
    for index, edge in enumerate(raw_relationships):
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        relation = str(edge.get("description") or edge.get("relation") or "related_to").strip()
        if (
            not source
            or not target
            or relation == "target_answer"
            or _is_evaluator_graph_item(edge)
            or source in evaluator_entities
            or target in evaluator_entities
        ):
            continue
        relation_text = " ".join(
            relation.replace("<SEP>", " ; ").replace("_", " ").split()
        )
        content = f"{source} -- {relation_text} --> {target}."
        source_ids = _graph_source_ids(edge)
        if not source_ids:
            source_ids = _unique_texts(
                [
                    *_graph_source_ids(entity_map.get(source, {})),
                    *_graph_source_ids(entity_map.get(target, {})),
                ]
            )
        chunks.append(
            MemoryChunk(
                memory_id=f"case_graph:{case_id}:relationship:{index}",
                content=content,
                metadata={
                    "source": "case_graph_baseline",
                    "case_id": case_id,
                    "graph_unit_type": "relationship",
                    "source_ids": source_ids,
                    "facts": [content],
                    "summary": content,
                    "keywords": _unique_texts([source, relation_text, target]),
                    "weight": edge.get("weight"),
                },
            )
        )

    return MemoryStore(chunks)


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
    ranked_hit_dicts: List[Dict[str, Any]],
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


def _is_evaluator_graph_item(item: Dict[str, Any]) -> bool:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return metadata.get("source") == EVALUATOR_METADATA_SOURCE


def _is_evaluator_graph_entity(entity: Dict[str, Any]) -> bool:
    description = str(entity.get("description") or "")
    return _is_evaluator_graph_item(entity) or (
        "Memory fact recovered from evaluator metadata." in description
        or "Target answer for question:" in description
    )


def _graph_source_ids(item: Dict[str, Any]) -> List[str]:
    raw = item.get("source_ids", [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        return []
    return _unique_texts(raw)


def _unique_texts(values: Iterable[Any]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _longmemeval_retrieval_targets(
    entry: Dict[str, Any],
    case_id: str,
) -> tuple[List[str], bool, str]:
    annotated_sources = [
        str(item) for item in entry.get("answer_session_ids", []) if str(item)
    ]
    session_ids = entry.get("haystack_session_ids", [])
    sessions = entry.get("haystack_sessions", [])
    all_turns = [
        turn
        for session in sessions
        if isinstance(session, list)
        for turn in session
        if isinstance(turn, dict)
    ]
    has_turn_labels = any("has_answer" in turn for turn in all_turns)

    if has_turn_labels:
        source_ids = []
        for session_id, session in zip(session_ids, sessions):
            if "answer" not in str(session_id) or not isinstance(session, list):
                continue
            if any(
                bool(turn.get("has_answer"))
                for turn in session
                if isinstance(turn, dict)
            ):
                source_ids.append(str(session_id))
        has_user_target = any(
            str(turn.get("role") or "") == "user" and bool(turn.get("has_answer"))
            for turn in all_turns
        )
    else:
        source_ids = annotated_sources
        has_user_target = bool(source_ids)

    if "_abs" in case_id:
        return source_ids, False, "abstention_case"
    if has_turn_labels and not has_user_target:
        return source_ids, False, "no_answer_label_in_user_turn"
    if not source_ids:
        return source_ids, False, "no_retrieval_target_session"
    return source_ids, True, ""


def _longmemeval_judge_prompt(
    *,
    case_id: str,
    question_type: str,
    question: str,
    gold_answer: str,
    candidate_answer: str,
) -> str:
    if "_abs" in case_id:
        criterion = (
            "The question is unanswerable. Mark yes only when the model recognizes "
            "that the requested information is unavailable or incomplete. It may "
            "mention available information as long as it does not invent the answer."
        )
        answer_label = "Reference explanation"
    elif question_type == "temporal-reasoning":
        criterion = (
            "Mark yes when the response contains or is equivalent to the complete "
            "reference answer. A response containing only part of the required answer "
            "is incorrect. For requested day, week, or month durations, tolerate an "
            "off-by-one numerical error."
        )
        answer_label = "Correct answer"
    elif question_type == "knowledge-update":
        criterion = (
            "Mark yes when the response contains the required latest answer. Mentioning "
            "older information as additional context is allowed when the updated answer "
            "is also clearly present."
        )
        answer_label = "Correct answer"
    elif question_type == "single-session-preference":
        criterion = (
            "Treat the reference as a personalization rubric. Mark yes when the response "
            "correctly recalls and uses the user's personal information; it need not "
            "cover every rubric point."
        )
        answer_label = "Personalization rubric"
    else:
        criterion = (
            "Mark yes when the response contains or is equivalent to the complete "
            "reference answer, including valid intermediate reasoning. A response that "
            "contains only a subset of the required information is incorrect."
        )
        answer_label = "Correct answer"

    return (
        "Evaluate the model response under the LongMemEval protocol. "
        f"{criterion}\n\n"
        f"Question: {question}\n\n"
        f"{answer_label}: {gold_answer}\n\n"
        f"Model response: {candidate_answer}\n\n"
        "Is the model response correct? Answer yes or no only."
    )


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
