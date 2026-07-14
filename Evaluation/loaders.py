"""Data loaders for LongMemEval and CaseGraph evaluation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from case_graph.models import EVALUATOR_METADATA_SOURCE
from case_graph.retriever import MemoryChunk, MemoryStore


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


def case_graph_to_memory_store(graph: Dict[str, Any]) -> MemoryStore:
    """Convert CaseGraph to target-safe memory (entities + relationships, no raw sessions)."""

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


def _raw_session_chars(sessions: Any) -> int:
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


def _unique_texts(values: Any) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result
