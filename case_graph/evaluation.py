"""Evaluate compressed memories on CaseGraph target questions."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .baseline import AnswerEquivalenceJudge
from .defense import RetrievedMemoryAnswerAgent
from .retriever import MemoryStore, build_memory_retriever


@dataclass(frozen=True)
class TargetEvaluationResult:
    case_id: str
    question: str
    gold_answer: str
    candidate_answer: str
    correct: bool
    retrieved_memories: List[Dict[str, Any]]
    answer_result: Dict[str, Any]
    judge: Dict[str, Any]
    memory_path: str = ""
    status: str = "evaluated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "candidate_answer": self.candidate_answer,
            "correct": self.correct,
            "retrieved_memories": self.retrieved_memories,
            "answer_result": self.answer_result,
            "judge": self.judge,
            "memory_path": self.memory_path,
            "status": self.status,
        }


def evaluate_case_target(
    graph: Dict[str, Any],
    memory_store: MemoryStore,
    answer_agent: RetrievedMemoryAnswerAgent,
    judge: AnswerEquivalenceJudge,
    top_k: int = 5,
    min_score: float = 0.0,
    memory_path: str = "",
    retriever_config: Optional[Dict[str, Any]] = None,
) -> TargetEvaluationResult:
    """Retrieve compressed memory and answer a graph's held-out target question."""
    case_id = str(graph.get("case_id", "unknown"))
    target = graph.get("target") or {}
    question = str(target.get("question") or graph.get("question") or "")
    gold_answer = str(target.get("answer") or graph.get("answer") or "")
    if not question or not gold_answer:
        raise ValueError(f"Case {case_id} is missing target question or answer.")

    hits = build_memory_retriever(retriever_config, memory_store).retrieve(
        question,
        top_k=top_k,
        min_score=min_score,
    )
    answer_result = answer_agent.answer(question, hits)
    candidate_answer = str(answer_result.get("answer", ""))
    judge_result = judge.judge(
        question=question,
        gold_answer=gold_answer,
        candidate_answer=candidate_answer,
    )
    return TargetEvaluationResult(
        case_id=case_id,
        question=question,
        gold_answer=gold_answer,
        candidate_answer=candidate_answer,
        correct=bool(judge_result.get("correct", False)),
        retrieved_memories=[hit.to_dict() for hit in hits],
        answer_result=answer_result,
        judge=judge_result,
        memory_path=memory_path,
    )


def graph_paths(path: str | Path) -> List[Path]:
    graph_path = Path(path)
    if graph_path.is_dir():
        return sorted(graph_path.glob("*.case_graph.json"))
    return [graph_path]


def load_graph(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_case_memory_path(case_id: str, memory_dir: str | Path) -> Optional[Path]:
    """Find the newest memory JSON for a case in an online checkpoint directory."""
    directory = Path(memory_dir)
    if not directory.exists():
        return None

    exact = directory / f"{case_id}.json"
    if exact.exists():
        return exact

    candidates = [
        path
        for path in directory.glob(f"{case_id}*.json")
        if not path.name.endswith("_pool.json") and not path.name.endswith("_buffer.json")
    ]
    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda path: (_episode_number(path.name), path.stat().st_mtime),
        reverse=True,
    )[0]


def summarize_results(results: List[TargetEvaluationResult]) -> Dict[str, Any]:
    evaluated = [item for item in results if item.status == "evaluated"]
    correct = sum(1 for item in evaluated if item.correct)
    total = len(evaluated)
    return {
        "total": len(results),
        "evaluated": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
    }


def _episode_number(filename: str) -> int:
    match = re.search(r"_ep(\d+)", filename)
    return int(match.group(1)) if match else -1
