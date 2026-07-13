import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .baseline import AnswerEquivalenceJudge, JsonClient, _answers_match
from .llm import OpenAIChatClient
from .retriever import FrozenBM25Retriever, MemoryStore, RetrievalHit


class RetrievedMemoryAnswerAgent:
    """Answer Q using only old-memory chunks retrieved from Mt."""

    def __init__(self, client: Optional[JsonClient] = None, max_output_tokens: int = 200):
        self.client = client
        self.max_output_tokens = max_output_tokens

    def answer(self, question: str, retrieved_memories: List[RetrievalHit]) -> Dict[str, Any]:
        if not question.strip() or not retrieved_memories:
            return {
                "answer": "UNKNOWN",
                "reason": "Missing question or retrieved memories.",
                "evidence_memory_ids": [],
            }
        response = self._client().complete_json(
            system_prompt=(
                "You answer memory questions using only the retrieved old memories. "
                "Return JSON only with answer, reason, and evidence_memory_ids."
            ),
            user_prompt=json.dumps(
                {
                    "question": question,
                    "retrieved_memories": [hit.to_dict() for hit in retrieved_memories],
                    "instruction": (
                        "Answer concisely using only retrieved_memories. "
                        "If the answer is not explicitly supported, answer UNKNOWN. "
                        "List only memory_id values that directly support the answer in evidence_memory_ids."
                    ),
                },
                ensure_ascii=False,
            ),
            max_tokens=self.max_output_tokens,
        )
        evidence_memory_ids = response.get("evidence_memory_ids", [])
        if isinstance(evidence_memory_ids, str):
            evidence_memory_ids = [evidence_memory_ids]
        return {
            "answer": str(response.get("answer", "")),
            "reason": str(response.get("reason", "")),
            "evidence_memory_ids": [str(item) for item in evidence_memory_ids],
        }

    def _client(self) -> JsonClient:
        if self.client is None:
            self.client = OpenAIChatClient.from_env()
        return self.client


@dataclass
class SuccessPool:
    """Global pool of questions already answered from current memory."""

    successes: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "SuccessPool":
        pool_path = Path(path)
        if not pool_path.exists():
            return cls()
        payload = json.loads(pool_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return cls(successes=payload)
        return cls(successes=list(payload.get("successes", payload.get("questions", []))))

    def add(
        self,
        question: str,
        answer: str,
        memory_ids: Iterable[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        question = str(question or "").strip()
        if not question:
            return
        record = {
            "question": question,
            "answer": str(answer or ""),
            "memory_ids": list(memory_ids),
            "metadata": dict(metadata or {}),
        }
        for index, existing in enumerate(self.successes):
            if existing.get("question") == question:
                self.successes[index] = record
                return
        self.successes.append(record)

    def save(self, path: str | Path) -> None:
        pool_path = Path(path)
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        pool_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def to_dict(self) -> Dict[str, Any]:
        return {"successes": self.successes}


@dataclass(frozen=True)
class InitialDefenseOutcome:
    question: str
    correct: bool
    needs_refactor: bool
    retrieved_memories: List[RetrievalHit]
    answer_result: Dict[str, Any]
    judge: Dict[str, Any]
    bound_memory_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "correct": self.correct,
            "needs_refactor": self.needs_refactor,
            "retrieved_memories": [hit.to_dict() for hit in self.retrieved_memories],
            "answer_result": self.answer_result,
            "judge": self.judge,
            "bound_memory_ids": self.bound_memory_ids,
        }


def run_initial_defense(
    question: str,
    gold_answer: str,
    memory_store: MemoryStore,
    answer_agent: RetrievedMemoryAnswerAgent,
    judge: AnswerEquivalenceJudge,
    success_pool: Optional[SuccessPool] = None,
    retriever: Optional[FrozenBM25Retriever] = None,
    top_k: int = 5,
    min_score: float = 0.0,
    use_answer_agent: bool = True,
    success_validator: Optional[
        Callable[
            [str, str, List[RetrievalHit], Dict[str, Any], Dict[str, Any]],
            Dict[str, Any],
        ]
    ] = None,
) -> InitialDefenseOutcome:
    retriever = retriever or FrozenBM25Retriever(memory_store)
    hits = retriever.retrieve(question, top_k=top_k, min_score=min_score)
    if use_answer_agent:
        answer_result = answer_agent.answer(question, hits)
        judge_result = judge.judge(
            question=question,
            gold_answer=gold_answer,
            candidate_answer=str(answer_result.get("answer", "")),
        )
    else:
        matching_hit_ids = [
            hit.memory_id
            for hit in hits
            if _answers_match(gold_answer, _hit_evidence_text(hit))
        ]
        correct_by_presence = bool(matching_hit_ids)
        answer_result = {
            "answer": gold_answer if correct_by_presence else "UNKNOWN",
            "reason": "Initial defense used deterministic retrieved-memory answer presence.",
            "evidence_memory_ids": matching_hit_ids,
        }
        judge_result = {
            "correct": correct_by_presence,
            "method": "retrieved_answer_presence",
            "reason": "Gold answer was checked against retrieved memory text without an LLM call.",
        }
    correct = bool(judge_result.get("correct", False))
    if correct and success_validator is not None:
        validation = success_validator(
            question,
            gold_answer,
            hits,
            answer_result,
            judge_result,
        )
        validation = dict(validation or {})
        memory_complete = bool(validation.get("complete", False))
        judge_result = {
            **judge_result,
            "memory_complete": memory_complete,
            "memory_completeness": validation,
        }
        correct = correct and memory_complete
    bound_memory_ids: List[str] = []

    if correct:
        hit_ids = {hit.memory_id for hit in hits}
        evidence_ids = [
            memory_id
            for memory_id in answer_result.get("evidence_memory_ids", [])
            if memory_id in hit_ids
        ]
        bound_memory_ids = evidence_ids or ([hits[0].memory_id] if hits else [])
        memory_store.bind_question(bound_memory_ids, question)
        if success_pool is not None:
            success_pool.add(
                question=question,
                answer=gold_answer,
                memory_ids=bound_memory_ids,
                metadata={"stage": "initial_defense"},
            )

    return InitialDefenseOutcome(
        question=question,
        correct=correct,
        needs_refactor=not correct,
        retrieved_memories=hits,
        answer_result=answer_result,
        judge=judge_result,
        bound_memory_ids=bound_memory_ids,
    )


def _hit_evidence_text(hit: RetrievalHit) -> str:
    metadata = hit.metadata or {}
    parts = [hit.content]
    for key in ("facts", "fact", "summary", "keywords"):
        value = metadata.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return "\n".join(part for part in parts if part)
