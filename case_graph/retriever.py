import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


@dataclass
class MemoryChunk:
    """Structured memory unit used by the defense stage."""

    memory_id: str
    content: str
    linked_questions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], fallback_id: str = "") -> "MemoryChunk":
        memory_id = str(
            payload.get("memory_id")
            or payload.get("id")
            or payload.get("chunk_id")
            or payload.get("session_id")
            or fallback_id
        )
        linked_questions = payload.get("linked_questions", [])
        if isinstance(linked_questions, str):
            linked_questions = [linked_questions]
        return cls(
            memory_id=memory_id,
            content=str(payload.get("content") or payload.get("text") or payload.get("summary") or ""),
            linked_questions=[str(item) for item in linked_questions],
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "linked_questions": self.linked_questions,
            "metadata": self.metadata,
        }

    def bind_question(self, question: str) -> None:
        question = str(question or "").strip()
        if question and question not in self.linked_questions:
            self.linked_questions.append(question)


@dataclass(frozen=True)
class RetrievalHit:
    memory_id: str
    content: str
    score: float
    rank: int
    linked_questions: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "score": self.score,
            "rank": self.rank,
            "linked_questions": self.linked_questions,
            "metadata": self.metadata,
        }


class MemoryStore:
    """In-memory representation of Mt, the current structured memory library."""

    def __init__(self, chunks: Iterable[MemoryChunk] = ()):
        self.chunks = list(chunks)

    @classmethod
    def from_dicts(cls, items: Iterable[Dict[str, Any]]) -> "MemoryStore":
        return cls(
            MemoryChunk.from_dict(item, fallback_id=f"memory-{index}")
            for index, item in enumerate(items)
        )

    @classmethod
    def load(cls, path: str | Path) -> "MemoryStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = payload.get("memories") or payload.get("chunks") or payload.get("memory_chunks") or []
        else:
            raise ValueError("Memory store must be a JSON object or list.")
        if not isinstance(items, list):
            raise ValueError("Memory store items must be a list.")
        return cls.from_dicts(items)

    def save(self, path: str | Path) -> None:
        payload = {"memories": [chunk.to_dict() for chunk in self.chunks]}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def bind_question(self, memory_ids: Iterable[str], question: str) -> None:
        wanted = set(memory_ids)
        for chunk in self.chunks:
            if chunk.memory_id in wanted:
                chunk.bind_question(question)


class FrozenBM25Retriever:
    """
    Frozen sparse retriever for stage-three old-memory lookup.

    The index is built from Mt contents only; no parameters are learned or updated
    during retrieval. Scores are BM25-style and deterministic.
    """

    def __init__(
        self,
        memory_store: MemoryStore | Sequence[MemoryChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.memory_store = (
            memory_store if isinstance(memory_store, MemoryStore) else MemoryStore(memory_store)
        )
        self.k1 = k1
        self.b = b
        self._tokenized = [_tokenize(chunk.content) for chunk in self.memory_store.chunks]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokenized]
        self._doc_lengths = [len(tokens) for tokens in self._tokenized]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        self._idf = self._build_idf()

    @classmethod
    def from_path(cls, path: str | Path, k1: float = 1.5, b: float = 0.75) -> "FrozenBM25Retriever":
        return cls(MemoryStore.load(path), k1=k1, b=b)

    def retrieve(self, question: str, top_k: int = 5, min_score: float = 0.0) -> List[RetrievalHit]:
        if top_k <= 0 or not self.memory_store.chunks:
            return []
        query_terms = _tokenize(question)
        if not query_terms:
            return []

        scored = []
        for index, chunk in enumerate(self.memory_store.chunks):
            score = self._score(query_terms, index)
            if score >= min_score:
                scored.append((score, index, chunk))

        scored.sort(key=lambda item: (-item[0], item[1]))
        hits = []
        for rank, (score, _, chunk) in enumerate(scored[:top_k], start=1):
            hits.append(
                RetrievalHit(
                    memory_id=chunk.memory_id,
                    content=chunk.content,
                    score=score,
                    rank=rank,
                    linked_questions=list(chunk.linked_questions),
                    metadata=dict(chunk.metadata),
                )
            )
        return hits

    def _build_idf(self) -> Dict[str, float]:
        document_count = len(self._tokenized)
        document_frequencies: Counter[str] = Counter()
        for tokens in self._tokenized:
            document_frequencies.update(set(tokens))
        return {
            term: math.log(1.0 + (document_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequencies.items()
        }

    def _score(self, query_terms: Sequence[str], doc_index: int) -> float:
        if not self._avg_doc_length:
            return 0.0
        score = 0.0
        term_frequencies = self._term_frequencies[doc_index]
        doc_length = self._doc_lengths[doc_index]
        for term in query_terms:
            frequency = term_frequencies.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * doc_length / self._avg_doc_length
            )
            score += self._idf.get(term, 0.0) * (frequency * (self.k1 + 1.0)) / denominator
        return score


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text or "").casefold())
