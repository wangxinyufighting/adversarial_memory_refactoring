"""Oracle answer-source upper-bound baseline."""

from __future__ import annotations

from typing import Any, Dict, List, Set

from case_graph.retriever import MemoryChunk, MemoryStore

from .base import BaselineMetadata


class OracleSourceBaseline:
    """Use only answer-source sessions as memory.

    This is an upper-bound diagnostic, not a fair deployable baseline, because
    it reads target answer_source_ids.
    """

    def __init__(self, name: str = "oracle_source"):
        self.name = name

    def build_memory(self, graph: Dict[str, Any]) -> MemoryStore:
        source_ids = set(_answer_source_ids(graph))
        chunks: List[MemoryChunk] = []
        for index, item in enumerate(graph.get("chunks", [])):
            chunk_id = str(item.get("chunk_id") or item.get("session_id") or f"{self.name}-{index}")
            if chunk_id not in source_ids:
                continue
            chunks.append(
                MemoryChunk(
                    memory_id=chunk_id,
                    content=str(item.get("content") or item.get("text") or ""),
                    metadata={
                        "baseline": self.name,
                        "source": "oracle_answer_source",
                        "source_ids": [chunk_id],
                        "timestamp": item.get("timestamp", ""),
                    },
                )
            )
        return MemoryStore(chunks)

    def metadata(self) -> Dict[str, Any]:
        return BaselineMetadata(
            name=self.name,
            kind="oracle_source",
            extra={
                "description": (
                    "Upper-bound diagnostic using target answer_source_ids; "
                    "do not treat as a fair memory method."
                )
            },
        ).to_dict()


def _answer_source_ids(graph: Dict[str, Any]) -> Set[str]:
    target = graph.get("target") or {}
    values = target.get("answer_source_ids") or graph.get("answer_source_ids") or []
    if isinstance(values, str):
        values = [values]
    return {str(item) for item in values}
