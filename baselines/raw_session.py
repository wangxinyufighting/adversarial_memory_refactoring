"""Raw-session memory baseline."""

from __future__ import annotations

from typing import Any, Dict, List

from case_graph.retriever import MemoryChunk, MemoryStore

from .base import BaselineMetadata


class RawSessionBaseline:
    """Use each CaseGraph raw session chunk as one memory item."""

    def __init__(self, name: str = "raw_session"):
        self.name = name

    def build_memory(self, graph: Dict[str, Any]) -> MemoryStore:
        chunks: List[MemoryChunk] = []
        for index, item in enumerate(graph.get("chunks", [])):
            chunk_id = str(item.get("chunk_id") or item.get("session_id") or f"{self.name}-{index}")
            chunks.append(
                MemoryChunk(
                    memory_id=chunk_id,
                    content=str(item.get("content") or item.get("text") or ""),
                    metadata={
                        "baseline": self.name,
                        "source": "case_graph_raw_session",
                        "source_ids": [chunk_id],
                        "timestamp": item.get("timestamp", ""),
                        "order": item.get("order", index),
                    },
                )
            )
        return MemoryStore(chunks)

    def metadata(self) -> Dict[str, Any]:
        return BaselineMetadata(
            name=self.name,
            kind="raw_session",
            extra={"description": "Each CaseGraph raw haystack session is one memory chunk."},
        ).to_dict()
