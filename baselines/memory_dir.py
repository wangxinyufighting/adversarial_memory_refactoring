"""Baseline wrapper for an existing per-case memory directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from case_graph.evaluation import resolve_case_memory_path
from case_graph.retriever import MemoryStore

from .base import BaselineMetadata, MissingMemoryError


class MemoryDirectoryBaseline:
    """Load memories produced by a training run from checkpoint memory_states."""

    def __init__(self, memory_dir: str | Path, name: str = "candidate_memory"):
        self.memory_dir = Path(memory_dir)
        self.name = name

    def build_memory(self, graph: Dict[str, Any]) -> MemoryStore:
        case_id = str(graph.get("case_id", "unknown"))
        memory_path = resolve_case_memory_path(case_id, self.memory_dir)
        if memory_path is None:
            raise MissingMemoryError(f"No memory file found for case {case_id} in {self.memory_dir}")
        return MemoryStore.load(memory_path)

    def metadata(self) -> Dict[str, Any]:
        return BaselineMetadata(
            name=self.name,
            kind="memory_dir",
            extra={"memory_dir": str(self.memory_dir)},
        ).to_dict()
