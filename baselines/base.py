"""Shared interfaces for memory baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

from case_graph.retriever import MemoryStore


class MissingMemoryError(FileNotFoundError):
    """Raised when a baseline has no memory for the requested case."""


class MemoryBaseline(Protocol):
    """A memory producer evaluated with the shared retriever and answer agent."""

    name: str

    def build_memory(self, graph: Dict[str, Any]) -> MemoryStore:
        ...

    def metadata(self) -> Dict[str, Any]:
        ...


@dataclass
class BaselineMetadata:
    name: str
    kind: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {"name": self.name, "kind": self.kind}
        payload.update(self.extra)
        return payload
