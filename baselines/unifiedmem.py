"""UnifiedMem memory baseline adapter.

The adapter deliberately exports UnifiedMem-style memory into this repository's
MemoryStore so evaluation can reuse the same FrozenBM25Retriever and answer
backbone for every method.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from case_graph.retriever import MemoryChunk, MemoryStore

from .base import BaselineMetadata


DEFAULT_UNIFIEDMEM_REPO = "/Users/ganning/Documents/project_python/大模型记忆/UnifiedMem"


class UnifiedMemBaseline:
    """Build a UnifiedMem-style session memory from CaseGraph chunks.

    If expansion caches are provided, the baseline uses UnifiedMem's flat-memory
    convention of indexing user facts, keyphrases, and summaries. If no cache is
    provided for a session, it falls back to the raw session text.
    """

    def __init__(
        self,
        repo_path: str | Path = DEFAULT_UNIFIEDMEM_REPO,
        expansion_cache_paths: Optional[Iterable[str | Path]] = None,
        join_mode: str = "merge",
        name: str = "unifiedmem",
    ):
        self.name = name
        self.repo_path = Path(repo_path)
        self.expansion_cache_paths = [Path(path) for path in (expansion_cache_paths or [])]
        self.join_mode = join_mode
        self.expansion_caches = [_load_cache(path) for path in self.expansion_cache_paths]

    def build_memory(self, graph: Dict[str, Any]) -> MemoryStore:
        memories: List[MemoryChunk] = []
        for index, item in enumerate(graph.get("chunks", [])):
            chunk_id = str(item.get("chunk_id") or item.get("session_id") or f"session-{index}")
            raw_text = str(item.get("content") or item.get("text") or "")
            expansions = self._expansions_for(chunk_id)
            memories.extend(self._chunk_to_memories(chunk_id, raw_text, expansions, item, index))
        return MemoryStore(memories)

    def metadata(self) -> Dict[str, Any]:
        return BaselineMetadata(
            name=self.name,
            kind="unifiedmem",
            extra={
                "repo_path": str(self.repo_path),
                "join_mode": self.join_mode,
                "expansion_cache_paths": [str(path) for path in self.expansion_cache_paths],
                "retrieval_policy": "shared_frozen_bm25_in_Evaluation",
            },
        ).to_dict()

    def _chunk_to_memories(
        self,
        chunk_id: str,
        raw_text: str,
        expansions: Mapping[str, str],
        item: Dict[str, Any],
        index: int,
    ) -> List[MemoryChunk]:
        base_metadata = {
            "baseline": self.name,
            "source": "unifiedmem_adapter",
            "source_repo": str(self.repo_path),
            "source_ids": [chunk_id],
            "timestamp": item.get("timestamp", ""),
            "order": item.get("order", index),
            "join_mode": self.join_mode,
        }

        nonempty_expansions = {
            key: value.strip()
            for key, value in expansions.items()
            if str(value or "").strip()
        }
        if self.join_mode == "separate":
            if nonempty_expansions:
                memories = []
                for key, value in nonempty_expansions.items():
                    metadata = dict(base_metadata)
                    metadata["expansion_type"] = key
                    memories.append(
                        MemoryChunk(
                            memory_id=f"{chunk_id}:{key}",
                            content=value,
                            metadata=metadata,
                        )
                    )
                return memories
            metadata = dict(base_metadata)
            metadata["expansion_type"] = "raw_fallback"
            return [MemoryChunk(memory_id=chunk_id, content=raw_text, metadata=metadata)]

        if self.join_mode == "merge_raw":
            content = "\n".join([raw_text, *_format_expansions(nonempty_expansions)]).strip()
        elif self.join_mode == "merge":
            content = "\n".join(_format_expansions(nonempty_expansions)).strip() or raw_text
        elif self.join_mode == "none":
            content = raw_text
        else:
            raise ValueError(f"Unsupported UnifiedMem join_mode: {self.join_mode}")

        metadata = dict(base_metadata)
        metadata["expansion_types"] = sorted(nonempty_expansions)
        return [
            MemoryChunk(
                memory_id=chunk_id,
                content=content,
                metadata=metadata,
            )
        ]

    def _expansions_for(self, session_id: str) -> Dict[str, str]:
        expansions: Dict[str, str] = {}
        for path, cache in zip(self.expansion_cache_paths, self.expansion_caches):
            label = _cache_label(path)
            value = _lookup_cache(cache, session_id)
            if value is not None:
                expansions[label] = _stringify_expansion(value)
        return expansions


def _load_cache(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"UnifiedMem expansion cache not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _lookup_cache(cache: Any, session_id: str) -> Any:
    if isinstance(cache, dict):
        if session_id in cache:
            return cache[session_id]
        for key in ("sessions", "data", "results", "items"):
            value = cache.get(key)
            found = _lookup_cache(value, session_id)
            if found is not None:
                return found
    if isinstance(cache, list):
        for item in cache:
            if not isinstance(item, dict):
                continue
            item_id = item.get("session_id") or item.get("chunk_id") or item.get("id")
            if str(item_id) == session_id:
                return item
    return None


def _stringify_expansion(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "; ".join(_stringify_expansion(item) for item in value if _stringify_expansion(item))
    if isinstance(value, dict):
        preferred_keys = (
            "summary",
            "summ",
            "facts",
            "userfacts",
            "user_facts",
            "keywords",
            "keyphrases",
            "keyphrase",
            "content",
            "text",
            "output",
            "response",
        )
        parts = []
        for key in preferred_keys:
            if key in value:
                rendered = _stringify_expansion(value[key])
                if rendered:
                    parts.append(rendered)
        if parts:
            return "; ".join(parts)
        return "; ".join(
            _stringify_expansion(item)
            for item in value.values()
            if _stringify_expansion(item)
        )
    return str(value)


def _format_expansions(expansions: Mapping[str, str]) -> List[str]:
    return [f"{key}: {value}" for key, value in sorted(expansions.items()) if value]


def _cache_label(path: Path) -> str:
    stem = path.stem.casefold()
    if "userfact" in stem or "user_fact" in stem:
        return "session-userfact"
    if "keyphrase" in stem or "keyword" in stem:
        return "session-keyphrase"
    if "summ" in stem or "summary" in stem:
        return "session-summ"
    return stem
