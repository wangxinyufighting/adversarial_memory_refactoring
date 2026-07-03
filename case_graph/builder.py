from dataclasses import dataclass
from hashlib import sha1
from typing import Callable, Dict, Iterable, Optional, Protocol

from .models import CaseGraph, ExtractionResult, SessionChunk


class Extractor(Protocol):
    def extract(self, chunk: SessionChunk) -> ExtractionResult:
        ...


@dataclass
class CaseGraphBuilder:
    extractor: Extractor
    extraction_cache: Optional[Dict[str, ExtractionResult]] = None
    progress_callback: Optional[Callable[[int, int, SessionChunk], None]] = None
    cache_namespace: str = "case_graph_extraction_unifiedmem_v1"

    def build(self, case_id: str, chunks: Iterable[SessionChunk]) -> CaseGraph:
        chunk_list = list(chunks)
        graph = CaseGraph(case_id=case_id)
        total = len(chunk_list)
        for index, chunk in enumerate(chunk_list, start=1):
            graph.add_chunk(chunk)
            if not chunk.content.strip():
                self._report_progress(index, total, chunk)
                continue
            cache_key = self._cache_key(chunk)
            if self.extraction_cache is not None and cache_key in self.extraction_cache:
                extraction = self.extraction_cache[cache_key]
            else:
                extraction = self.extractor.extract(chunk)
                if self.extraction_cache is not None:
                    self.extraction_cache[cache_key] = extraction
            graph.apply_extraction(chunk.chunk_id, extraction)
            self._report_progress(index, total, chunk)
        return graph

    def _cache_key(self, chunk: SessionChunk) -> str:
        cache_basis = f"{chunk.timestamp}\0{chunk.content}"
        content_hash = sha1(cache_basis.encode("utf-8")).hexdigest()
        return f"{self.cache_namespace}:{content_hash}"

    def _report_progress(self, current: int, total: int, chunk: SessionChunk) -> None:
        if self.progress_callback is not None:
            self.progress_callback(current, total, chunk)
