"""Post-construction memory certification for evaluation."""

from typing import Any, Dict, List, Optional

from .retriever import MemoryStore, build_memory_retriever


def certify_memory_coverage(
    memory_store: MemoryStore,
    target_question: str,
    answer_source_ids: List[str],
    retriever_config: Dict[str, Any],
    top_k: int = 10,
) -> Dict[str, Any]:
    """Check if constructed memory can answer target by verifying source coverage."""

    if not answer_source_ids:
        return {
            "certified": False,
            "reason": "no_answer_sources",
            "source_coverage": 0.0,
            "retrieved_sources": [],
            "missing_sources": [],
        }

    retriever = build_memory_retriever(retriever_config, memory_store)
    hits = retriever.retrieve(target_question, top_k=top_k, min_score=-2.0)

    retrieved_source_ids = set()
    for hit in hits:
        metadata = hit.chunk.metadata or {}
        source_ids = metadata.get("source_ids", [])
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        retrieved_source_ids.update(str(sid) for sid in source_ids if sid)

    answer_sources = set(str(sid) for sid in answer_source_ids)
    covered_sources = retrieved_source_ids & answer_sources
    missing_sources = answer_sources - retrieved_source_ids

    source_coverage = len(covered_sources) / len(answer_sources) if answer_sources else 0.0
    certified = source_coverage >= 0.9

    return {
        "certified": certified,
        "reason": "coverage_sufficient" if certified else "missing_answer_sources",
        "source_coverage": source_coverage,
        "retrieved_sources": sorted(covered_sources),
        "missing_sources": sorted(missing_sources),
        "retrieved_chunk_count": len(hits),
        "answer_source_count": len(answer_sources),
    }


def batch_certify_memories(
    memory_dir: str,
    graphs: Dict[str, Dict[str, Any]],
    retriever_config: Dict[str, Any],
    top_k: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Certify all memories in directory against their target questions."""

    from pathlib import Path

    results = {}
    memory_path = Path(memory_dir)

    for case_id, graph in graphs.items():
        case_memory_path = memory_path / f"{case_id}.json"
        if not case_memory_path.exists():
            results[case_id] = {
                "certified": False,
                "reason": "memory_file_missing",
                "source_coverage": 0.0,
            }
            continue

        target = graph.get("target", {})
        if isinstance(target, dict):
            question = target.get("question", "")
            answer_sources = target.get("answer_source_ids", [])
        else:
            question = ""
            answer_sources = []

        if not question:
            results[case_id] = {
                "certified": False,
                "reason": "no_target_question",
                "source_coverage": 0.0,
            }
            continue

        try:
            memory_store = MemoryStore.load(case_memory_path)
            result = certify_memory_coverage(
                memory_store,
                question,
                answer_sources,
                retriever_config,
                top_k=top_k,
            )
            results[case_id] = result
        except Exception as exc:
            results[case_id] = {
                "certified": False,
                "reason": f"certification_error: {exc}",
                "source_coverage": 0.0,
            }

    return results
