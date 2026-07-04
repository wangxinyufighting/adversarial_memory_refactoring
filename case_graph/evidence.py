from typing import Any, Dict, Iterable, List

from .routing import GraphRoute


def route_source_ids(route: GraphRoute) -> List[str]:
    seen = set()
    source_ids = []
    for edge in route.relationships:
        for source_id in edge.get("source_ids", []):
            if source_id not in seen:
                source_ids.append(source_id)
                seen.add(source_id)
    return source_ids


def resolve_golden_facts(graph: Dict[str, Any], source_ids: Iterable[str]) -> List[Dict[str, Any]]:
    wanted = set(source_ids)
    if not wanted:
        return []
    facts = []
    for chunk in sorted(graph.get("chunks", []), key=lambda item: item.get("order", 0)):
        if chunk.get("chunk_id") in wanted:
            facts.append(
                {
                    "session_id": chunk.get("chunk_id", ""),
                    "timestamp": chunk.get("timestamp", ""),
                    "text": chunk.get("content", ""),
                }
            )
    return facts


def route_golden_facts(graph: Dict[str, Any], route: GraphRoute) -> List[Dict[str, Any]]:
    return resolve_golden_facts(graph, route_source_ids(route))
