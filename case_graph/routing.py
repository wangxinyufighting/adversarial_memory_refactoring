import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .models import EVALUATOR_METADATA_SOURCE


@dataclass
class GraphRoute:
    policy: str
    nodes: List[str]
    relationships: List[Dict[str, Any]]
    score: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "nodes": self.nodes,
            "relationships": self.relationships,
            "score": self.score,
            "features": self.features,
            "reason": self.reason,
        }

    def to_evidence_dict(self) -> Dict[str, Any]:
        """Public route view passed to the attacker: graph evidence only, no target metadata."""
        return {
            "policy": self.policy,
            "nodes": self.nodes,
            "relationships": [_public_relationship(edge) for edge in self.relationships],
        }


def _public_relationship(edge: Dict[str, Any]) -> Dict[str, Any]:
    description = edge.get("description", "")
    if description == "target_answer":
        description = "related_to"
    return {
        "source": edge["source"],
        "target": edge["target"],
        "description": description,
        "weight": edge.get("weight", 1.0),
    }


class RerankClient(Protocol):
    def complete_json(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        ...


def _entity_map(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {entity["name"]: entity for entity in graph.get("entities", [])}


def _is_evaluator_injected_entity(entity: Dict[str, Any]) -> bool:
    metadata = entity.get("metadata", {})
    description = entity.get("description", "")
    return (
        metadata.get("source") == EVALUATOR_METADATA_SOURCE
        or "Memory fact recovered from evaluator metadata." in description
        or "Target answer for question:" in description
    )


def _is_evaluator_injected_edge(edge: Dict[str, Any], entities: Dict[str, Dict[str, Any]]) -> bool:
    metadata = edge.get("metadata", {})
    return (
        metadata.get("source") == EVALUATOR_METADATA_SOURCE
        or edge.get("description") == "target_answer"
        or _is_evaluator_injected_entity(entities.get(edge["source"], {}))
        or _is_evaluator_injected_entity(entities.get(edge["target"], {}))
    )


def _route_relationships(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    entities = _entity_map(graph)
    return [
        edge
        for edge in graph.get("relationships", [])
        if not _is_evaluator_injected_edge(edge, entities)
    ]


def _out_edges(relationships: List[Dict[str, Any]], node: str) -> List[Dict[str, Any]]:
    return [edge for edge in relationships if edge["source"] == node]


def _incident_edges(relationships: List[Dict[str, Any]], node: str) -> List[Dict[str, Any]]:
    return [
        edge
        for edge in relationships
        if edge["source"] == node or edge["target"] == node
    ]


def _edge_sort_key(edge: Dict[str, Any]) -> tuple:
    return (edge["source"], edge["target"], edge.get("description", ""))


def _route_nodes(edges: List[Dict[str, Any]]) -> List[str]:
    if not edges:
        return []
    nodes = [edges[0]["source"]]
    for edge in edges:
        if nodes[-1] != edge["source"]:
            nodes.append(edge["source"])
        nodes.append(edge["target"])
    return nodes


def _start_node(relationships: List[Dict[str, Any]]) -> str:
    if any(edge["source"] == "USER" or edge["target"] == "USER" for edge in relationships):
        return "USER"
    if relationships:
        first = sorted(relationships, key=_edge_sort_key)[0]
        return first["source"]
    return "USER"


def _score_route(graph: Dict[str, Any], route: GraphRoute) -> GraphRoute:
    del graph
    weights = [float(edge.get("weight", 0.0)) for edge in route.relationships]
    source_ids = {
        source_id
        for edge in route.relationships
        for source_id in edge.get("source_ids", [])
    }
    features = {
        "includes_user": 1.0 if "USER" in route.nodes else 0.0,
        "path_length": float(len(route.relationships)),
        "avg_edge_weight": sum(weights) / len(weights) if weights else 0.0,
        "source_diversity": float(len(source_ids)),
    }
    route.features = features
    route.score = (
        2.0 * features["includes_user"]
        + 0.05 * features["avg_edge_weight"]
        + 0.2 * features["source_diversity"]
        - 0.1 * features["path_length"]
    )
    return route


class RandomWalkRoutingPolicy:
    name = "random_walk"

    def __init__(self, seed: int = 0, max_steps: int = 3):
        self.seed = seed
        self.max_steps = max_steps

    def select_route(self, graph: Dict[str, Any]) -> GraphRoute:
        rng = random.Random(self.seed)
        relationships = _route_relationships(graph)
        current = _start_node(relationships)
        edges = []
        for _ in range(self.max_steps):
            candidates = sorted(_out_edges(relationships, current) or _incident_edges(relationships, current), key=_edge_sort_key)
            if not candidates:
                break
            edge = rng.choice(candidates)
            edges.append(edge)
            current = edge["target"] if edge["source"] == current else edge["source"]
        route = GraphRoute(policy=self.name, nodes=_route_nodes(edges) or [current], relationships=edges)
        return _score_route(graph, route)


class HeuristicRoutingPolicy:
    name = "heuristic"

    def select_route(self, graph: Dict[str, Any]) -> GraphRoute:
        relationships = _route_relationships(graph)
        user_edges = [
            edge
            for edge in relationships
            if edge["source"] == "USER" or edge["target"] == "USER"
        ]
        candidates = user_edges or relationships
        edge = sorted(candidates, key=lambda e: (-float(e.get("weight", 0.0)), _edge_sort_key(e)))[0]
        route = GraphRoute(
            policy=self.name,
            nodes=[edge["source"], edge["target"]],
            relationships=[edge],
            reason="Highest-weight user-centered evidence edge.",
        )
        return _score_route(graph, route)


class FeatureScoredLLMRerankRoutingPolicy:
    name = "feature_scored_llm_rerank"

    def __init__(self, client: Optional[RerankClient] = None, top_k: int = 5, max_output_tokens: int = 300):
        self.client = client
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens

    def select_route(self, graph: Dict[str, Any]) -> GraphRoute:
        candidates = self._candidate_routes(graph)
        candidates = sorted(candidates, key=lambda route: route.score, reverse=True)[: self.top_k]
        if self.client is None:
            candidates[0].reason = "Selected by feature score."
            return candidates[0]

        response = self.client.complete_json(
            system_prompt="You rerank graph routes for adversarial question generation. Return JSON only.",
            user_prompt=json.dumps(
                {
                    "routes": [route.to_evidence_dict() for route in candidates],
                    "instruction": (
                        "Select the route that can support a specific, tricky, answerable memory question. "
                        "Use only the route evidence. Return selected_index and reason."
                    ),
                },
                ensure_ascii=False,
            ),
            max_tokens=self.max_output_tokens,
        )
        index = int(response.get("selected_index", 0))
        selected = candidates[index]
        selected.reason = response.get("reason", "")
        return selected

    def _candidate_routes(self, graph: Dict[str, Any]) -> List[GraphRoute]:
        routes = []

        for edge in _route_relationships(graph):
            route = GraphRoute(
                policy=self.name,
                nodes=[edge["source"], edge["target"]],
                relationships=[edge],
            )
            routes.append(_score_route(graph, route))
        return routes
