"""Coverage tracking and coverage-aware scheduling for memory construction."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .models import EVALUATOR_METADATA_SOURCE
from .routing import GraphRoute, RandomWalkRoutingPolicy


def relationship_unit_id(edge: Dict[str, Any]) -> str:
    parts = [
        str(edge.get("source", "")).strip().casefold(),
        str(edge.get("description") or edge.get("relation") or "").strip().casefold(),
        str(edge.get("target", "")).strip().casefold(),
    ]
    return "edge-" + hashlib.sha1("\0".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CoverageUnit:
    unit_id: str
    source: str
    relation: str
    target: str
    source_ids: List[str] = field(default_factory=list)
    qualifiers: List[str] = field(default_factory=list)
    critical: bool = True
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "kind": "relationship",
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "source_ids": self.source_ids,
            "qualifiers": self.qualifiers,
            "critical": self.critical,
            "weight": self.weight,
        }


@dataclass
class CoverageStatus:
    attempts: int = 0
    failures: int = 0
    passes: int = 0
    certified_passes: int = 0
    covered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempts": self.attempts,
            "failures": self.failures,
            "passes": self.passes,
            "certified_passes": self.certified_passes,
            "covered": self.covered,
        }


class CaseCoverageTracker:
    def __init__(self, case_id: str, units: Iterable[CoverageUnit]):
        self.case_id = str(case_id)
        self.units = {unit.unit_id: unit for unit in units}
        self.statuses = {unit_id: CoverageStatus() for unit_id in self.units}
        self.question_count = 0
        self.certification_attempts = 0
        self.certification_failures = 0
        self.consecutive_certification_passes = 0

    @classmethod
    def from_graph(cls, graph: Dict[str, Any]) -> "CaseCoverageTracker":
        entities = {
            str(item.get("name", "")): item
            for item in graph.get("entities", [])
            if isinstance(item, dict)
        }
        units = []
        for edge in graph.get("relationships", []):
            if not isinstance(edge, dict) or not _public_edge(edge, entities):
                continue
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            relation = str(edge.get("description") or edge.get("relation") or "")
            if not source or not target:
                continue
            qualifiers = []
            for name in (source, target):
                description = str(entities.get(name, {}).get("description", "")).strip()
                if description:
                    qualifiers.append(description)
            units.append(
                CoverageUnit(
                    unit_id=relationship_unit_id(edge),
                    source=source,
                    relation=relation,
                    target=target,
                    source_ids=[str(item) for item in edge.get("source_ids", [])],
                    qualifiers=qualifiers,
                    weight=max(
                        1.0,
                        max(0.0, float(edge.get("weight", 1.0) or 1.0)) ** 0.5,
                    ),
                )
            )
        return cls(str(graph.get("case_id", "unknown")), units)

    def unit_ids_for_route(self, route: Any) -> List[str]:
        if isinstance(route, GraphRoute):
            relationships = route.relationships
        elif isinstance(route, dict):
            relationships = route.get("relationships", [])
        else:
            relationships = []
        result = []
        for edge in relationships:
            unit_id = relationship_unit_id(edge)
            if unit_id in self.units and unit_id not in result:
                result.append(unit_id)
        return result

    def record(self, unit_ids: Iterable[str], success: bool, certification: bool = False) -> None:
        known_ids = [unit_id for unit_id in unit_ids if unit_id in self.statuses]
        if not certification:
            self.question_count += 1
        for unit_id in known_ids:
            status = self.statuses[unit_id]
            status.attempts += 1
            if success:
                status.passes += 1
                status.covered = True
                if certification:
                    status.certified_passes += 1
            else:
                status.failures += 1
        if certification:
            self.certification_attempts += 1
            if success:
                self.consecutive_certification_passes += 1
            else:
                self.certification_failures += 1
                self.consecutive_certification_passes = 0

    def structural_coverage(self) -> float:
        total = sum(unit.weight for unit in self.units.values())
        if total <= 0:
            return 1.0
        covered = sum(
            self.units[unit_id].weight
            for unit_id, status in self.statuses.items()
            if status.covered
        )
        return covered / total

    def critical_coverage(self) -> float:
        critical = [unit for unit in self.units.values() if unit.critical]
        if not critical:
            return 1.0
        return sum(self.statuses[unit.unit_id].covered for unit in critical) / len(critical)

    def pending_weight(self, unit_ids: Iterable[str]) -> float:
        return sum(
            self.units[unit_id].weight
            for unit_id in unit_ids
            if unit_id in self.units and not self.statuses[unit_id].covered
        )

    def route_weight(self, unit_ids: Iterable[str]) -> float:
        return sum(self.units[unit_id].weight for unit_id in unit_ids if unit_id in self.units)

    def is_coverage_ready(self, threshold: float = 0.98, critical_threshold: float = 1.0) -> bool:
        return self.structural_coverage() >= threshold and self.critical_coverage() >= critical_threshold

    def snapshot(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question_count": self.question_count,
            "structural_coverage": self.structural_coverage(),
            "critical_coverage": self.critical_coverage(),
            "certification_attempts": self.certification_attempts,
            "certification_failures": self.certification_failures,
            "consecutive_certification_passes": self.consecutive_certification_passes,
            "units": {
                unit_id: {**self.units[unit_id].to_dict(), **status.to_dict()}
                for unit_id, status in self.statuses.items()
            },
            "uncovered_unit_ids": [
                unit_id for unit_id, status in self.statuses.items() if not status.covered
            ],
        }


class CoverageAwareRouteScheduler:
    """Keep random-walk routing, but prefer walks that contain uncovered units."""

    def __init__(
        self,
        base_policy: RandomWalkRoutingPolicy,
        candidate_attempts: int = 16,
        random_exploration_ratio: float = 0.15,
    ):
        self.base_policy = base_policy
        self.candidate_attempts = max(1, int(candidate_attempts))
        self.random_exploration_ratio = min(1.0, max(0.0, float(random_exploration_ratio)))

    def select_route(
        self,
        graph: Dict[str, Any],
        tracker: Optional[CaseCoverageTracker],
        seed: int,
    ) -> GraphRoute:
        if tracker is None or random.Random(seed).random() < self.random_exploration_ratio:
            return self.base_policy.select_route(graph, seed=seed)
        best_route = None
        best_score = float("-inf")
        for attempt in range(self.candidate_attempts):
            route = self.base_policy.select_route(graph, seed=seed + attempt * 7919)
            unit_ids = tracker.unit_ids_for_route(route)
            pending = tracker.pending_weight(unit_ids)
            failures = sum(
                tracker.statuses[unit_id].failures
                for unit_id in unit_ids
                if unit_id in tracker.statuses
            )
            try:
                route_score = float(route.score)
            except (TypeError, ValueError):
                route_score = 0.0
            score = 4.0 * pending + 0.5 * failures + 0.01 * route_score
            if score > best_score:
                best_route = route
                best_score = score
        return best_route or self.base_policy.select_route(graph, seed=seed)


def _public_edge(edge: Dict[str, Any], entities: Dict[str, Dict[str, Any]]) -> bool:
    metadata = edge.get("metadata", {})
    if edge.get("description") == "target_answer":
        return False
    if isinstance(metadata, dict) and metadata.get("source") == EVALUATOR_METADATA_SOURCE:
        return False
    for endpoint in (str(edge.get("source", "")), str(edge.get("target", ""))):
        entity = entities.get(endpoint, {})
        entity_metadata = entity.get("metadata", {})
        if isinstance(entity_metadata, dict) and entity_metadata.get("source") == EVALUATOR_METADATA_SOURCE:
            return False
    return True
