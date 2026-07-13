"""Coverage tracking and coverage-aware scheduling for memory construction."""

from __future__ import annotations

import hashlib
import random
import re
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
    kind: str = "relationship"
    source_ids: List[str] = field(default_factory=list)
    qualifiers: List[str] = field(default_factory=list)
    critical: bool = False
    weight: float = 1.0
    priority: float = 1.0
    critical_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "kind": self.kind,
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "source_ids": self.source_ids,
            "qualifiers": self.qualifiers,
            "critical": self.critical,
            "weight": self.weight,
            "priority": self.priority,
            "critical_reason": self.critical_reason,
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
        self.certification_resets = 0
        self.consecutive_certification_passes = 0

    def add_unit(self, unit: CoverageUnit) -> None:
        """Register a target-free composite probe before construction starts."""

        if unit.unit_id in self.units:
            return
        self.units[unit.unit_id] = unit
        self.statuses[unit.unit_id] = CoverageStatus()

    @classmethod
    def from_graph(cls, graph: Dict[str, Any]) -> "CaseCoverageTracker":
        entities = {
            str(item.get("name", "")): item
            for item in graph.get("entities", [])
            if isinstance(item, dict)
        }
        units = []
        chunks = {
            str(item.get("chunk_id", "")): item
            for item in graph.get("chunks", [])
            if isinstance(item, dict)
        }
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
            critical, priority, critical_reason = _coverage_importance(
                edge=edge,
                entities=entities,
                chunks=chunks,
            )
            units.append(
                CoverageUnit(
                    unit_id=relationship_unit_id(edge),
                    source=source,
                    relation=relation,
                    target=target,
                    source_ids=[str(item) for item in edge.get("source_ids", [])],
                    qualifiers=qualifiers,
                    critical=critical,
                    weight=max(
                        1.0,
                        max(0.0, float(edge.get("weight", 1.0) or 1.0)) ** 0.5,
                    ),
                    priority=priority,
                    critical_reason=critical_reason,
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

    def mark_covered(self, unit_ids: Iterable[str]) -> None:
        """Mark facts directly verified in committed memory without adding probe attempts."""

        for unit_id in unit_ids:
            status = self.statuses.get(unit_id)
            if status is None or status.covered:
                continue
            status.covered = True
            status.passes += 1

    def reset_certification(self) -> None:
        """Invalidate certification evidence after the memory store changes."""

        self.certification_resets += 1
        self.consecutive_certification_passes = 0
        for status in self.statuses.values():
            status.certified_passes = 0

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

    def required_coverage(self) -> float:
        """Weighted coverage over memory-worthy units, excluding background facts."""

        required = [unit for unit in self.units.values() if unit.critical]
        if not required:
            return self.structural_coverage()
        total = sum(unit.weight for unit in required)
        if total <= 0:
            return 1.0
        covered = sum(
            unit.weight
            for unit in required
            if self.statuses[unit.unit_id].covered
        )
        return covered / total

    def pending_weight(self, unit_ids: Iterable[str]) -> float:
        return sum(
            self.units[unit_id].weight
            for unit_id in unit_ids
            if unit_id in self.units and not self.statuses[unit_id].covered
        )

    def route_weight(self, unit_ids: Iterable[str]) -> float:
        return sum(self.units[unit_id].weight for unit_id in unit_ids if unit_id in self.units)

    def route_priority(self, unit_ids: Iterable[str]) -> float:
        return sum(self.units[unit_id].priority for unit_id in unit_ids if unit_id in self.units)

    def critical_pending_weight(self, unit_ids: Iterable[str]) -> float:
        return sum(
            self.units[unit_id].weight
            for unit_id in unit_ids
            if (
                unit_id in self.units
                and self.units[unit_id].critical
                and not self.statuses[unit_id].covered
            )
        )

    def critical_unit_count(self) -> int:
        return sum(unit.critical for unit in self.units.values())

    def covered_unit_count(self) -> int:
        return sum(status.covered for status in self.statuses.values())

    def is_coverage_ready(self, threshold: float = 0.98, critical_threshold: float = 1.0) -> bool:
        return self.required_coverage() >= threshold and self.critical_coverage() >= critical_threshold

    def snapshot(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question_count": self.question_count,
            "coverage_unit_count": len(self.units),
            "critical_unit_count": self.critical_unit_count(),
            "covered_unit_count": self.covered_unit_count(),
            "structural_coverage": self.structural_coverage(),
            "required_coverage": self.required_coverage(),
            "critical_coverage": self.critical_coverage(),
            "certification_attempts": self.certification_attempts,
            "certification_failures": self.certification_failures,
            "certification_resets": self.certification_resets,
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
            critical_pending = tracker.critical_pending_weight(unit_ids)
            priority = tracker.route_priority(unit_ids)
            score = (
                6.0 * critical_pending
                + 4.0 * pending
                + 0.5 * failures
                + 0.25 * priority
                + 0.01 * route_score
            )
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


_EPISODIC_RELATION_TERMS = {
    "appointment",
    "attend",
    "bought",
    "completed",
    "consult",
    "diagnosed",
    "experience",
    "graduated",
    "met",
    "occur",
    "participated",
    "plan",
    "scheduled",
    "started",
    "took",
    "travel",
    "undergo",
    "visit",
    "went",
}
_TEMPORAL_TYPES = {"time", "duration", "statistic"}
_PERSONAL_TYPES = {
    "behavior",
    "event",
    "goal",
    "goal/intention",
    "health",
    "interest",
    "interest/skill",
    "person",
    "sentiment",
}
_FIRST_PERSON_RE = re.compile(r"\b(i|i'm|i've|me|my|mine|we|our|user)\b", re.IGNORECASE)


def _coverage_importance(
    edge: Dict[str, Any],
    entities: Dict[str, Dict[str, Any]],
    chunks: Dict[str, Dict[str, Any]],
) -> tuple[bool, float, str]:
    """Classify target-free episodic edges without using evaluator metadata."""

    metadata = edge.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("critical"), bool):
        critical = bool(metadata["critical"])
        return critical, 4.0 if critical else 1.0, "explicit_metadata"

    source = str(edge.get("source", ""))
    target = str(edge.get("target", ""))
    if "USER" in {source, target}:
        return True, 5.0, "user_centered"

    relation = str(edge.get("description") or edge.get("relation") or "").casefold()
    relation = relation.replace("<sep>", " ").replace("_", " ")
    relation_tokens = set(re.findall(r"[a-z0-9]+", relation))
    source_type = _entity_type(entities.get(source, {}))
    target_type = _entity_type(entities.get(target, {}))
    entity_types = {source_type, target_type}
    episodic_relation = bool(relation_tokens & _EPISODIC_RELATION_TERMS)
    temporal_edge = bool(entity_types & _TEMPORAL_TYPES)
    personal_edge = bool(entity_types & _PERSONAL_TYPES)

    source_text = " ".join(
        str(chunks.get(str(source_id), {}).get("content", ""))
        for source_id in edge.get("source_ids", [])
    )
    first_person = bool(_FIRST_PERSON_RE.search(source_text))

    if episodic_relation and (temporal_edge or personal_edge or first_person):
        return True, 4.5, "episodic_relation"
    if temporal_edge and (personal_edge or first_person):
        return True, 4.0, "temporal_personal"
    if first_person and personal_edge:
        return True, 3.5, "personal_session"
    return False, 1.0, "background_relation"


def _entity_type(entity: Dict[str, Any]) -> str:
    return str(entity.get("entity_type") or entity.get("type") or "").strip().casefold()
