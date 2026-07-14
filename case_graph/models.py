import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple


GRAPH_FIELD_SEP = "<SEP>"
COMPACT_USER_DESCRIPTION = "The user in this case."
EVALUATOR_METADATA_SOURCE = "evaluator_target_safeguard"
CANONICAL_ENTITY_TYPES = {
    "behavior": "Behavior",
    "duration": "Duration",
    "event": "Event",
    "goal": "Goal/Intention",
    "goal/intention": "Goal/Intention",
    "health": "Health",
    "intention": "Goal/Intention",
    "interest": "Interest/Skill",
    "interest/skill": "Interest/Skill",
    "object": "Object",
    "organization": "Organization",
    "other": "Other",
    "person": "Person",
    "place": "Place",
    "resource": "Resource",
    "sentiment": "Sentiment",
    "skill": "Interest/Skill",
    "statistic": "Statistic",
    "time": "Time",
    "user": "User",
}


def normalize_name(value: str) -> str:
    """Normalize entity names the same simple way GraphRAG-style systems do."""
    return " ".join(str(value).strip().strip('"').split()).upper()


def normalize_entity_type(value: str) -> str:
    raw = " ".join(str(value or "Other").strip().strip('"').split())
    return CANONICAL_ENTITY_TYPES.get(raw.lower(), raw or "Other")


def _append_unique_text(existing: str, incoming: str) -> str:
    incoming = str(incoming or "").strip()
    if not incoming:
        return existing
    if not existing:
        return incoming
    parts = [p.strip() for p in existing.split(GRAPH_FIELD_SEP) if p.strip()]
    if incoming not in parts:
        parts.append(incoming)
    return GRAPH_FIELD_SEP.join(parts)


def _append_unique_ids(existing: List[str], incoming: Iterable[str]) -> List[str]:
    seen = set(existing)
    result = list(existing)
    for item in incoming:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _answer_texts(answer: Any) -> List[str]:
    if isinstance(answer, list):
        values = answer
    else:
        values = [answer]
    return [str(value).strip() for value in values if str(value).strip()]


def _contains_text(haystack: str, needle: str) -> bool:
    needle_tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", str(needle or "").casefold())
    haystack_tokens = re.findall(
        r"[a-z0-9]+|[\u4e00-\u9fff]", str(haystack or "").casefold()
    )
    if not needle_tokens or len(needle_tokens) > len(haystack_tokens):
        return False
    width = len(needle_tokens)
    return any(
        haystack_tokens[index : index + width] == needle_tokens
        for index in range(len(haystack_tokens) - width + 1)
    )


@dataclass(frozen=True)
class SessionChunk:
    case_id: str
    chunk_id: str
    content: str
    timestamp: str = ""
    order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_content: bool = False) -> Dict[str, Any]:
        payload = {
            "case_id": self.case_id,
            "chunk_id": self.chunk_id,
            "timestamp": self.timestamp,
            "order": self.order,
            "metadata": self.metadata,
        }
        if include_content:
            payload["content"] = self.content
        return payload


@dataclass(frozen=True)
class EntityRecord:
    name: str
    entity_type: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": normalize_entity_type(self.entity_type),
            "description": self.description,
        }


@dataclass(frozen=True)
class RelationshipRecord:
    source: str
    target: str
    description: str
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "description": self.description,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class ExtractionResult:
    entities: List[EntityRecord] = field(default_factory=list)
    relationships: List[RelationshipRecord] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExtractionResult":
        entities = [
            EntityRecord(
                name=item.get("name") or item.get("entity_name") or "",
                entity_type=normalize_entity_type(item.get("type") or item.get("entity_type") or "Other"),
                description=item.get("description") or item.get("entity_description") or "",
            )
            for item in payload.get("entities", [])
            if item.get("name") or item.get("entity_name")
        ]
        relationships = []
        for item in payload.get("relationships", []):
            source = item.get("source") or item.get("src") or item.get("source_entity") or ""
            target = item.get("target") or item.get("tgt") or item.get("target_entity") or ""
            if not source or not target:
                continue
            relationships.append(
                RelationshipRecord(
                    source=source,
                    target=target,
                    description=item.get("description")
                    or item.get("relationship")
                    or item.get("predicate")
                    or "",
                    weight=float(item.get("weight") or item.get("strength") or 1.0),
                )
            )
        return cls(entities=entities, relationships=relationships)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "relationships": [relationship.to_dict() for relationship in self.relationships],
        }


@dataclass
class EntityNode:
    name: str
    entity_type: str
    description: str = ""
    source_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def merge(self, record: EntityRecord, chunk_id: str) -> None:
        if self.name == "USER":
            self.entity_type = "User"
            self.description = COMPACT_USER_DESCRIPTION
            self.source_ids = _append_unique_ids(self.source_ids, [chunk_id])
            return
        incoming_type = normalize_entity_type(record.entity_type)
        if not self.entity_type or self.entity_type.lower() == "other":
            self.entity_type = incoming_type
        self.description = _append_unique_text(self.description, record.description)
        self.source_ids = _append_unique_ids(self.source_ids, [chunk_id])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "source_ids": self.source_ids,
            "metadata": self.metadata,
        }


@dataclass
class RelationshipEdge:
    source: str
    target: str
    description: str = ""
    weight: float = 0.0
    source_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def merge(self, record: RelationshipRecord, chunk_id: str) -> None:
        self.description = _append_unique_text(self.description, record.description)
        self.weight += float(record.weight)
        self.source_ids = _append_unique_ids(self.source_ids, [chunk_id])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "description": self.description,
            "weight": self.weight,
            "source_ids": self.source_ids,
            "metadata": self.metadata,
        }


@dataclass
class CaseGraph:
    case_id: str
    chunks: List[SessionChunk] = field(default_factory=list)
    entities: Dict[str, EntityNode] = field(default_factory=dict)
    relationships: Dict[Tuple[str, str], RelationshipEdge] = field(default_factory=dict)
    target: Dict[str, Any] = field(default_factory=dict)

    def add_chunk(self, chunk: SessionChunk) -> None:
        if chunk.chunk_id not in {item.chunk_id for item in self.chunks}:
            self.chunks.append(chunk)

    def apply_extraction(self, chunk_id: str, extraction: ExtractionResult) -> None:
        for record in extraction.entities:
            name = normalize_name(record.name)
            if not name:
                continue
            entity_type = normalize_entity_type(record.entity_type)
            if name not in self.entities:
                self.entities[name] = EntityNode(
                    name=name,
                    entity_type=entity_type,
                )
            self.entities[name].merge(record, chunk_id)

        for record in extraction.relationships:
            source = normalize_name(record.source)
            target = normalize_name(record.target)
            if not source or not target or source == target:
                continue
            key = (source, target)
            if key not in self.relationships:
                self.relationships[key] = RelationshipEdge(source=source, target=target)
            self.relationships[key].merge(record, chunk_id)

    def ensure_target_answer(
        self,
        question: str,
        answer: Any,
        source_ids: Iterable[str] = (),
        inject_missing: bool = True,
    ) -> None:
        """Record target labels and optionally add a legacy evaluator-only answer unit."""
        answer_texts = _answer_texts(answer)
        answer_source_ids = list(source_ids)
        self.target = {
            "question": question,
            "answer": answer,
            "answer_source_ids": answer_source_ids,
        }
        if not inject_missing:
            return
        for answer_text in answer_texts:
            if self.contains_answer(answer_text):
                continue
            answer_name = normalize_name(answer_text)
            if not answer_name:
                continue
            self.apply_extraction(
                chunk_id=answer_source_ids[0] if answer_source_ids else "target_answer",
                extraction=ExtractionResult(
                    entities=[
                        EntityRecord(
                            name=answer_text,
                            entity_type="Other",
                            description="Memory fact recovered from evaluator metadata.",
                        ),
                    ],
                    relationships=[
                        RelationshipRecord(
                            source="User",
                            target=answer_text,
                            description="related_to",
                            weight=10.0,
                        )
                    ],
                ),
            )
            if answer_source_ids:
                self.entities[answer_name].source_ids = _append_unique_ids(
                    [],
                    answer_source_ids,
                )
                self.relationships[("USER", answer_name)].source_ids = _append_unique_ids(
                    [],
                    answer_source_ids,
                )
            self.entities[answer_name].metadata["source"] = EVALUATOR_METADATA_SOURCE
            self.relationships[("USER", answer_name)].metadata["source"] = EVALUATOR_METADATA_SOURCE

    def contains_answer(self, answer: str) -> bool:
        answer = str(answer or "").strip()
        if not answer:
            return True
        for entity in self.entities.values():
            if _contains_text(entity.name, answer) or _contains_text(entity.description, answer):
                return True
        for relationship in self.relationships.values():
            if (
                _contains_text(relationship.source, answer)
                or _contains_text(relationship.target, answer)
                or _contains_text(relationship.description, answer)
            ):
                return True
        return False

    def to_dict(self, include_chunk_content: bool = True) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "target": self.target,
            "chunks": [
                chunk.to_dict(include_content=include_chunk_content)
                for chunk in sorted(self.chunks, key=lambda c: c.order)
            ],
            "entities": [
                self.entities[key].to_dict()
                for key in sorted(self.entities)
            ],
            "relationships": [
                self.relationships[key].to_dict()
                for key in sorted(self.relationships)
            ],
        }
