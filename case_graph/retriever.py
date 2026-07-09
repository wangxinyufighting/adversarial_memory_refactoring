import json
import math
import re
import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


@dataclass
class MemoryChunk:
    """Structured memory unit used by the defense stage."""

    memory_id: str
    content: str
    linked_questions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], fallback_id: str = "") -> "MemoryChunk":
        memory_id = str(
            payload.get("memory_id")
            or payload.get("id")
            or payload.get("chunk_id")
            or payload.get("session_id")
            or fallback_id
        )
        linked_questions = payload.get("linked_questions", [])
        if isinstance(linked_questions, str):
            linked_questions = [linked_questions]
        metadata = dict(payload.get("metadata", {}))
        for key in ("facts", "summary", "keywords", "source_ids", "timestamp"):
            if key in payload and key not in metadata:
                metadata[key] = payload[key]
        return cls(
            memory_id=memory_id,
            content=str(payload.get("content") or payload.get("text") or payload.get("summary") or ""),
            linked_questions=[str(item) for item in linked_questions],
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "linked_questions": self.linked_questions,
            "metadata": self.metadata,
        }

    def bind_question(self, question: str) -> None:
        question = str(question or "").strip()
        if question and question not in self.linked_questions:
            self.linked_questions.append(question)


@dataclass(frozen=True)
class RetrievalHit:
    memory_id: str
    content: str
    score: float
    rank: int
    linked_questions: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "score": self.score,
            "rank": self.rank,
            "linked_questions": self.linked_questions,
            "metadata": self.metadata,
        }


class MemoryStore:
    """In-memory representation of Mt, the current structured memory library."""

    def __init__(self, chunks: Iterable[MemoryChunk] = ()):
        self.chunks = list(chunks)

    @classmethod
    def from_dicts(cls, items: Iterable[Dict[str, Any]]) -> "MemoryStore":
        return cls(
            MemoryChunk.from_dict(item, fallback_id=f"memory-{index}")
            for index, item in enumerate(items)
        )

    @classmethod
    def load(cls, path: str | Path) -> "MemoryStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = payload.get("memories") or payload.get("chunks") or payload.get("memory_chunks") or []
        else:
            raise ValueError("Memory store must be a JSON object or list.")
        if not isinstance(items, list):
            raise ValueError("Memory store items must be a list.")
        return cls.from_dicts(items)

    def save(self, path: str | Path) -> None:
        payload = {"memories": [chunk.to_dict() for chunk in self.chunks]}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def bind_question(self, memory_ids: Iterable[str], question: str) -> None:
        wanted = set(memory_ids)
        for chunk in self.chunks:
            if chunk.memory_id in wanted:
                chunk.bind_question(question)


class FrozenBM25Retriever:
    """
    Frozen sparse retriever for stage-three old-memory lookup.

    The index is built from Mt contents only; no parameters are learned or updated
    during retrieval. Scores are BM25-style and deterministic.
    """

    def __init__(
        self,
        memory_store: MemoryStore | Sequence[MemoryChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.memory_store = (
            memory_store if isinstance(memory_store, MemoryStore) else MemoryStore(memory_store)
        )
        self.k1 = k1
        self.b = b
        self._tokenized = [_tokenize(chunk.content) for chunk in self.memory_store.chunks]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokenized]
        self._doc_lengths = [len(tokens) for tokens in self._tokenized]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        self._idf = self._build_idf()

    @classmethod
    def from_path(cls, path: str | Path, k1: float = 1.5, b: float = 0.75) -> "FrozenBM25Retriever":
        return cls(MemoryStore.load(path), k1=k1, b=b)

    def retrieve(self, question: str, top_k: int = 5, min_score: float = 0.0) -> List[RetrievalHit]:
        if top_k <= 0 or not self.memory_store.chunks:
            return []
        query_terms = _tokenize(question)
        if not query_terms:
            return []

        scored = []
        for index, chunk in enumerate(self.memory_store.chunks):
            score = self._score(query_terms, index)
            if score >= min_score:
                scored.append((score, index, chunk))

        scored.sort(key=lambda item: (-item[0], item[1]))
        hits = []
        for rank, (score, _, chunk) in enumerate(scored[:top_k], start=1):
            hits.append(
                RetrievalHit(
                    memory_id=chunk.memory_id,
                    content=chunk.content,
                    score=score,
                    rank=rank,
                    linked_questions=list(chunk.linked_questions),
                    metadata=dict(chunk.metadata),
                )
            )
        return hits

    def _build_idf(self) -> Dict[str, float]:
        document_count = len(self._tokenized)
        document_frequencies: Counter[str] = Counter()
        for tokens in self._tokenized:
            document_frequencies.update(set(tokens))
        return {
            term: math.log(1.0 + (document_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequencies.items()
        }

    def _score(self, query_terms: Sequence[str], doc_index: int) -> float:
        if not self._avg_doc_length:
            return 0.0
        score = 0.0
        term_frequencies = self._term_frequencies[doc_index]
        doc_length = self._doc_lengths[doc_index]
        for term in query_terms:
            frequency = term_frequencies.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * doc_length / self._avg_doc_length
            )
            score += self._idf.get(term, 0.0) * (frequency * (self.k1 + 1.0)) / denominator
        return score


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text or "").casefold())


@dataclass(frozen=True)
class RetrievalPoint:
    memory_id: str
    chunk_index: int
    point: str
    point_type: str
    weight: float = 1.0


class DenseStructuredMemoryRetriever:
    """Dense, UnifiedMem-style retriever over structured memory fields.

    The retriever indexes compact memory chunks through multiple retrieval keys:
    facts, summary, keywords, and content.  In ``flatten`` mode each field is
    ranked as an independent point and then mapped back to its source memory
    chunk.  This mirrors the non graph-based UnifiedMem retrieval path while
    keeping a small dependency surface for the CaseGraph training loop.
    """

    def __init__(
        self,
        memory_store: MemoryStore | Sequence[MemoryChunk],
        model_name: str = "facebook/contriever",
        embedding_model: str = "contriever",
        retrieval_mode: str = "flatten",
        top_k_points: int = 24,
        fields: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        require_model: bool = False,
        hash_dim: int = 384,
    ):
        self.memory_store = (
            memory_store if isinstance(memory_store, MemoryStore) else MemoryStore(memory_store)
        )
        self.model_name = str(model_name or "facebook/contriever")
        self.embedding_model = str(embedding_model or "contriever")
        self.retrieval_mode = str(retrieval_mode or "flatten")
        self.top_k_points = int(top_k_points or 24)
        self.fields = list(fields or ("facts", "summary", "keywords", "content"))
        self.device = device
        self.cache_dir = cache_dir
        self.require_model = bool(require_model)
        self.hash_dim = int(hash_dim or 384)
        self.points = self._build_points()
        self._encoder = None
        self._point_embeddings: Optional[List[List[float]]] = None

    def retrieve(self, question: str, top_k: int = 5, min_score: float = 0.0) -> List[RetrievalHit]:
        if top_k <= 0 or not self.points or not str(question or "").strip():
            return []

        scores = self._score_points(question)
        scored_points = [
            (score * point.weight, score, point)
            for score, point in zip(scores, self.points)
            if score * point.weight >= min_score
        ]
        scored_points.sort(key=lambda item: (-item[0], item[2].chunk_index, item[2].point_type))

        if self.retrieval_mode == "flatten":
            candidate_points = scored_points[: max(self.top_k_points, top_k)]
        else:
            candidate_points = scored_points

        by_memory: Dict[str, Dict[str, Any]] = {}
        for weighted_score, raw_score, point in candidate_points:
            record = by_memory.setdefault(
                point.memory_id,
                {
                    "score": weighted_score,
                    "chunk_index": point.chunk_index,
                    "points": [],
                },
            )
            record["score"] = max(float(record["score"]), float(weighted_score))
            record["points"].append(
                {
                    "point": point.point,
                    "point_type": point.point_type,
                    "score": float(raw_score),
                    "weighted_score": float(weighted_score),
                }
            )

        ranked = sorted(
            by_memory.items(),
            key=lambda item: (-float(item[1]["score"]), int(item[1]["chunk_index"])),
        )[:top_k]

        hits: List[RetrievalHit] = []
        for rank, (memory_id, record) in enumerate(ranked, start=1):
            chunk = self.memory_store.chunks[int(record["chunk_index"])]
            metadata = dict(chunk.metadata)
            metadata["retrieval_points"] = record["points"][: self.top_k_points]
            metadata["retriever"] = {
                "type": "dense_structured",
                "embedding_model": self.embedding_model,
                "model_name": self.model_name,
                "retrieval_mode": self.retrieval_mode,
            }
            hits.append(
                RetrievalHit(
                    memory_id=memory_id,
                    content=chunk.content,
                    score=float(record["score"]),
                    rank=rank,
                    linked_questions=list(chunk.linked_questions),
                    metadata=metadata,
                )
            )
        return hits

    def _build_points(self) -> List[RetrievalPoint]:
        points: List[RetrievalPoint] = []
        mode = self.retrieval_mode
        for index, chunk in enumerate(self.memory_store.chunks):
            field_points = []
            for field_name in self.fields:
                for value in _field_values(chunk, field_name):
                    text = str(value or "").strip()
                    if not text:
                        continue
                    field_points.append(
                        RetrievalPoint(
                            memory_id=chunk.memory_id,
                            chunk_index=index,
                            point=text,
                            point_type=field_name,
                            weight=_field_weight(field_name),
                        )
                    )
            if not field_points and chunk.content.strip():
                field_points.append(
                    RetrievalPoint(
                        memory_id=chunk.memory_id,
                        chunk_index=index,
                        point=chunk.content.strip(),
                        point_type="content",
                        weight=_field_weight("content"),
                    )
                )
            if mode in {"merge", "merge_raw"} and field_points:
                merged = " ".join(point.point for point in field_points)
                points.append(
                    RetrievalPoint(
                        memory_id=chunk.memory_id,
                        chunk_index=index,
                        point=merged,
                        point_type="merged",
                        weight=1.0,
                    )
                )
            else:
                points.extend(field_points)
        return points

    def _score_points(self, question: str) -> List[float]:
        encoder = self._get_encoder()
        if self._point_embeddings is None:
            self._point_embeddings = encoder.encode([point.point for point in self.points])
        query_embedding = encoder.encode([question])[0]
        return [_dot(query_embedding, embedding) for embedding in self._point_embeddings]

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = _get_embedding_encoder(
                embedding_model=self.embedding_model,
                model_name=self.model_name,
                device=self.device,
                cache_dir=self.cache_dir,
                require_model=self.require_model,
                hash_dim=self.hash_dim,
            )
        return self._encoder


def build_memory_retriever(
    config: Optional[Dict[str, Any]],
    memory_store: MemoryStore | Sequence[MemoryChunk],
):
    """Build the configured retriever.

    Omitted config preserves the legacy BM25 behavior.  Training and evaluation
    configs should pass ``{"type": "dense_structured"}`` to use the optimized
    UnifiedMem-style dense retriever.
    """

    config = dict(config or {})
    retriever_type = str(
        config.get("type")
        or config.get("retriever_type")
        or config.get("name")
        or "bm25"
    ).casefold()
    if retriever_type in {"bm25", "frozen_bm25", "frozen-bm25"}:
        return FrozenBM25Retriever(
            memory_store,
            k1=float(config.get("k1", 1.5)),
            b=float(config.get("b", 0.75)),
        )
    if retriever_type in {"dense", "dense_structured", "structured_dense", "contriever"}:
        return DenseStructuredMemoryRetriever(
            memory_store,
            model_name=str(config.get("model_name") or config.get("retriever_model") or "facebook/contriever"),
            embedding_model=str(config.get("embedding_model") or "contriever"),
            retrieval_mode=str(config.get("retrieval_mode") or config.get("mode") or "flatten"),
            top_k_points=int(config.get("top_k_points", 24)),
            fields=config.get("fields"),
            device=config.get("device"),
            cache_dir=config.get("cache_dir"),
            require_model=bool(config.get("require_model", False)),
            hash_dim=int(config.get("hash_dim", 384)),
        )
    raise ValueError(f"Unsupported retriever type: {retriever_type}")


def retriever_config_from_mapping(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = dict(config or {})
    nested = config.get("retriever")
    if isinstance(nested, dict):
        result = dict(nested)
    else:
        result = {}
    aliases = {
        "retriever_type": "type",
        "retriever_model_name": "model_name",
        "retriever_embedding_model": "embedding_model",
        "retriever_retrieval_mode": "retrieval_mode",
        "retriever_top_k_points": "top_k_points",
        "retriever_device": "device",
        "retriever_cache_dir": "cache_dir",
        "retriever_require_model": "require_model",
    }
    for source, target in aliases.items():
        if source in config and target not in result:
            result[target] = config[source]
    return result


def _field_values(chunk: MemoryChunk, field_name: str) -> List[str]:
    field_name = str(field_name)
    metadata = chunk.metadata or {}
    if field_name == "content":
        return [chunk.content]
    value = metadata.get(field_name)
    if value is None and field_name == "facts":
        value = metadata.get("fact")
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False)]
    text = str(value)
    if field_name in {"facts", "keywords"}:
        return [item.strip() for item in re.split(r"[;\n]+", text) if item.strip()]
    return [text.strip()] if text.strip() else []


def _field_weight(field_name: str) -> float:
    return {
        "facts": 1.2,
        "summary": 1.0,
        "content": 0.95,
        "keywords": 0.75,
        "merged": 1.0,
    }.get(field_name, 1.0)


class _HashEmbeddingEncoder:
    def __init__(self, dim: int = 384):
        self.dim = int(dim or 384)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return [_normalize_vector(_hash_vector(text, self.dim)) for text in texts]


class _HFEmbeddingEncoder:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        torch = self.torch
        vectors: List[List[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.detach().cpu().tolist())
        return vectors


_ENCODER_CACHE: Dict[tuple, Any] = {}


def _get_embedding_encoder(
    embedding_model: str,
    model_name: str,
    device: Optional[str],
    cache_dir: Optional[str],
    require_model: bool,
    hash_dim: int,
):
    embedding_model = str(embedding_model or "").casefold()
    if embedding_model in {"hash", "hashed", "hash_dense"}:
        return _HashEmbeddingEncoder(dim=hash_dim)

    cache_key = (model_name, device or "", cache_dir or "")
    if cache_key in _ENCODER_CACHE:
        return _ENCODER_CACHE[cache_key]
    try:
        encoder = _HFEmbeddingEncoder(model_name=model_name, device=device, cache_dir=cache_dir)
        _ENCODER_CACHE[cache_key] = encoder
        return encoder
    except Exception as exc:
        if require_model:
            raise
        logger.warning(
            "Falling back to hashed dense retrieval because embedding model %s could not be loaded: %s",
            model_name,
            exc,
        )
        encoder = _HashEmbeddingEncoder(dim=hash_dim)
        _ENCODER_CACHE[cache_key] = encoder
        return encoder


def _hash_vector(text: str, dim: int) -> List[float]:
    vector = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    return vector


def _normalize_vector(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))
