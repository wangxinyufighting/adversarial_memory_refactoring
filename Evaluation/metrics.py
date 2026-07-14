"""Metrics for memory retrieval and LongMemEval answer quality."""

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


ANSWER_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", flags=re.IGNORECASE)
ARTICLES = {"a", "an", "the"}


def normalize_answer(value: Any) -> str:
    tokens = [
        token.casefold()
        for token in ANSWER_TOKEN_RE.findall(str(value or ""))
        if token.casefold() not in ARTICLES
    ]
    return " ".join(tokens)


def compute_answer_metrics(gold_answer: Any, candidate_answer: Any) -> Dict[str, Any]:
    gold = normalize_answer(gold_answer)
    candidate = normalize_answer(candidate_answer)
    return {
        "exact_match": float(bool(gold) and gold == candidate),
        "token_f1": _token_f1(gold.split(), candidate.split()),
        "candidate_unknown": float(_is_unknown(candidate)),
        "normalized_gold": gold,
        "normalized_candidate": candidate,
    }


def compute_retrieval_metrics(
    retrieved_memories: Sequence[Any],
    answer_source_ids: Iterable[str],
    *,
    all_memory_items: Optional[Sequence[Any]] = None,
    gold_answer: Any = None,
) -> Dict[str, Any]:
    """Measure retrieval against LongMemEval's answer-session annotations.

    ``source_ids`` are provenance annotations, not proof that the compressed
    text retained every fact from a source session. The end-to-end answer
    accuracy and lexical support metrics therefore remain separate.
    """

    gold_sources = {str(item) for item in answer_source_ids if str(item)}
    retrieved = [_item_dict(item) for item in retrieved_memories]
    full_memory = (
        [_item_dict(item) for item in all_memory_items]
        if all_memory_items is not None
        else None
    )

    retrieved_source_sets = [_source_ids(item, gold_sources) for item in retrieved]
    retrieved_sources = set().union(*retrieved_source_sets) if retrieved_source_sets else set()
    retrieved_gold = retrieved_sources & gold_sources
    relevant_chunks = sum(bool(source_ids & gold_sources) for source_ids in retrieved_source_sets)
    first_relevant_rank = _first_relevant_rank(retrieved, retrieved_source_sets, gold_sources)

    if full_memory is None:
        memory_sources: Set[str] = set()
        memory_gold: Set[str] = set()
        memory_source_hit = None
        memory_source_recall = None
        retrieval_recall_given_memory = None
        answer_in_memory = None
    else:
        memory_source_sets = [_source_ids(item, gold_sources) for item in full_memory]
        memory_sources = set().union(*memory_source_sets) if memory_source_sets else set()
        memory_gold = memory_sources & gold_sources
        memory_source_hit = float(bool(memory_gold)) if gold_sources else None
        memory_source_recall = _recall(memory_gold, gold_sources)
        retrieval_recall_given_memory = (
            len(retrieved_gold & memory_gold) / len(memory_gold) if memory_gold else 0.0
        )
        answer_in_memory = float(_answer_in_items(gold_answer, full_memory))

    return {
        "num_answer_sources": len(gold_sources),
        "num_retrieved_chunks": len(retrieved),
        "num_relevant_retrieved_chunks": relevant_chunks,
        "retrieved_answer_source_ids": sorted(retrieved_gold),
        "source_hit": float(bool(retrieved_gold)) if gold_sources else None,
        "source_recall": _recall(retrieved_gold, gold_sources),
        "all_sources_retrieved": (
            float(bool(gold_sources) and retrieved_gold == gold_sources)
            if gold_sources
            else None
        ),
        "relevant_chunk_precision": (
            relevant_chunks / len(retrieved) if retrieved else 0.0
        ),
        "first_relevant_rank": first_relevant_rank,
        "mrr": (1.0 / first_relevant_rank) if first_relevant_rank else 0.0,
        "memory_answer_source_ids": sorted(memory_gold),
        "memory_source_hit": memory_source_hit,
        "memory_source_recall": memory_source_recall,
        "retrieval_recall_given_memory": retrieval_recall_given_memory,
        "gold_answer_text_in_retrieved": float(_answer_in_items(gold_answer, retrieved)),
        "gold_answer_text_in_memory": answer_in_memory,
    }


def memory_size_metrics(memory_items: Sequence[Any], raw_session_chars: int = 0) -> Dict[str, Any]:
    items = [_item_dict(item) for item in memory_items]
    contents = [str(item.get("content") or item.get("text") or "") for item in items]
    memory_chars = sum(len(content) for content in contents)
    memory_tokens = sum(len(ANSWER_TOKEN_RE.findall(content)) for content in contents)
    return {
        "memory_chunks": len(items),
        "memory_chars": memory_chars,
        "memory_tokens_approx": memory_tokens,
        "raw_session_chars": int(raw_session_chars or 0),
        "memory_to_raw_char_ratio": (
            memory_chars / raw_session_chars if raw_session_chars else None
        ),
        "char_reduction": (
            1.0 - memory_chars / raw_session_chars if raw_session_chars else None
        ),
    }


def _item_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return item
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        return payload if isinstance(payload, dict) else {}
    return {
        "memory_id": getattr(item, "memory_id", ""),
        "content": getattr(item, "content", ""),
        "metadata": getattr(item, "metadata", {}) or {},
    }


def _source_ids(item: Dict[str, Any], gold_sources: Set[str]) -> Set[str]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    raw_ids: List[Any] = []
    for value in (item.get("source_ids"), metadata.get("source_ids")):
        if isinstance(value, (list, tuple, set)):
            raw_ids.extend(value)
        elif value:
            raw_ids.append(value)

    memory_id = str(item.get("memory_id") or item.get("chunk_id") or item.get("id") or "")
    if memory_id and memory_id in gold_sources:
        raw_ids.append(memory_id)
    return {str(value) for value in raw_ids if str(value)}


def _first_relevant_rank(
    items: Sequence[Dict[str, Any]],
    source_sets: Sequence[Set[str]],
    gold_sources: Set[str],
) -> Optional[int]:
    for fallback_rank, (item, source_ids) in enumerate(zip(items, source_sets), start=1):
        if source_ids & gold_sources:
            try:
                return int(item.get("rank") or fallback_rank)
            except (TypeError, ValueError):
                return fallback_rank
    return None


def _answer_in_items(answer: Any, items: Sequence[Dict[str, Any]]) -> bool:
    normalized_answer = normalize_answer(answer)
    if not normalized_answer:
        return False
    evidence = " ".join(_item_evidence(item) for item in items)
    normalized_evidence = normalize_answer(evidence)
    if not normalized_evidence:
        return False
    pattern = r"(?:^|\s)" + re.escape(normalized_answer) + r"(?:$|\s)"
    return re.search(pattern, normalized_evidence) is not None


def _item_evidence(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    parts = [str(item.get("content") or item.get("text") or "")]
    for key in ("facts", "summary", "keywords"):
        value = metadata.get(key)
        if isinstance(value, list):
            parts.extend(str(entry) for entry in value)
        elif value:
            parts.append(str(value))
    return "\n".join(parts)


def _recall(found: Set[str], gold: Set[str]) -> Optional[float]:
    return len(found & gold) / len(gold) if gold else None


def _token_f1(gold_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> float:
    if not gold_tokens or not candidate_tokens:
        return 0.0
    overlap = sum((Counter(gold_tokens) & Counter(candidate_tokens)).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _is_unknown(normalized: str) -> bool:
    return normalized in {
        "",
        "unknown",
        "unk",
        "none",
        "not known",
        "i don t know",
        "cannot determine",
        "not enough information",
    }
