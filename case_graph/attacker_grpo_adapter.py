"""GRPO adapter for training the memory attacker policy.

The attacker is rewarded for producing grounded, solvable attacks that expose
useful memory gaps. It is not rewarded for impossible questions or for merely
making the defender fail.
"""

import json
import math
import re
from typing import Any, Dict, Iterable, List

from .attacker import _extract_question_answer
from .retriever import FrozenBM25Retriever, MemoryStore


SYSTEM_PROMPT = (
    "You are the Attacker Policy for memory refactoring training. "
    "Generate one adversarial but answerable memory question from the given "
    "public graph-route evidence. Your goal is to expose missing or poorly "
    "compressed memory so the defender can build a better memory, not to make "
    "an impossible question. Return JSON only: "
    '{"question": "...", "answer": "...", "evidence_ids": ["..."], "attack_type": "..."}'
)


ATTACKER_REWARD_KEYS = (
    "score",
    "format_valid",
    "grounded",
    "oracle_solvable",
    "memory_gap",
    "question_relevance",
    "novelty",
    "route_complexity",
    "answer_leak_penalty",
    "length_penalty",
)


def build_attacker_verl_row(state: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """Convert an attacker state into a verl row."""
    ground_truth = {
        "case_id": state.get("case_id", "unknown"),
        "uid": state.get("uid", ""),
        "step": state.get("step", state.get("episode", 0)),
        "episode": state.get("episode", 0),
        "route": state.get("route", {}),
        "golden_facts": state.get("golden_facts", []),
        "current_memory": state.get("current_memory", []),
        "memory_observation": state.get("memory_observation", {}),
        "recent_attacks": state.get("recent_attacks", []),
        "top_k": state.get("top_k", 5),
    }
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_attacker_user_prompt(state)},
    ]
    uid = state.get("uid", f"{state.get('case_id', 'unknown')}_att_ep{state.get('episode', 0)}")
    return {
        "data_source": "memory_attacker_online",
        "prompt": prompt_messages,
        "raw_prompt": prompt_messages,
        "index": index,
        "reward_model": {"ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
        "extra_info": {
            "uid": uid,
            "case_id": state.get("case_id", "unknown"),
            "episode": state.get("episode", 0),
            "role": "attacker",
        },
    }


def build_attacker_user_prompt(state: Dict[str, Any]) -> str:
    payload = {
        "objective": (
            "Find a grounded, answerable question that reveals a useful gap in "
            "the current compressed memory. Do not create an impossible, vague, "
            "or unsupported question."
        ),
        "route_evidence": state.get("route", {}),
        "current_memory_view": state.get("current_memory", []),
        "defender_behavior_observation": state.get("memory_observation", {}),
        "recent_attacks_to_avoid": state.get("recent_attacks", []),
        "required_output_schema": {
            "question": "string",
            "answer": "string grounded in route_evidence",
            "evidence_ids": ["source ids used by the answer"],
            "attack_type": "missing_fact | underlinked_fact | multi_hop | disambiguation",
        },
        "quality_rules": [
            "The answer must be directly supported by route_evidence.",
            "Prefer facts absent from or hard to retrieve from current_memory_view.",
            "The question must have one clear referent.",
            "Do not mention route, graph, evidence, memory chunks, or IDs in the question.",
            "Do not put the answer verbatim in the question.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def compute_attacker_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | Dict[str, Any],
    extra_info: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Reward one generated attacker question."""
    del data_source, extra_info
    state = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    try:
        payload = _parse_json(solution_str)
    except (json.JSONDecodeError, ValueError):
        return _attacker_reward_payload({"score": -3.0, "format_valid": 0.0})

    question, answer = _extract_question_answer(payload)
    if not question or not answer:
        return _attacker_reward_payload({"score": -3.0, "format_valid": 0.0})

    route_text = _route_text(state)
    gold_text = _gold_text(state)
    support_text = f"{route_text}\n{gold_text}"
    grounded = float(_answer_in_text(answer, support_text))
    oracle_solvable = grounded
    if not grounded:
        return _attacker_reward_payload(
            {
                "score": -2.5,
                "format_valid": 1.0,
                "grounded": 0.0,
                "oracle_solvable": 0.0,
            }
        )

    memory_store = _memory_store_from_state(state)
    hits = FrozenBM25Retriever(memory_store).retrieve(
        question,
        top_k=int(state.get("top_k", 5)),
        min_score=0.0,
    )
    retrieved_text = "\n".join(hit.content for hit in hits)
    memory_gap = float(not _answer_in_text(answer, retrieved_text))
    relevance = _question_relevance(question, support_text)
    novelty = _novelty(question, state.get("recent_attacks", []))
    complexity = _route_complexity(state.get("route", {}))
    answer_leak_penalty = float(_answer_in_text(answer, question))
    length_penalty = _length_penalty(question, answer)

    score = (
        0.50
        + 1.20 * grounded
        + 0.90 * oracle_solvable
        + 1.20 * memory_gap
        + 0.50 * relevance
        + 0.35 * novelty
        + 0.25 * complexity
        - 1.00 * answer_leak_penalty
        - 0.60 * length_penalty
    )
    return _attacker_reward_payload(
        {
            "score": score,
            "format_valid": 1.0,
            "grounded": grounded,
            "oracle_solvable": oracle_solvable,
            "memory_gap": memory_gap,
            "question_relevance": relevance,
            "novelty": novelty,
            "route_complexity": complexity,
            "answer_leak_penalty": answer_leak_penalty,
            "length_penalty": length_penalty,
        }
    )


def _attacker_reward_payload(values: Dict[str, float]) -> Dict[str, float]:
    return {key: float(values.get(key, 0.0)) for key in ATTACKER_REWARD_KEYS}


def _parse_json(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object found in attacker output")
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Attacker output must be a JSON object")
    return payload


def _memory_store_from_state(state: Dict[str, Any]) -> MemoryStore:
    payload = state.get("current_memory") or state.get("memory") or []
    if isinstance(payload, MemoryStore):
        return payload
    if isinstance(payload, list):
        return MemoryStore.from_dicts(payload)
    if isinstance(payload, dict):
        return MemoryStore.from_dicts(payload.get("memories", payload.get("chunks", [])))
    return MemoryStore()


def _route_text(state: Dict[str, Any]) -> str:
    return json.dumps(state.get("route", {}), ensure_ascii=False)


def _gold_text(state: Dict[str, Any]) -> str:
    return "\n".join(
        str(item.get("text") or item.get("content") or item)
        for item in state.get("golden_facts", [])
    )


def _question_relevance(question: str, support_text: str) -> float:
    q_tokens = set(_tokens(question))
    support_tokens = set(_tokens(support_text))
    if not q_tokens or not support_tokens:
        return 0.0
    overlap = len(q_tokens & support_tokens) / max(1, len(q_tokens))
    return min(1.0, overlap * 2.0)


def _novelty(question: str, recent_attacks: Iterable[Any]) -> float:
    q_tokens = set(_tokens(question))
    if not q_tokens:
        return 0.0
    max_similarity = 0.0
    for item in recent_attacks:
        if isinstance(item, dict):
            other = str(item.get("question", ""))
        else:
            other = str(item)
        other_tokens = set(_tokens(other))
        if not other_tokens:
            continue
        max_similarity = max(
            max_similarity,
            len(q_tokens & other_tokens) / max(1, len(q_tokens | other_tokens)),
        )
    return max(0.0, 1.0 - max_similarity)


def _route_complexity(route: Dict[str, Any]) -> float:
    relationships = route.get("relationships", [])
    entities = route.get("entities", [])
    nodes = route.get("nodes", [])
    raw = min(1.0, (len(relationships) + len(entities) + len(nodes)) / 12.0)
    return math.sqrt(raw)


def _length_penalty(question: str, answer: str) -> float:
    q_len = len(_tokens(question))
    a_len = len(_tokens(answer))
    too_short = q_len < 5 or a_len < 1
    too_long = q_len > 48 or a_len > 24
    return float(too_short or too_long)


def _answer_in_text(answer: str, text: str) -> bool:
    answer_norm = _normalize(answer)
    text_norm = _normalize(text)
    return bool(answer_norm and answer_norm in text_norm)


def _normalize(text: str) -> str:
    tokens = _tokens(text)
    tokens = [token for token in tokens if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", str(text or "").casefold())
