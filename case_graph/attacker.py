import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from .llm import OpenAIChatClient
from .routing import GraphRoute, public_route_evidence


class AttackClient(Protocol):
    def complete_json(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        ...


@dataclass
class AttackExample:
    case_id: str
    question: str
    answer: str
    golden_facts: List[Dict[str, Any]]
    route: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "answer": self.answer,
            "golden_facts": self.golden_facts,
            "route": self.route,
        }


class FrozenLLMAttacker:
    """Frozen LLM attacker: no training, only prompt-time generation."""

    def __init__(self, client: Optional[AttackClient] = None, max_output_tokens: int = 700):
        self.client = client
        self.max_output_tokens = max_output_tokens

    def generate_from_route(
        self,
        case_id: str,
        route: Dict[str, Any],
        golden_facts: Optional[List[Dict[str, Any]]] = None,
    ) -> AttackExample:
        response = self._client().complete_json(
            system_prompt=(
                "You generate adversarial memory questions from graph-route evidence. "
                "Return JSON only. The JSON object must have exactly these required "
                "string keys: question and answer."
            ),
            user_prompt=json.dumps(
                {
                    "case_id": case_id,
                    "route": route,
                    "required_schema": {"question": "string", "answer": "string"},
                    "instruction": (
                        "Generate one tricky but answerable memory question Q and its correct answer A. "
                        "Both the question and answer must be grounded only in the route evidence. "
                        "The question must have exactly one plausible referent. "
                        "Avoid vague references such as 'the player', 'the team', 'the event', or 'the person' "
                        "unless they are made unique with qualifiers from the evidence. "
                        "When sports, teams, people, or organizations are involved, include the most specific "
                        "available qualifier such as sport, league, city, event, role, or full name. "
                        "Do not introduce temporal or causal wording such as before, after, prior to, following, "
                        "because, or caused by unless that relation is explicitly present in the route evidence. "
                        "For event-linked facts, prefer a simpler non-temporal question when possible. "
                        "Do not mention the route, graph, nodes, or edges in the question."
                    ),
                },
                ensure_ascii=False,
            ),
            max_tokens=self.max_output_tokens,
        )
        question, answer = _extract_question_answer(response)
        if not question or not answer:
            raise ValueError(f"Attacker response missing question or answer: {response}")
        return AttackExample(
            case_id=case_id,
            question=question,
            answer=answer,
            golden_facts=golden_facts or [],
            route=route,
        )

    def generate(self, graph: Dict[str, Any], route: GraphRoute) -> AttackExample:
        from .evidence import route_golden_facts

        return self.generate_from_route(
            graph.get("case_id", ""),
            public_route_evidence(graph, route),
            golden_facts=route_golden_facts(graph, route),
        )

    def _client(self) -> AttackClient:
        if self.client is None:
            self.client = OpenAIChatClient.from_env()
        return self.client


QUESTION_KEYS = {
    "question",
    "q",
    "query",
    "prompt",
    "memory_question",
    "generated_question",
}
ANSWER_KEYS = {
    "answer",
    "a",
    "gold_answer",
    "correct_answer",
    "target_answer",
    "expected_answer",
    "generated_answer",
}


def _extract_question_answer(response: Dict[str, Any]) -> tuple[str, str]:
    """Read Q/A from common model response shapes without accepting empty fields."""
    candidates = _response_candidates(response)
    for candidate in candidates:
        question = _first_string(candidate, QUESTION_KEYS)
        answer = _first_string(candidate, ANSWER_KEYS)
        if question and answer:
            return question, answer
    return "", ""


def _response_candidates(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    candidates = [response]
    for key in ("qa", "attack", "result", "output", "data"):
        value = response.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    for key in ("questions", "attacks", "items"):
        value = response.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
    return candidates


def _first_string(payload: Dict[str, Any], keys: set[str]) -> str:
    for key, value in payload.items():
        if str(key).casefold() in keys:
            text = _stringify_response_value(value)
            if text:
                return text
    return ""


def _stringify_response_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    if isinstance(value, dict):
        for key in ("text", "value", "content"):
            if key in value:
                text = _stringify_response_value(value[key])
                if text:
                    return text
    return ""
