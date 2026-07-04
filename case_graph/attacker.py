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
                "Return JSON only with question and answer."
            ),
            user_prompt=json.dumps(
                {
                    "case_id": case_id,
                    "route": route,
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
        return AttackExample(
            case_id=case_id,
            question=str(response.get("question", "")),
            answer=str(response.get("answer", "")),
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
