import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from .llm import OpenAIChatClient
from .routing import GraphRoute


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
    golden_facts: List[str]
    route: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "golden_facts": self.golden_facts,
            "route": self.route,
        }


class FrozenLLMAttacker:
    """Frozen LLM attacker: no training, only prompt-time generation."""

    def __init__(self, client: Optional[AttackClient] = None, max_output_tokens: int = 700):
        self.client = client
        self.max_output_tokens = max_output_tokens

    def generate_from_route(self, case_id: str, route: Dict[str, Any]) -> AttackExample:
        response = self._client().complete_json(
            system_prompt=(
                "You generate adversarial memory questions from graph-route evidence. "
                "Return JSON only with question and golden_facts."
            ),
            user_prompt=json.dumps(
                {
                    "case_id": case_id,
                    "route": route,
                    "instruction": (
                        "Generate one tricky but answerable memory question Q. "
                        "The question and gold facts F must be grounded only in the route evidence. "
                        "Do not mention the route, graph, nodes, or edges in the question."
                    ),
                },
                ensure_ascii=False,
            ),
            max_tokens=self.max_output_tokens,
        )
        facts = response.get("golden_facts") or response.get("facts") or []
        if isinstance(facts, str):
            facts = [facts]
        return AttackExample(
            case_id=case_id,
            question=str(response.get("question", "")),
            golden_facts=[str(fact) for fact in facts],
            route=route,
        )

    def generate(self, graph: Dict[str, Any], route: GraphRoute) -> AttackExample:
        return self.generate_from_route(graph.get("case_id", ""), route.to_evidence_dict())

    def _client(self) -> AttackClient:
        if self.client is None:
            self.client = OpenAIChatClient.from_env()
        return self.client
