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
    answer: Any
    route: GraphRoute

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "golden_facts": self.golden_facts,
            "answer": self.answer,
            "route": self.route.to_evidence_dict(),
        }


class FrozenLLMAttacker:
    """Frozen LLM attacker: no training, only prompt-time generation."""

    def __init__(self, client: Optional[AttackClient] = None, max_output_tokens: int = 700):
        self.client = client
        self.max_output_tokens = max_output_tokens

    def generate(self, graph: Dict[str, Any], route: GraphRoute) -> AttackExample:
        response = self._client().complete_json(
            system_prompt=(
                "You generate adversarial memory questions from graph evidence. "
                "Return JSON only with question, golden_facts, and answer."
            ),
            user_prompt=json.dumps(
                    {
                        "case_id": graph.get("case_id"),
                        "route": route.to_evidence_dict(),
                        "instruction": (
                            "Write a tricky but answerable question Q. "
                            "Golden facts F and the answer must be grounded only in the route evidence."
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
            case_id=graph.get("case_id", ""),
            question=response.get("question", ""),
            golden_facts=[str(fact) for fact in facts],
            answer=response.get("answer", ""),
            route=route,
        )

    def _client(self) -> AttackClient:
        if self.client is None:
            self.client = OpenAIChatClient.from_env()
        return self.client
