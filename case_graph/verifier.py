import json
import re
from typing import Any, Dict, List, Optional, Protocol

from .llm import OpenAIChatClient


TEMPORAL_ORDER_CUES = (
    "before",
    "after",
    "prior to",
    "following",
    "subsequent to",
    "earlier than",
    "later than",
)


class VerificationClient(Protocol):
    def complete_json(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        ...


class GoldenFactVerifier:
    """Check whether an attack answer is supported by raw-session gold facts."""

    def __init__(self, client: Optional[VerificationClient] = None, max_output_tokens: int = 300):
        self.client = client
        self.max_output_tokens = max_output_tokens

    def verify(
        self,
        question: str,
        answer: str,
        golden_facts: List[Dict[str, Any]],
        route: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not question.strip() or not answer.strip() or not golden_facts:
            return {
                "supported": False,
                "unambiguous": False,
                "temporal_supported": False,
                "evidence_session_ids": [],
                "reason": "Missing question, answer, or golden facts.",
            }

        response = self._client().complete_json(
            system_prompt=(
                "You verify whether an answer to a memory question is fully supported by raw session evidence. "
                "Return JSON only with supported, unambiguous, evidence_session_ids, and reason."
            ),
            user_prompt=json.dumps(
                {
                    "question": question,
                    "answer": answer,
                    "golden_facts": golden_facts,
                    "instruction": (
                        "Set unambiguous=true only if the question has exactly one plausible referent in the raw facts. "
                        "If the raw facts contain multiple similar candidates, such as multiple players, teams, sports, "
                        "organizations, locations, or events, the question must include enough qualifiers to identify one. "
                        "Set supported=true only if unambiguous=true and the answer is directly entailed by the raw session text. "
                        "Reject unsupported temporal ordering. If the question uses words like before, after, prior to, "
                        "following, earlier than, or later than, that ordering must be directly supported by the raw facts. "
                        "Use evidence_session_ids to list the supporting sessions. "
                        "If the answer is incomplete, contradicted, not grounded, temporally unsupported, "
                        "or the question is ambiguous, set supported=false."
                    ),
                },
                ensure_ascii=False,
            ),
            max_tokens=self.max_output_tokens,
        )
        unambiguous = bool(response.get("unambiguous", response.get("supported", False)))
        temporal_supported = _temporal_order_supported(question, route)
        reason = str(response.get("reason", ""))
        if not temporal_supported:
            reason = (
                "The question introduces temporal ordering such as before/after, "
                "but the route evidence does not explicitly support that ordering."
            )
        return {
            "supported": bool(response.get("supported", False)) and unambiguous and temporal_supported,
            "unambiguous": unambiguous,
            "temporal_supported": temporal_supported,
            "evidence_session_ids": [str(item) for item in response.get("evidence_session_ids", [])],
            "reason": reason,
        }

    def _client(self) -> VerificationClient:
        if self.client is None:
            self.client = OpenAIChatClient.from_env()
        return self.client


def _temporal_order_supported(question: str, route: Optional[Dict[str, Any]]) -> bool:
    question_text = question.casefold()
    cues = [cue for cue in TEMPORAL_ORDER_CUES if _has_phrase(question_text, cue)]
    if not cues:
        return True
    route_text = json.dumps(route or {}, ensure_ascii=False).casefold()
    return any(_has_phrase(route_text, cue) for cue in cues)


def _has_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None
