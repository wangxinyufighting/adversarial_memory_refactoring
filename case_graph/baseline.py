import copy
import json
import re
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .llm import OpenAIChatClient


class JsonClient(Protocol):
    def complete_json(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        ...


class GoldenFactAnswerAgent:
    """Answer Q using only the raw-session golden facts F."""

    def __init__(self, client: Optional[JsonClient] = None, max_output_tokens: int = 200):
        self.client = client
        self.max_output_tokens = max_output_tokens

    def answer(self, question: str, golden_facts: List[Dict[str, Any]]) -> Dict[str, str]:
        if not question.strip() or not golden_facts:
            return {
                "answer": "UNKNOWN",
                "reason": "Missing question or golden facts.",
            }
        response = self._client().complete_json(
            system_prompt=(
                "You answer memory questions using only the provided raw sessions. "
                "Return JSON only with answer and reason."
            ),
            user_prompt=json.dumps(
                {
                    "question": question,
                    "golden_facts": golden_facts,
                    "instruction": (
                        "Answer concisely using only the golden_facts. "
                        "If the answer is not explicitly supported, answer UNKNOWN."
                    ),
                },
                ensure_ascii=False,
            ),
            max_tokens=self.max_output_tokens,
        )
        return {
            "answer": str(response.get("answer", "")),
            "reason": str(response.get("reason", "")),
        }

    def _client(self) -> JsonClient:
        if self.client is None:
            self.client = OpenAIChatClient.from_env()
        return self.client


class AnswerEquivalenceJudge:
    """Judge whether the answer agent output matches the attack's gold answer."""

    def __init__(
        self,
        client: Optional[JsonClient] = None,
        max_output_tokens: int = 200,
        use_llm: bool = True,
    ):
        self.client = client
        self.max_output_tokens = max_output_tokens
        self.use_llm = use_llm

    def judge(self, question: str, gold_answer: str, candidate_answer: str) -> Dict[str, Any]:
        if _answers_match(gold_answer, candidate_answer):
            return {
                "correct": True,
                "method": "string_match",
                "reason": "The candidate answer matches the gold answer after normalization.",
            }
        if _is_unknown_answer(candidate_answer):
            return {
                "correct": False,
                "method": "string_match",
                "reason": "The candidate answer is UNKNOWN or empty.",
            }
        if not self.use_llm:
            return {
                "correct": False,
                "method": "string_match",
                "reason": "The candidate answer does not match the gold answer after normalization.",
            }
        response = self._client().complete_json(
            system_prompt=(
                "You judge whether a candidate answer is equivalent to the gold answer for a memory question. "
                "Return JSON only with correct and reason."
            ),
            user_prompt=json.dumps(
                {
                    "question": question,
                    "gold_answer": gold_answer,
                    "candidate_answer": candidate_answer,
                    "instruction": (
                        "Set correct=true only if the candidate gives the same answer as the gold answer. "
                        "Ignore casing, articles, and harmless wording differences. "
                        "Set correct=false if the candidate is wrong, unknown, contradictory, too broad, or too narrow."
                    ),
                },
                ensure_ascii=False,
            ),
            max_tokens=self.max_output_tokens,
        )
        return {
            "correct": bool(response.get("correct", False)),
            "method": "llm_judge",
            "reason": str(response.get("reason", "")),
        }

    def _client(self) -> JsonClient:
        if self.client is None:
            self.client = OpenAIChatClient.from_env()
        return self.client


def run_baseline_sanity_test(
    attacks: List[Dict[str, Any]],
    answer_agent: GoldenFactAnswerAgent,
    judge: AnswerEquivalenceJudge,
    keep_failed: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    passed = []
    discarded = []
    for attack in attacks:
        tested_attack = copy.deepcopy(attack)
        answer_result = answer_agent.answer(
            question=str(tested_attack.get("question", "")),
            golden_facts=tested_attack.get("golden_facts", []),
        )
        judge_result = judge.judge(
            question=str(tested_attack.get("question", "")),
            gold_answer=str(tested_attack.get("answer", "")),
            candidate_answer=answer_result["answer"],
        )
        baseline = {
            "correct": bool(judge_result["correct"]),
            "answer_agent_answer": answer_result["answer"],
            "answer_agent_reason": answer_result["reason"],
            "judge": judge_result,
        }
        tested_attack["baseline_sanity"] = baseline
        if baseline["correct"]:
            passed.append(tested_attack)
        else:
            discarded.append(tested_attack)
            if keep_failed:
                passed.append(tested_attack)
    return passed, discarded


def _answers_match(gold_answer: str, candidate_answer: str) -> bool:
    gold = _normalize_answer(gold_answer)
    candidate = _normalize_answer(candidate_answer)
    if not gold or not candidate or candidate == "unknown":
        return False
    if gold == candidate:
        return True
    return len(gold) >= 4 and (gold in candidate or candidate in gold)


def _is_unknown_answer(value: str) -> bool:
    normalized = _normalize_answer(value)
    if not normalized:
        return True
    return normalized in {
        "unknown",
        "unk",
        "n a",
        "na",
        "none",
        "not known",
        "i don t know",
        "cannot determine",
        "not enough information",
    }


def _normalize_answer(value: str) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [token for token in text.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)
