"""Answer agents and judges for memory QA evaluation."""

import json
from typing import Any, Dict, Optional, Protocol, Sequence

from case_graph.retriever import RetrievalHit


class JsonClient(Protocol):
    def complete_json(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        ...


class TextClient(Protocol):
    def complete_text(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        ...


class LongMemEvalMemoryAnswerAgent:
    """Answer a LongMemEval target using only Contriever-retrieved memories."""

    def __init__(self, client: JsonClient, max_output_tokens: int = 256):
        self.client = client
        self.max_output_tokens = int(max_output_tokens)

    def answer(
        self,
        question: str,
        retrieved_memories: Sequence[RetrievalHit],
        *,
        question_date: str = "",
        question_type: str = "",
    ) -> Dict[str, Any]:
        if not question.strip() or not retrieved_memories:
            return {
                "answer": "UNKNOWN",
                "reason": "Missing question or retrieved memories.",
                "evidence_memory_ids": [],
            }

        response = self.client.complete_json(
            system_prompt=(
                "You are a memory question-answering agent. Use only the retrieved "
                "memories, never outside knowledge. Combine evidence across memories "
                "when the question asks for a count, comparison, update, or temporal "
                "reasoning. Return one JSON object with answer, reason, and "
                "evidence_memory_ids."
            ),
            user_prompt=json.dumps(
                {
                    "question": question,
                    "question_date": question_date,
                    "question_type": question_type,
                    "retrieved_memories": [item.to_dict() for item in retrieved_memories],
                    "instruction": (
                        "Answer concisely from the memory content. Treat question_date "
                        "as the reference date for relative time expressions. For counts, "
                        "count distinct supported events rather than mentions. If the "
                        "memories do not support an answer, answer UNKNOWN. Cite only "
                        "retrieved memory_id values that directly support the answer. "
                        "Metadata IDs are provenance labels, not facts."
                    ),
                },
                ensure_ascii=False,
            ),
            max_tokens=self.max_output_tokens,
        )
        evidence_ids = response.get("evidence_memory_ids", [])
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
        return {
            "answer": str(response.get("answer", "")),
            "reason": str(response.get("reason", "")),
            "evidence_memory_ids": [str(item) for item in evidence_ids if str(item)],
        }


class LongMemEvalAnswerJudge:
    """Question-type-aware judge matching the released LongMemEval protocol."""

    def __init__(
        self,
        client: TextClient,
        max_output_tokens: int = 10,
        provider: str = "llm",
    ):
        self.client = client
        self.max_output_tokens = int(max_output_tokens)
        self.provider = str(provider or "llm")

    def judge_longmemeval(
        self,
        *,
        case_id: str,
        question_type: str,
        question: str,
        gold_answer: str,
        candidate_answer: str,
    ) -> Dict[str, Any]:
        prompt = _longmemeval_judge_prompt(
            case_id=case_id,
            question_type=question_type,
            question=question,
            gold_answer=gold_answer,
            candidate_answer=candidate_answer,
        )
        response = self.client.complete_text(
            system_prompt=None,
            user_prompt=prompt,
            max_tokens=self.max_output_tokens,
        ).strip()
        correct = "yes" in response.casefold()
        return {
            "correct": correct,
            "method": f"longmemeval_{self.provider}_judge",
            "reason": response,
            "question_type": question_type,
            "abstention": "_abs" in case_id,
        }


def _longmemeval_judge_prompt(
    *,
    case_id: str,
    question_type: str,
    question: str,
    gold_answer: str,
    candidate_answer: str,
) -> str:
    if "_abs" in case_id:
        criterion = (
            "The question is unanswerable. Mark yes only when the model recognizes "
            "that the requested information is unavailable or incomplete. It may "
            "mention available information as long as it does not invent the answer."
        )
        answer_label = "Reference explanation"
    elif question_type == "temporal-reasoning":
        criterion = (
            "Mark yes when the response contains or is equivalent to the complete "
            "reference answer. A response containing only part of the required answer "
            "is incorrect. For requested day, week, or month durations, tolerate an "
            "off-by-one numerical error."
        )
        answer_label = "Correct answer"
    elif question_type == "knowledge-update":
        criterion = (
            "Mark yes when the response contains the required latest answer. Mentioning "
            "older information as additional context is allowed when the updated answer "
            "is also clearly present."
        )
        answer_label = "Correct answer"
    elif question_type == "single-session-preference":
        criterion = (
            "Treat the reference as a personalization rubric. Mark yes when the response "
            "correctly recalls and uses the user's personal information; it need not "
            "cover every rubric point."
        )
        answer_label = "Personalization rubric"
    else:
        criterion = (
            "Mark yes when the response contains or is equivalent to the complete "
            "reference answer, including valid intermediate reasoning. A response that "
            "contains only a subset of the required information is incorrect."
        )
        answer_label = "Correct answer"

    return (
        "Evaluate the model response under the LongMemEval protocol. "
        f"{criterion}\n\n"
        f"Question: {question}\n\n"
        f"{answer_label}: {gold_answer}\n\n"
        f"Model response: {candidate_answer}\n\n"
        "Is the model response correct? Answer yes or no only."
    )
