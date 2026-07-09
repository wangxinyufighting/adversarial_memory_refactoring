"""Shared memory evaluation utilities for training rewards and sandboxes."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from .baseline import AnswerEquivalenceJudge
from .defense import RetrievedMemoryAnswerAgent
from .refactoring import QuestionTestResult, RefactorProposal, SandboxEvaluation
from .retriever import MemoryChunk, MemoryStore, RetrievalHit, build_memory_retriever


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "user",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "with",
}


def evaluate_refactor_proposal(
    temp_memory: MemoryStore,
    proposal: RefactorProposal,
    state: Dict[str, Any],
    answer_agent: Optional[RetrievedMemoryAnswerAgent] = None,
    judge: Optional[AnswerEquivalenceJudge] = None,
) -> SandboxEvaluation:
    """Evaluate a proposal with the configured retriever and reward checks."""

    top_k = int(state.get("top_k", 5))
    min_score = float(state.get("min_score", 0.0))
    retriever_config = dict(state.get("retriever_config") or {})
    reward_config = dict(state.get("reward_config") or {})
    reward_mode = str(reward_config.get("mode", "semantic_complete")).casefold()
    use_llm_answer = reward_mode in {"evaluation_aligned", "llm", "answer_agent"}

    current = _test_question(
        memory_store=temp_memory,
        question=str(state.get("question", "")),
        gold_answer=str(state.get("answer", "")),
        retriever_config=retriever_config,
        top_k=top_k,
        min_score=min_score,
        answer_agent=answer_agent,
        judge=judge,
        use_llm_answer=use_llm_answer,
    )
    completeness = evaluate_completeness(
        proposal=proposal,
        state=state,
        retrieved_memories=current.retrieved_memories,
        reward_config=reward_config,
    )
    current_judge = dict(current.judge or {})
    current_judge.update(completeness)
    current = QuestionTestResult(
        question=current.question,
        gold_answer=current.gold_answer,
        correct=current.correct,
        retrieved_memories=current.retrieved_memories,
        answer_result=current.answer_result,
        judge=current_judge,
    )

    regressions = [
        _test_question(
            memory_store=temp_memory,
            question=str(item.get("question", "")),
            gold_answer=str(item.get("answer", "")),
            retriever_config=retriever_config,
            top_k=top_k,
            min_score=min_score,
            answer_agent=answer_agent,
            judge=judge,
            use_llm_answer=use_llm_answer,
        )
        for item in state.get("regression_questions", [])
    ]
    return SandboxEvaluation(current_test=current, regression_tests=regressions)


def evaluate_completeness(
    proposal: RefactorProposal,
    state: Dict[str, Any],
    retrieved_memories: List[RetrievalHit],
    reward_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Heuristic relation/fact completeness check for the current question."""

    reward_config = dict(reward_config or {})
    question = str(state.get("question", ""))
    answer = str(state.get("answer", ""))
    golden_texts = _golden_fact_texts(state.get("golden_facts", []))
    generated_text = "\n".join(chunk.content for chunk in proposal.new_chunks)
    retrieved_text = "\n".join(hit.content for hit in retrieved_memories)
    evidence_text = "\n".join([retrieved_text, generated_text])

    answer_present = _answer_in_text(answer, evidence_text)
    answer_tokens = set(_significant_tokens(answer, keep_stopwords=True))
    question_tokens = set(_significant_tokens(question))
    gold_tokens = set(_significant_tokens(" ".join(golden_texts)))
    generated_tokens = set(_significant_tokens(generated_text))
    reference_tokens = set(_significant_tokens(_reference_text_for_grounding(state)))

    relation_tokens = (question_tokens & gold_tokens) - answer_tokens
    if not relation_tokens:
        relation_tokens = question_tokens - answer_tokens
    fact_tokens = gold_tokens - answer_tokens

    relation_overlap = _coverage(relation_tokens, generated_tokens)
    fact_overlap = _bounded_coverage(fact_tokens, generated_tokens)
    grounding_score = _jaccard(generated_tokens, reference_tokens) if generated_tokens else 0.0
    duplicate_score = _duplicate_score(proposal, state)
    raw_copy_ratio = _raw_copy_ratio(proposal, golden_texts)

    min_relation_overlap = float(reward_config.get("min_relation_overlap", 0.25))
    min_fact_overlap = float(reward_config.get("min_fact_overlap", 0.25))
    min_non_answer_tokens = int(reward_config.get("min_non_answer_tokens", 3))
    non_answer_tokens = generated_tokens - answer_tokens
    answer_only = answer_present and (
        len(non_answer_tokens) < min_non_answer_tokens or relation_overlap < 0.15
    )
    complete = bool(
        answer_present
        and not answer_only
        and relation_overlap >= min_relation_overlap
        and fact_overlap >= min_fact_overlap
    )
    completeness_score = (
        0.4 * float(answer_present)
        + 0.3 * relation_overlap
        + 0.3 * fact_overlap
    )
    return {
        "completeness_method": "semantic_relation_completeness",
        "complete": complete,
        "completeness_score": completeness_score,
        "answer_present": answer_present,
        "answer_only": answer_only,
        "relation_overlap": relation_overlap,
        "fact_overlap": fact_overlap,
        "grounding_score": grounding_score,
        "duplicate_score": duplicate_score,
        "raw_copy_ratio": raw_copy_ratio,
        "missing_relation_terms": sorted(relation_tokens - generated_tokens),
    }


def _test_question(
    memory_store: MemoryStore,
    question: str,
    gold_answer: str,
    retriever_config: Dict[str, Any],
    top_k: int,
    min_score: float,
    answer_agent: Optional[RetrievedMemoryAnswerAgent],
    judge: Optional[AnswerEquivalenceJudge],
    use_llm_answer: bool,
) -> QuestionTestResult:
    retriever = build_memory_retriever(retriever_config, memory_store)
    hits = retriever.retrieve(question, top_k=top_k, min_score=min_score)
    if use_llm_answer:
        if answer_agent is None:
            answer_agent = RetrievedMemoryAnswerAgent()
        if judge is None:
            judge = AnswerEquivalenceJudge()
        answer_result = answer_agent.answer(question, hits)
        judge_result = judge.judge(
            question=question,
            gold_answer=gold_answer,
            candidate_answer=str(answer_result.get("answer", "")),
        )
        correct = bool(judge_result.get("correct", False))
    else:
        evidence = "\n".join(hit.content for hit in hits)
        correct = _answer_in_text(gold_answer, evidence)
        answer_result = {
            "answer": gold_answer if correct else "UNKNOWN",
            "reason": "semantic reward uses retrieved-memory answer presence before completeness gating",
            "evidence_memory_ids": [hit.memory_id for hit in hits],
        }
        judge_result = {"correct": correct, "method": "semantic_answer_presence"}
    return QuestionTestResult(
        question=question,
        gold_answer=gold_answer,
        correct=correct,
        retrieved_memories=hits,
        answer_result=answer_result,
        judge=judge_result,
    )


def _golden_fact_texts(golden_facts: Any) -> List[str]:
    texts: List[str] = []
    for fact in golden_facts or []:
        if isinstance(fact, str):
            texts.append(fact)
        elif isinstance(fact, dict):
            preferred = [
                fact.get("text"),
                fact.get("content"),
                fact.get("description"),
                fact.get("source"),
                fact.get("relation"),
                fact.get("target"),
            ]
            text = " ".join(str(item) for item in preferred if item)
            if not text:
                text = " ".join(str(value) for value in fact.values() if isinstance(value, (str, int, float)))
            texts.append(text)
    return [text for text in texts if str(text).strip()]


def _reference_text_for_grounding(state: Dict[str, Any]) -> str:
    texts = _golden_fact_texts(state.get("golden_facts", []))
    selected = set(str(item) for item in state.get("selected_memory_ids", []))
    memory = state.get("current_memory") or []
    if isinstance(memory, dict):
        memory = memory.get("memories", memory.get("chunks", []))
    for item in memory or []:
        chunk = item if isinstance(item, MemoryChunk) else MemoryChunk.from_dict(item)
        if not selected or chunk.memory_id in selected:
            texts.append(chunk.content)
    return "\n".join(texts)


def _duplicate_score(proposal: RefactorProposal, state: Dict[str, Any]) -> float:
    removed = set(str(item) for item in proposal.remove_memory_ids)
    memory = state.get("current_memory") or []
    if isinstance(memory, dict):
        memory = memory.get("memories", memory.get("chunks", []))
    old_chunks = []
    for item in memory or []:
        chunk = item if isinstance(item, MemoryChunk) else MemoryChunk.from_dict(item)
        if chunk.memory_id not in removed:
            old_chunks.append(chunk)
    max_score = 0.0
    for new_chunk in proposal.new_chunks:
        new_tokens = set(_significant_tokens(new_chunk.content))
        for old_chunk in old_chunks:
            max_score = max(max_score, _jaccard(new_tokens, set(_significant_tokens(old_chunk.content))))
    return max_score


def _raw_copy_ratio(proposal: RefactorProposal, golden_texts: List[str]) -> float:
    max_ratio = 0.0
    golden_token_sets = [set(_significant_tokens(text, keep_stopwords=True)) for text in golden_texts]
    for chunk in proposal.new_chunks:
        chunk_tokens = set(_significant_tokens(chunk.content, keep_stopwords=True))
        if len(chunk_tokens) < 20:
            continue
        for gold_tokens in golden_token_sets:
            if not gold_tokens:
                continue
            max_ratio = max(max_ratio, len(chunk_tokens & gold_tokens) / max(1, len(chunk_tokens)))
    return max_ratio


def _answer_in_text(answer: str, text: str) -> bool:
    answer_norm = _normalize(answer)
    text_norm = _normalize(text)
    return bool(answer_norm and answer_norm in text_norm)


def _normalize(text: str) -> str:
    tokens = _tokens(text)
    tokens = [token for token in tokens if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def _significant_tokens(text: str, keep_stopwords: bool = False) -> List[str]:
    tokens = _tokens(text)
    if keep_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", str(text or "").casefold())


def _coverage(required: Iterable[str], actual: Iterable[str]) -> float:
    required_set = set(required)
    if not required_set:
        return 1.0
    actual_set = set(actual)
    return len(required_set & actual_set) / len(required_set)


def _bounded_coverage(required: Iterable[str], actual: Iterable[str]) -> float:
    required_set = set(required)
    if not required_set:
        return 1.0
    actual_set = set(actual)
    denominator = max(1, min(6, len(required_set)))
    return min(1.0, len(required_set & actual_set) / denominator)


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)
