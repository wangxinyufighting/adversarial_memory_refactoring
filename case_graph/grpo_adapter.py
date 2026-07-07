import json
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from .refactoring import (
        ADD_ACTION,
        MERGE_ACTION,
        RefactorProposal,
        SandboxEvaluation,
        QuestionTestResult,
        build_sandbox_memory,
        compute_reward,
    )
    from .retriever import FrozenBM25Retriever, MemoryChunk, MemoryStore
except ImportError:
    from case_graph.refactoring import (
        ADD_ACTION,
        MERGE_ACTION,
        RefactorProposal,
        SandboxEvaluation,
        QuestionTestResult,
        build_sandbox_memory,
        compute_reward,
    )
    from case_graph.retriever import FrozenBM25Retriever, MemoryChunk, MemoryStore


SYSTEM_PROMPT = (
    "You are the Memory Refactoring Policy. Generate structured memory chunks "
    "for the selected action. Return JSON only in this format: "
    "{\"chunks\": [{\"memory_id\": \"...\", \"content\": \"...\"}]}."
)

REWARD_KEYS = (
    "score",
    "format_error",
    "current_correct",
    "regression_accuracy",
    "failed_regression_count",
    "new_chunk_count",
    "current",
    "regression_failure",
    "chunk_count",
    "length",
)


def build_verl_row(state: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """把一个记忆重构状态转成 verl 的 RLHF parquet 行。"""

    return {
        "data_source": "memory_refactor",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(state)},
        ],
        "reward_model": {"ground_truth": json.dumps(state, ensure_ascii=False)},
        "extra_info": {"index": index, "case_id": state.get("case_id", index)},
    }


def build_verl_row_online(state: Dict[str, Any], tokenizer=None) -> Dict[str, Any]:
    """Convert online training state to verl row format.

    Similar to build_verl_row but includes UID for GRPO grouping and episode metadata.
    """
    return {
        "data_source": "memory_refactor_online",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(state)},
        ],
        "reward_model": {"ground_truth": json.dumps(state, ensure_ascii=False)},
        "extra_info": {
            "uid": state.get("uid", f"{state.get('case_id', 'unknown')}_ep{state.get('episode', 0)}"),
            "case_id": state.get("case_id", "unknown"),
            "episode": state.get("episode", 0),
        },
    }


def build_user_prompt(state: Dict[str, Any]) -> str:
    memory_store = _memory_store_from_state(state)
    selected_ids = state.get("selected_memory_ids", [])
    selected_chunks = [
        chunk.to_dict() for chunk in memory_store.chunks if chunk.memory_id in set(selected_ids)
    ]
    payload = {
        "action": state["action"],
        "question": state["question"],
        "golden_facts": state.get("golden_facts", []),
        "selected_old_chunks": selected_chunks,
        "requirements": _action_requirements(state["action"], len(selected_chunks)),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def write_verl_parquet(states: List[Dict[str, Any]], output_path: str | Path) -> None:
    """写出 verl 可直接读取的 parquet；pandas 只在数据准备时需要。"""

    import pandas as pd

    rows = [build_verl_row(state, index=index) for index, state in enumerate(states)]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | Dict[str, Any],
    extra_info: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """verl 自定义 reward：解析模型输出，在沙盒里执行，再计算 reward。"""

    state = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    try:
        proposal = _proposal_from_response(state, solution_str)
    except (json.JSONDecodeError, ValueError):
        return _bad_score()
    temp_memory = build_sandbox_memory(
        _memory_store_from_state(state),
        proposal,
        current_question=state["question"],
    )
    evaluation = _evaluate_temp_memory(temp_memory, state)
    reward = compute_reward(proposal, evaluation)
    return _reward_payload({
        "score": reward.reward,
        "format_error": 0.0,
        "current_correct": float(evaluation.current_test.correct),
        "regression_accuracy": evaluation.regression_accuracy,
        "failed_regression_count": float(evaluation.failed_regression_count),
        "new_chunk_count": float(len(proposal.new_chunks)),
        **reward.parts,
    })


def _proposal_from_response(state: Dict[str, Any], solution_str: str) -> RefactorProposal:
    payload = _parse_json(solution_str)
    chunks = [
        MemoryChunk.from_dict(item, fallback_id=f"{state['action']}_{index}")
        for index, item in enumerate(payload.get("chunks", []))
    ]
    if not chunks:
        raise ValueError("policy response contains no chunks")
    if state["action"] == ADD_ACTION:
        chunks = chunks[:1]
        remove_ids: List[str] = []
    else:
        selected_ids = [str(item) for item in state.get("selected_memory_ids", [])]
        chunks = chunks[: max(1, len(selected_ids))]
        remove_ids = selected_ids
    return RefactorProposal(
        action=state["action"],
        new_chunks=chunks,
        remove_memory_ids=remove_ids,
        metadata={"source": "verl_grpo"},
    )


def _bad_score() -> Dict[str, float]:
    return _reward_payload({
        "score": -3.0,
        "format_error": 1.0,
        "current_correct": 0.0,
        "regression_accuracy": 0.0,
        "failed_regression_count": 0.0,
        "new_chunk_count": 0.0,
        "current": -3.0,
        "regression_failure": 0.0,
        "chunk_count": 0.0,
        "length": 0.0,
    })


def _reward_payload(values: Dict[str, float]) -> Dict[str, float]:
    return {key: float(values.get(key, 0.0)) for key in REWARD_KEYS}


def _evaluate_temp_memory(memory_store: MemoryStore, state: Dict[str, Any]) -> SandboxEvaluation:
    top_k = int(state.get("top_k", 5))
    current = _presence_test(memory_store, state["question"], state["answer"], top_k)
    regressions = [
        _presence_test(memory_store, item["question"], item["answer"], top_k)
        for item in state.get("regression_questions", [])
    ]
    return SandboxEvaluation(current_test=current, regression_tests=regressions)


def _presence_test(
    memory_store: MemoryStore,
    question: str,
    answer: str,
    top_k: int,
) -> QuestionTestResult:
    hits = FrozenBM25Retriever(memory_store).retrieve(question, top_k=top_k)
    evidence = "\n".join(hit.content for hit in hits)
    correct = _answer_in_text(answer, evidence)
    return QuestionTestResult(
        question=question,
        gold_answer=answer,
        correct=correct,
        retrieved_memories=hits,
        answer_result={
            "answer": answer if correct else "UNKNOWN",
            "reason": "local answer-presence reward",
            "evidence_memory_ids": [hit.memory_id for hit in hits],
        },
        judge={"correct": correct, "method": "answer_presence"},
    )


def _memory_store_from_state(state: Dict[str, Any]) -> MemoryStore:
    payload = state.get("current_memory") or state.get("memory") or state.get("memory_store")
    if isinstance(payload, MemoryStore):
        return payload
    if isinstance(payload, list):
        return MemoryStore.from_dicts(payload)
    return MemoryStore.from_dicts(payload.get("memories", payload.get("chunks", [])))


def _parse_json(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start : end + 1])


def _answer_in_text(answer: str, text: str) -> bool:
    answer_norm = _normalize(answer)
    text_norm = _normalize(text)
    return bool(answer_norm and answer_norm in text_norm)


def _normalize(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", str(text or "").casefold())
    tokens = [token for token in tokens if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def _action_requirements(action: str, selected_count: int) -> List[str]:
    if action == ADD_ACTION:
        return [
            "Use only golden_facts to generate exactly one new chunk.",
            "Do not output linked_questions; the system will bind the current question automatically.",
        ]
    return [
        "Merge selected_old_chunks with golden_facts.",
        f"Return no more than {selected_count} chunks.",
        "Preserve key facts needed to answer the old linked_questions.",
        "Do not output linked_questions; the system will inherit lineage automatically.",
    ]
