import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from .refactoring import (
        ADD_ACTION,
        MERGE_ACTION,
        RefactorProposal,
        build_sandbox_memory,
        compute_reward,
        RewardWeights,
    )
    from .memory_evaluator import evaluate_refactor_proposal
    from .retriever import MemoryChunk, MemoryStore
except ImportError:
    from case_graph.refactoring import (
        ADD_ACTION,
        MERGE_ACTION,
        RefactorProposal,
        build_sandbox_memory,
        compute_reward,
        RewardWeights,
    )
    from case_graph.memory_evaluator import evaluate_refactor_proposal
    from case_graph.retriever import MemoryChunk, MemoryStore


SYSTEM_PROMPT = (
    "You are the Memory Refactoring Policy. Generate structured memory chunks "
    "for the selected action. Return JSON only in this format: "
    "{\"chunks\": [{\"memory_id\": \"...\", \"content\": \"...\", "
    "\"facts\": [\"...\"], \"keywords\": [\"...\"], \"summary\": \"...\", "
    "\"source_ids\": [\"...\"]}]}."
)

REWARD_KEYS = (
    "score",
    "format_error",
    "current_correct",
    "regression_accuracy",
    "failed_regression_count",
    "new_chunk_count",
    "current",
    "regression_reward",
    "regression_failure",
    "chunk_count",
    "length",
    "completeness",
    "grounded",
    "duplicate",
    "answer_only",
    "raw_copy",
    "completeness_score",
    "answer_present",
    "relation_overlap",
    "fact_overlap",
    "grounding_score",
    "duplicate_score",
    "raw_copy_ratio",
    "structured_available",
    "structured_complete",
    "subject_coverage",
    "object_coverage",
    "structured_relation_coverage",
    "qualifier_coverage",
    "coverage_gain",
    "critical_coverage_gain",
    "coverage_gain_score",
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


def build_verl_row_online(state: Dict[str, Any], tokenizer=None, index: int = 0) -> Dict[str, Any]:
    """Convert online training state to verl row format.

    Similar to build_verl_row but includes UID for GRPO grouping and episode metadata.
    """
    del tokenizer

    memory_store = _memory_store_from_state(state)
    memory_chunks = [chunk.to_dict() for chunk in memory_store.chunks]

    # Extract only JSON-serializable fields for ground_truth
    ground_truth = {
        "case_id": state.get("case_id", "unknown"),
        "step": state.get("step", state.get("episode", 0)),
        "episode": state.get("episode", 0),
        "uid": state.get("uid", ""),
        "question": state.get("question", ""),
        "answer": state.get("answer", ""),
        "golden_facts": state.get("golden_facts", []),
        # Evaluator-only evidence: build_user_prompt intentionally excludes it.
        "route_evidence": state.get("route_evidence", {}),
        "action": state.get("action", "add"),
        "selected_memory_ids": state.get("selected_memory_ids", []),
        "current_memory": memory_chunks,
        "regression_questions": state.get("regression_questions", []),
        "top_k": state.get("top_k", 5),
        "top_k_points": state.get("top_k_points", 24),
        "retriever_config": state.get("retriever_config", {}),
        "reward_config": state.get("reward_config", {}),
        "coverage_unit_ids": state.get("coverage_unit_ids", []),
        "coverage_route_weight": state.get("coverage_route_weight", 0.0),
        "coverage_pending_weight": state.get("coverage_pending_weight", 0.0),
        "coverage_critical_pending_weight": state.get("coverage_critical_pending_weight", 0.0),
        "coverage_before": state.get("coverage_before", 0.0),
    }

    # Build the prompt messages
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(state)},
    ]

    return {
        "data_source": "memory_refactor_online",
        "prompt": prompt_messages,
        "raw_prompt": prompt_messages,  # verl expects this field
        "index": index,  # verl expects this field
        "reward_model": {"ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
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
    evaluation = evaluate_refactor_proposal(temp_memory, proposal, state)
    reward_config = state.get("reward_config") or {}
    reward = compute_reward(
        proposal,
        evaluation,
        RewardWeights.from_config(reward_config.get("weights", {})),
    )
    current_judge = evaluation.current_test.judge or {}
    structured_complete = bool(
        current_judge.get("complete", False)
        and current_judge.get("structured_complete", current_judge.get("complete", False))
    )
    coverage_route_weight = float(state.get("coverage_route_weight", 0.0) or 0.0)
    coverage_pending_weight = float(state.get("coverage_pending_weight", 0.0) or 0.0)
    critical_pending_weight = float(state.get("coverage_critical_pending_weight", 0.0) or 0.0)
    coverage_gain_score = (
        min(1.0, coverage_pending_weight / coverage_route_weight)
        if coverage_route_weight > 0 and structured_complete and evaluation.current_test.correct
        else 0.0
    )
    critical_gain_score = (
        min(1.0, critical_pending_weight / coverage_route_weight)
        if coverage_route_weight > 0 and structured_complete and evaluation.current_test.correct
        else 0.0
    )
    weights_config = dict(reward_config.get("weights", {}) or {})
    coverage_part = float(weights_config.get("coverage_gain", 1.5)) * coverage_gain_score
    critical_part = float(weights_config.get("critical_coverage_gain", 1.0)) * critical_gain_score
    total_score = reward.reward + coverage_part + critical_part
    return _reward_payload({
        "score": total_score,
        "format_error": 0.0,
        "current_correct": float(evaluation.current_test.correct),
        "regression_accuracy": evaluation.regression_accuracy,
        "failed_regression_count": float(evaluation.failed_regression_count),
        "new_chunk_count": float(len(proposal.new_chunks)),
        "completeness_score": float(current_judge.get("completeness_score", 0.0)),
        "answer_present": float(current_judge.get("answer_present", 0.0)),
        "relation_overlap": float(current_judge.get("relation_overlap", 0.0)),
        "fact_overlap": float(current_judge.get("fact_overlap", 0.0)),
        "grounding_score": float(current_judge.get("grounding_score", 0.0)),
        "duplicate_score": float(current_judge.get("duplicate_score", 0.0)),
        "raw_copy_ratio": float(current_judge.get("raw_copy_ratio", 0.0)),
        "structured_available": float(current_judge.get("structured_available", 0.0)),
        "structured_complete": float(current_judge.get("structured_complete", 0.0)),
        "subject_coverage": float(current_judge.get("subject_coverage", 0.0)),
        "object_coverage": float(current_judge.get("object_coverage", 0.0)),
        "structured_relation_coverage": float(
            current_judge.get("structured_relation_coverage", 0.0)
        ),
        "qualifier_coverage": float(current_judge.get("qualifier_coverage", 0.0)),
        "coverage_gain": coverage_part,
        "critical_coverage_gain": critical_part,
        "coverage_gain_score": coverage_gain_score,
        **reward.parts,
    })


def _proposal_from_response(state: Dict[str, Any], solution_str: str) -> RefactorProposal:
    payload = _parse_json(solution_str)
    chunks = []
    for index, item in enumerate(payload.get("chunks", [])):
        # Handle malformed output where chunks contains strings instead of dicts
        if isinstance(item, str):
            # Wrap string content in a minimal dict structure
            item = {"content": item}
        elif not isinstance(item, dict):
            # Skip non-dict, non-string items
            continue
        chunks.append(MemoryChunk.from_dict(item, fallback_id=f"{state['action']}_{index}"))
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
        "completeness": 0.0,
        "grounded": 0.0,
        "duplicate": 0.0,
        "answer_only": 0.0,
        "raw_copy": 0.0,
    })


def _reward_payload(values: Dict[str, float]) -> Dict[str, float]:
    return {key: float(values.get(key, 0.0)) for key in REWARD_KEYS}


def _memory_store_from_state(state: Dict[str, Any]) -> MemoryStore:
    payload = state.get("current_memory") or state.get("memory") or state.get("memory_store")
    if isinstance(payload, MemoryStore):
        return payload
    if isinstance(payload, list):
        return MemoryStore.from_dicts(payload)
    if payload is None:
        # Return empty memory store if no memory data
        return MemoryStore()
    return MemoryStore.from_dicts(payload.get("memories", payload.get("chunks", [])))


def _parse_json(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start : end + 1])


def _action_requirements(action: str, selected_count: int) -> List[str]:
    if action == ADD_ACTION:
        return [
            "Use only golden_facts to generate exactly one new chunk.",
            "The chunk must contain the subject, relation, object, and useful qualifiers needed to answer the question; do not output an answer-only memory.",
            "Populate facts, keywords, summary, and source_ids when available; these fields are retrieval keys.",
            "Do not output linked_questions; the system will bind the current question automatically.",
        ]
    return [
        "Merge selected_old_chunks with golden_facts.",
        f"Return no more than {selected_count} chunks.",
        "Preserve key facts needed to answer the old linked_questions.",
        "Preserve the subject, relation, object, and useful qualifiers for the current question; do not output an answer-only memory.",
        "Populate facts, keywords, summary, and source_ids when available; these fields are retrieval keys.",
        "Do not output linked_questions; the system will inherit lineage automatically.",
    ]
