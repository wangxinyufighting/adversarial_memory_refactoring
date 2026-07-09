"""Construct evaluation memories with a trained defender policy."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from case_graph.attacker import FrozenLLMAttacker
from case_graph.baseline import AnswerEquivalenceJudge
from case_graph.defense import RetrievedMemoryAnswerAgent, SuccessPool
from case_graph.grpo_adapter import SYSTEM_PROMPT, build_user_prompt
from case_graph.llm import OpenAIChatClient
from case_graph.pipeline import AlgorithmConfig, prepare_refactor_state
from case_graph.refactoring import (
    ADD_ACTION,
    MERGE_ACTION,
    HighPriorityBuffer,
    RefactorProposal,
    RegressionQuestion,
    run_sandbox_refactor,
    settle_grpo_rollouts,
)
from case_graph.retriever import MemoryChunk, MemoryStore
from case_graph.routing import RandomWalkRoutingPolicy

logger = logging.getLogger(__name__)


class DefenderClient(Protocol):
    def complete_json(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        ...


@dataclass(frozen=True)
class MemoryConstructionConfig:
    """Runtime settings for target-free evaluation memory construction."""

    tau: float = 0.7
    top_k: int = 5
    min_score: float = 0.0
    regression_sample_size: int = 3
    episodes_per_case: int = 100
    proposal_count: int = 1
    commit_threshold: float = 0.0
    seed: int = 42
    routing_max_steps: int = 3
    routing_min_nodes: int = 1
    routing_attempts: int = 8
    defender_max_output_tokens: int = 800
    attacker_max_output_tokens: int = 700
    memory_save_interval: int = 0
    max_attack_failures: int = 20
    exp_name: str = "eval_memory_construction"

    def algorithm_config(self, memory_archive_dir: Optional[str] = None) -> AlgorithmConfig:
        return AlgorithmConfig(
            tau=self.tau,
            top_k=self.top_k,
            min_score=self.min_score,
            regression_sample_size=self.regression_sample_size,
            proposal_count=self.proposal_count,
            seed=self.seed,
            commit_threshold=self.commit_threshold,
            memory_archive_dir=memory_archive_dir,
            exp_name=self.exp_name,
        )


@dataclass(frozen=True)
class ConstructionStepTrace:
    case_id: str
    episode: int
    status: str
    question: str = ""
    answer: str = ""
    route: Dict[str, Any] = field(default_factory=dict)
    initial_defense: Dict[str, Any] = field(default_factory=dict)
    decision: Optional[Dict[str, Any]] = None
    settlement: Optional[Dict[str, Any]] = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "episode": self.episode,
            "status": self.status,
            "question": self.question,
            "answer": self.answer,
            "route": self.route,
            "initial_defense": self.initial_defense,
            "decision": self.decision,
            "settlement": self.settlement,
            "error": self.error,
        }


@dataclass(frozen=True)
class CaseConstructionResult:
    case_id: str
    memory_store: MemoryStore
    success_pool: SuccessPool
    high_priority_buffer: HighPriorityBuffer
    traces: List[ConstructionStepTrace]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "summary": summarize_case_traces(self.traces, self.memory_store),
            "final_memory": {"memories": [chunk.to_dict() for chunk in self.memory_store.chunks]},
            "success_pool": self.success_pool.to_dict(),
            "high_priority_buffer": self.high_priority_buffer.to_dict(),
            "traces": [trace.to_dict() for trace in self.traces],
        }


class DefenderCheckpointPolicy:
    """Inference-only wrapper around the trained defender checkpoint server."""

    def __init__(
        self,
        client: DefenderClient,
        max_output_tokens: int = 800,
    ):
        self.client = client
        self.max_output_tokens = max_output_tokens

    def propose_from_state(self, state: Dict[str, Any]) -> RefactorProposal:
        response = self.client.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(state),
            max_tokens=self.max_output_tokens,
        )
        chunks = [
            MemoryChunk.from_dict(item, fallback_id=f"{state['action']}_{index}")
            for index, item in enumerate(response.get("chunks", []))
            if isinstance(item, dict)
        ]
        if not chunks:
            raise ValueError(f"Defender response contains no chunks: {response}")

        source_ids = _source_ids_from_state(state)
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata.setdefault("source", "defender_checkpoint_eval_construction")
            metadata.setdefault("case_id", state.get("case_id", ""))
            metadata.setdefault("construction_question", state.get("question", ""))
            if source_ids:
                metadata.setdefault("source_ids", source_ids)
            chunk.metadata = metadata

        if state["action"] == ADD_ACTION:
            chunks = chunks[:1]
            remove_ids: List[str] = []
        elif state["action"] == MERGE_ACTION:
            selected_ids = [str(item) for item in state.get("selected_memory_ids", [])]
            chunks = chunks[: max(1, len(selected_ids))]
            remove_ids = selected_ids
        else:
            remove_ids = []

        return RefactorProposal(
            action=state["action"],
            new_chunks=chunks,
            remove_memory_ids=remove_ids,
            metadata={"policy": "defender_checkpoint", "mode": "eval_memory_construction"},
        )


class EvaluationMemoryConstructor:
    """Build memories for held-out cases without using target metadata."""

    def __init__(
        self,
        config: MemoryConstructionConfig,
        defender_policy: DefenderCheckpointPolicy,
        attacker: FrozenLLMAttacker,
        answer_agent: RetrievedMemoryAnswerAgent,
        judge: AnswerEquivalenceJudge,
    ):
        self.config = config
        self.defender_policy = defender_policy
        self.attacker = attacker
        self.answer_agent = answer_agent
        self.judge = judge

    def construct_case(
        self,
        graph: Dict[str, Any],
        initial_memory: Optional[MemoryStore] = None,
        memory_archive_dir: Optional[str] = None,
    ) -> CaseConstructionResult:
        safe_graph = strip_target_metadata(graph)
        case_id = str(safe_graph.get("case_id", "unknown"))
        memory_store = initial_memory or MemoryStore()
        success_pool = SuccessPool()
        high_priority_buffer = HighPriorityBuffer()
        routing_policy = RandomWalkRoutingPolicy(
            seed=self.config.seed,
            max_steps=self.config.routing_max_steps,
            min_nodes=self.config.routing_min_nodes,
            attempts=self.config.routing_attempts,
        )
        algorithm_config = self.config.algorithm_config(memory_archive_dir=memory_archive_dir)
        traces: List[ConstructionStepTrace] = []
        attack_failures = 0

        for episode in range(max(0, self.config.episodes_per_case)):
            seed = self.config.seed + episode * 1009
            try:
                route = routing_policy.select_route(safe_graph, seed=seed)
                attack = self.attacker.generate(safe_graph, route)
                attack_dict = attack.to_dict()
                attack_dict["case_id"] = case_id
            except Exception as exc:
                attack_failures += 1
                traces.append(
                    ConstructionStepTrace(
                        case_id=case_id,
                        episode=episode,
                        status="attack_generation_failed",
                        error=str(exc),
                    )
                )
                if attack_failures >= self.config.max_attack_failures:
                    logger.warning("Stopping case %s after %d attack failures", case_id, attack_failures)
                    break
                continue

            try:
                state = prepare_refactor_state(
                    attack=attack_dict,
                    memory_store=memory_store,
                    success_pool=success_pool,
                    answer_agent=self.answer_agent,
                    judge=self.judge,
                    config=algorithm_config,
                    step=episode,
                )
            except ValueError as exc:
                traces.append(
                    ConstructionStepTrace(
                        case_id=case_id,
                        episode=episode,
                        status="invalid_probe",
                        question=str(attack_dict.get("question", "")),
                        answer=str(attack_dict.get("answer", "")),
                        route=attack_dict.get("route", {}),
                        error=str(exc),
                    )
                )
                continue

            if state is None:
                traces.append(
                    ConstructionStepTrace(
                        case_id=case_id,
                        episode=episode,
                        status="initial_defense_success",
                        question=str(attack_dict.get("question", "")),
                        answer=str(attack_dict.get("answer", "")),
                        route=attack_dict.get("route", {}),
                    )
                )
                continue

            sandbox_results = []
            regression_questions = [
                RegressionQuestion(
                    question=item["question"],
                    answer=item["answer"],
                    source_memory_ids=item.get("source_memory_ids", []),
                )
                for item in state.get("regression_questions", [])
            ]
            for rollout_index in range(max(1, self.config.proposal_count)):
                try:
                    proposal = self.defender_policy.propose_from_state(state)
                    sandbox_results.append(
                        run_sandbox_refactor(
                            memory_store=memory_store,
                            proposal=proposal,
                            current_question=state["question"],
                            current_answer=state["answer"],
                            regression_questions=regression_questions,
                            answer_agent=self.answer_agent,
                            judge=self.judge,
                            top_k=self.config.top_k,
                            min_score=self.config.min_score,
                        )
                    )
                except Exception as exc:
                    traces.append(
                        ConstructionStepTrace(
                            case_id=case_id,
                            episode=episode,
                            status="proposal_failed",
                            question=state["question"],
                            answer=state["answer"],
                            route=attack_dict.get("route", {}),
                            initial_defense=state.get("initial_defense", {}),
                            decision=state["decision"].to_dict(),
                            error=f"rollout {rollout_index}: {exc}",
                        )
                    )

            settlement = settle_grpo_rollouts(
                memory_store=memory_store,
                sandbox_results=sandbox_results,
                question=state["question"],
                answer=state["answer"],
                success_pool=success_pool,
                high_priority_buffer=high_priority_buffer,
                memory_archive_dir=memory_archive_dir,
                case_id=case_id,
                step=episode,
                exp_name=self.config.exp_name,
                commit_threshold=self.config.commit_threshold,
            )
            memory_store = settlement.memory_store
            traces.append(
                ConstructionStepTrace(
                    case_id=case_id,
                    episode=episode,
                    status="committed" if settlement.committed else "rolled_back",
                    question=state["question"],
                    answer=state["answer"],
                    route=attack_dict.get("route", {}),
                    initial_defense=state.get("initial_defense", {}),
                    decision=state["decision"].to_dict(),
                    settlement=settlement.to_dict(),
                )
            )

        return CaseConstructionResult(
            case_id=case_id,
            memory_store=memory_store,
            success_pool=success_pool,
            high_priority_buffer=high_priority_buffer,
            traces=traces,
        )


def construct_memories_for_graphs(
    graphs: List[Dict[str, Any]],
    constructor: EvaluationMemoryConstructor,
    output_dir: str | Path,
    initial_memory_dir: Optional[str | Path] = None,
    save_traces: bool = True,
) -> Dict[str, Any]:
    """Construct and persist per-case memories for later Evaluation."""
    output_path = Path(output_dir)
    memory_dir = output_path / "memory_states"
    trace_dir = output_path / "traces"
    archive_dir = output_path / "memory_archive"
    memory_dir.mkdir(parents=True, exist_ok=True)
    if save_traces:
        trace_dir.mkdir(parents=True, exist_ok=True)

    case_summaries = []
    for graph in graphs:
        case_id = str(graph.get("case_id", "unknown"))
        initial_memory = _load_initial_memory(case_id, initial_memory_dir)
        result = constructor.construct_case(
            graph=graph,
            initial_memory=initial_memory,
            memory_archive_dir=str(archive_dir),
        )
        result.memory_store.save(memory_dir / f"{case_id}.json")
        result.success_pool.save(memory_dir / f"{case_id}_pool.json")
        result.high_priority_buffer.save(memory_dir / f"{case_id}_buffer.json")
        if save_traces:
            (trace_dir / f"{case_id}.trace.json").write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        case_summaries.append(summarize_case_traces(result.traces, result.memory_store, case_id=case_id))

    summary = {
        "case_count": len(case_summaries),
        "memory_dir": str(memory_dir),
        "cases": case_summaries,
        "totals": summarize_construction(case_summaries),
    }
    (output_path / "construction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def strip_target_metadata(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Return the graph view allowed during memory construction."""
    safe_graph = copy.deepcopy(graph)
    for key in ("target", "answer", "question", "answer_source_ids", "gold_answer"):
        safe_graph.pop(key, None)
    return safe_graph


def summarize_case_traces(
    traces: List[ConstructionStepTrace],
    memory_store: MemoryStore,
    case_id: str = "",
) -> Dict[str, Any]:
    return {
        "case_id": case_id or (traces[0].case_id if traces else ""),
        "episodes": len(traces),
        "committed": sum(1 for trace in traces if trace.status == "committed"),
        "rolled_back": sum(1 for trace in traces if trace.status == "rolled_back"),
        "initial_defense_success": sum(1 for trace in traces if trace.status == "initial_defense_success"),
        "proposal_failed": sum(1 for trace in traces if trace.status == "proposal_failed"),
        "attack_generation_failed": sum(1 for trace in traces if trace.status == "attack_generation_failed"),
        "memory_chunks": len(memory_store.chunks),
        "memory_chars": sum(len(chunk.content) for chunk in memory_store.chunks),
    }


def summarize_construction(case_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "episodes",
        "committed",
        "rolled_back",
        "initial_defense_success",
        "proposal_failed",
        "attack_generation_failed",
        "memory_chunks",
        "memory_chars",
    ]
    return {key: sum(int(item.get(key, 0)) for item in case_summaries) for key in keys}


def build_openai_client(
    model: str,
    api_base: str,
    api_key: str = "dummy-key",
    timeout: int = 120,
) -> OpenAIChatClient:
    return OpenAIChatClient(
        model=model,
        api_key=api_key,
        base_url=api_base.rstrip("/"),
        timeout=timeout,
    )


def _source_ids_from_golden_facts(golden_facts: List[Dict[str, Any]]) -> List[str]:
    source_ids = []
    seen = set()
    for fact in golden_facts:
        source_id = str(fact.get("session_id") or fact.get("chunk_id") or fact.get("source_id") or "")
        if source_id and source_id not in seen:
            source_ids.append(source_id)
            seen.add(source_id)
    return source_ids


def _source_ids_from_state(state: Dict[str, Any]) -> List[str]:
    source_ids = _source_ids_from_golden_facts(state.get("golden_facts", []))
    seen = set(source_ids)
    selected_ids = {str(item) for item in state.get("selected_memory_ids", [])}
    memory_payload = state.get("current_memory", {})
    chunks = memory_payload.get("memories", []) if isinstance(memory_payload, dict) else []
    for chunk in chunks:
        if not isinstance(chunk, dict) or str(chunk.get("memory_id", "")) not in selected_ids:
            continue
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        old_source_ids = metadata.get("source_ids", [])
        if isinstance(old_source_ids, str):
            old_source_ids = [old_source_ids]
        for source_id in old_source_ids:
            source_id = str(source_id)
            if source_id and source_id not in seen:
                source_ids.append(source_id)
                seen.add(source_id)
    return source_ids


def _load_initial_memory(case_id: str, initial_memory_dir: Optional[str | Path]) -> Optional[MemoryStore]:
    if not initial_memory_dir:
        return None
    directory = Path(initial_memory_dir)
    exact = directory / f"{case_id}.json"
    if exact.exists():
        return MemoryStore.load(exact)
    candidates = sorted(
        path
        for path in directory.glob(f"{case_id}*.json")
        if not path.name.endswith("_pool.json") and not path.name.endswith("_buffer.json")
    )
    if candidates:
        return MemoryStore.load(candidates[-1])
    return None
