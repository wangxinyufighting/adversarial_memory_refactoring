"""Construct evaluation memories with a trained defender policy."""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from case_graph.attacker import FrozenLLMAttacker
from case_graph.baseline import AnswerEquivalenceJudge
from case_graph.coverage import CaseCoverageTracker, CoverageAwareRouteScheduler
from case_graph.defense import RetrievedMemoryAnswerAgent, SuccessPool
from case_graph.evidence import route_golden_facts
from case_graph.grpo_adapter import SYSTEM_PROMPT, build_user_prompt
from case_graph.llm import OpenAIChatClient
from case_graph.models import EVALUATOR_METADATA_SOURCE
from case_graph.pipeline import AlgorithmConfig, prepare_refactor_state
from case_graph.memory_evaluator import evaluate_refactor_proposal
from case_graph.refactoring import (
    ADD_ACTION,
    MERGE_ACTION,
    HighPriorityBuffer,
    RefactorActionDecision,
    RefactorProposal,
    RewardWeights,
    SandboxResult,
    build_sandbox_memory,
    compute_reward,
    settle_grpo_rollouts,
)
from case_graph.retriever import MemoryChunk, MemoryStore
from case_graph.routing import GraphRoute, RandomWalkRoutingPolicy, public_route_evidence

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

    tau: float = 0.55
    top_k: int = 8
    top_k_points: int = 24
    min_score: float = 0.0
    regression_sample_size: int = 8
    episodes_per_case: int = 250
    min_questions_per_case: int = 20
    coverage_threshold: float = 0.98
    critical_coverage_threshold: float = 1.0
    certification_questions: int = 60
    adaptive_stopping: bool = True
    proposal_count: int = 1
    commit_threshold: float = 0.0
    seed: int = 42
    force_add: bool = False
    routing_max_steps: int = 3
    routing_min_nodes: int = 1
    routing_attempts: int = 8
    defender_max_output_tokens: int = 800
    attacker_max_output_tokens: int = 700
    memory_save_interval: int = 0
    max_attack_failures: int = 20
    progress_log_interval: int = 1
    retriever_config: Dict[str, Any] = field(default_factory=dict)
    reward_config: Dict[str, Any] = field(default_factory=dict)
    exp_name: str = "eval_memory_construction"

    def algorithm_config(self, memory_archive_dir: Optional[str] = None) -> AlgorithmConfig:
        return AlgorithmConfig(
            tau=self.tau,
            top_k=self.top_k,
            top_k_points=self.top_k_points,
            min_score=self.min_score,
            regression_sample_size=self.regression_sample_size,
            proposal_count=self.proposal_count,
            seed=self.seed,
            commit_threshold=self.commit_threshold,
            memory_archive_dir=memory_archive_dir,
            exp_name=self.exp_name,
            retriever_config=dict(self.retriever_config or {}),
            reward_config=dict(self.reward_config or {}),
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
    phase: str = "cover"
    coverage: Dict[str, Any] = field(default_factory=dict)

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
            "phase": self.phase,
            "coverage": self.coverage,
        }


@dataclass(frozen=True)
class CaseConstructionResult:
    case_id: str
    memory_store: MemoryStore
    success_pool: SuccessPool
    high_priority_buffer: HighPriorityBuffer
    traces: List[ConstructionStepTrace]
    coverage_state: Dict[str, Any] = field(default_factory=dict)
    completion_status: str = "incomplete"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "summary": summarize_case_traces(
                self.traces,
                self.memory_store,
                completion_status=self.completion_status,
            ),
            "final_memory": {"memories": [chunk.to_dict() for chunk in self.memory_store.chunks]},
            "success_pool": self.success_pool.to_dict(),
            "high_priority_buffer": self.high_priority_buffer.to_dict(),
            "traces": [trace.to_dict() for trace in self.traces],
            "coverage_state": self.coverage_state,
            "completion_status": self.completion_status,
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
        started_at = time.monotonic()
        logger.info(
            "Calling defender policy: case=%s step=%s action=%s selected=%s",
            state.get("case_id", ""),
            state.get("step", ""),
            state.get("action", ""),
            state.get("selected_memory_ids", []),
        )
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
        logger.info(
            "Defender policy returned: case=%s step=%s chunks=%d elapsed=%.1fs",
            state.get("case_id", ""),
            state.get("step", ""),
            len(chunks),
            time.monotonic() - started_at,
        )

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


class RouteEvidenceAttacker:
    """Deterministic construction-probe generator for runs without an attacker LLM."""

    def generate(self, graph: Dict[str, Any], route: Any):
        from case_graph.attacker import AttackExample

        route_view = public_route_evidence(graph, route)
        facts = route_golden_facts(graph, route)
        edge = _preferred_route_edge(route.relationships)
        if edge is None:
            if not facts:
                raise ValueError("Route has no relationships or source facts for deterministic probe generation.")
            question = "What key information is recorded in the selected memory evidence?"
            answer = str(facts[0].get("text", ""))[:300].strip()
        else:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            relation = str(edge.get("description", "related_to") or "related_to")
            if source == "USER":
                question = f"Which entity is the user connected to by the relation '{relation}'?"
                answer = target
            elif target == "USER":
                question = f"Which entity is connected to the user by the relation '{relation}'?"
                answer = source
            else:
                question = f"Which entity is connected from {source} by the relation '{relation}'?"
                answer = target

        if not answer:
            raise ValueError("Deterministic probe generation produced an empty answer.")
        return AttackExample(
            case_id=str(graph.get("case_id", "")),
            question=question,
            answer=answer,
            golden_facts=facts,
            route=route_view,
        )


class CoverageGraphAttacker:
    """Deterministic high-coverage probe generator for evaluation memory building."""

    def __init__(self):
        self._case_candidates: Dict[str, List[Dict[str, Any]]] = {}
        self._case_positions: Dict[str, int] = {}
        self._fallback = RouteEvidenceAttacker()

    def generate(self, graph: Dict[str, Any], route: Any):
        from case_graph.attacker import AttackExample

        case_id = str(graph.get("case_id", ""))
        candidates = self._case_candidates.get(case_id)
        if candidates is None:
            candidates = _build_coverage_candidates(graph)
            self._case_candidates[case_id] = candidates

        if not candidates:
            return self._fallback.generate(graph, route)

        position = self._case_positions.get(case_id, 0)
        self._case_positions[case_id] = position + 1
        candidate = dict(candidates[position % len(candidates)])
        candidate["question"] = _coverage_question_variant(
            candidate,
            variant=(position // len(candidates)) % 4,
        )
        return AttackExample(
            case_id=case_id,
            question=candidate["question"],
            answer=candidate["answer"],
            golden_facts=candidate["golden_facts"],
            route=candidate["route"],
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
        coverage_tracker = CaseCoverageTracker.from_graph(safe_graph)
        route_scheduler = CoverageAwareRouteScheduler(
            routing_policy,
            candidate_attempts=max(8, self.config.routing_attempts * 2),
            random_exploration_ratio=0.15,
        )
        algorithm_config = self.config.algorithm_config(memory_archive_dir=memory_archive_dir)
        traces: List[ConstructionStepTrace] = []
        attack_failures = 0
        completion_status = "incomplete"
        logger.info(
            "Starting case %s: episodes=%d initial_memory_chunks=%d",
            case_id,
            self.config.episodes_per_case,
            len(memory_store.chunks),
        )

        for episode in range(max(0, self.config.episodes_per_case)):
            seed = self.config.seed + episode * 1009
            route_payload: Dict[str, Any] = {}
            should_log = _should_log_progress(
                episode=episode,
                total=self.config.episodes_per_case,
                interval=self.config.progress_log_interval,
            )
            if should_log:
                logger.info(
                    "Case %s episode %d/%d begin: memory_chunks=%d",
                    case_id,
                    episode + 1,
                    self.config.episodes_per_case,
                    len(memory_store.chunks),
                )
            try:
                route = route_scheduler.select_route(safe_graph, coverage_tracker, seed=seed)
                route_payload = route.to_dict()
                attack = self.attacker.generate(safe_graph, route)
                attack_dict = attack.to_dict()
                attack_dict["case_id"] = case_id
                attack_dict["route_evidence"] = attack_dict.get("route", {})
                coverage_unit_ids = coverage_tracker.unit_ids_for_route(
                    attack_dict.get("route", route)
                )
                attack_dict["coverage_unit_ids"] = coverage_unit_ids
            except Exception as exc:
                attack_failures += 1
                if attack_failures <= 3:
                    logger.warning(
                        "Attack generation failed for case %s episode %d: %s",
                        case_id,
                        episode,
                        exc,
                    )
                traces.append(
                    ConstructionStepTrace(
                        case_id=case_id,
                        episode=episode,
                        status="attack_generation_failed",
                        route=route_payload,
                        error=str(exc),
                    )
                )
                if attack_failures >= self.config.max_attack_failures:
                    logger.warning("Stopping case %s after %d attack failures", case_id, attack_failures)
                    break
                continue

            try:
                certification_phase = bool(
                    self.config.adaptive_stopping
                    and episode >= self.config.min_questions_per_case
                    and coverage_tracker.is_coverage_ready(
                        self.config.coverage_threshold,
                        self.config.critical_coverage_threshold,
                    )
                )
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
                coverage_tracker.record(
                    coverage_unit_ids,
                    success=True,
                    certification=certification_phase,
                )
                traces.append(
                    ConstructionStepTrace(
                        case_id=case_id,
                        episode=episode,
                        status="initial_defense_success",
                        question=str(attack_dict.get("question", "")),
                        answer=str(attack_dict.get("answer", "")),
                        route=attack_dict.get("route", {}),
                        phase="certify" if certification_phase else "cover",
                        coverage=coverage_tracker.snapshot(),
                    )
                )
                if should_log:
                    logger.info(
                        "Case %s episode %d/%d initial defense success: memory_chunks=%d",
                        case_id,
                        episode + 1,
                        self.config.episodes_per_case,
                        len(memory_store.chunks),
                    )
                if (
                    certification_phase
                    and coverage_tracker.consecutive_certification_passes
                    >= self.config.certification_questions
                ):
                    completion_status = "done"
                    logger.info(
                        "Case %s certified after %d questions: coverage=%.3f",
                        case_id,
                        episode + 1,
                        coverage_tracker.structural_coverage(),
                    )
                    break
                continue

            state["coverage_unit_ids"] = coverage_unit_ids
            route_weight = coverage_tracker.route_weight(coverage_unit_ids)
            pending_weight = coverage_tracker.pending_weight(coverage_unit_ids)
            state["coverage_route_weight"] = route_weight
            state["coverage_pending_weight"] = pending_weight
            state["coverage_critical_pending_weight"] = pending_weight
            state["coverage_before"] = coverage_tracker.structural_coverage()
            if certification_phase:
                coverage_tracker.record(coverage_unit_ids, success=False, certification=True)

            if self.config.force_add and state["action"] != ADD_ACTION:
                state = _force_add_state(state, tau=self.config.tau)

            sandbox_results = []
            for rollout_index in range(max(1, self.config.proposal_count)):
                try:
                    proposal = self.defender_policy.propose_from_state(state)
                    if should_log:
                        logger.info(
                            "Running sandbox: case=%s episode=%d/%d rollout=%d/%d",
                            case_id,
                            episode + 1,
                            self.config.episodes_per_case,
                            rollout_index + 1,
                            max(1, self.config.proposal_count),
                        )
                    sandbox_results.append(
                        _run_evaluation_aligned_sandbox(
                            state=state,
                            memory_store=memory_store,
                            proposal=proposal,
                            answer_agent=self.answer_agent,
                            judge=self.judge,
                            reward_config=self.config.reward_config,
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
            selected_evaluation = (
                settlement.selected_result.evaluation
                if settlement.selected_result is not None
                else None
            )
            selected_judge = (
                selected_evaluation.current_test.judge
                if selected_evaluation is not None
                else {}
            ) or {}
            coverage_success = bool(
                settlement.committed
                and selected_evaluation is not None
                and selected_evaluation.current_test.correct
                and selected_judge.get("structured_complete", selected_judge.get("complete", False))
            )
            coverage_tracker.record(coverage_unit_ids, success=coverage_success)
            if should_log:
                selected_reward = (
                    settlement.selected_result.reward.reward
                    if settlement.selected_result is not None
                    else (max(settlement.all_rewards) if settlement.all_rewards else 0.0)
                )
                logger.info(
                    "Case %s episode %d/%d %s: reward=%.3f memory_chunks=%d",
                    case_id,
                    episode + 1,
                    self.config.episodes_per_case,
                    "committed" if settlement.committed else "rolled back",
                    float(selected_reward),
                    len(memory_store.chunks),
                )
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
                    phase="repair" if certification_phase else "cover",
                    coverage=coverage_tracker.snapshot(),
                )
            )

        return CaseConstructionResult(
            case_id=case_id,
            memory_store=memory_store,
            success_pool=success_pool,
            high_priority_buffer=high_priority_buffer,
            traces=traces,
            coverage_state=coverage_tracker.snapshot(),
            completion_status=completion_status,
        )


def _run_evaluation_aligned_sandbox(
    state: Dict[str, Any],
    memory_store: MemoryStore,
    proposal: RefactorProposal,
    answer_agent: RetrievedMemoryAnswerAgent,
    judge: AnswerEquivalenceJudge,
    reward_config: Optional[Dict[str, Any]] = None,
) -> SandboxResult:
    temp_memory = build_sandbox_memory(
        memory_store,
        proposal,
        current_question=str(state.get("question", "")),
    )
    evaluation = evaluate_refactor_proposal(
        temp_memory=temp_memory,
        proposal=proposal,
        state=state,
        answer_agent=answer_agent,
        judge=judge,
    )
    weights = RewardWeights.from_config((reward_config or {}).get("weights", {}))
    reward = compute_reward(proposal, evaluation, weights)
    return SandboxResult(
        proposal=proposal,
        temp_memory=temp_memory,
        evaluation=evaluation,
        reward=reward,
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
    coverage_dir = output_path / "coverage_states"
    archive_dir = output_path / "memory_archive"
    memory_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir.mkdir(parents=True, exist_ok=True)
    if save_traces:
        trace_dir.mkdir(parents=True, exist_ok=True)

    case_summaries = []
    for index, graph in enumerate(graphs, start=1):
        case_id = str(graph.get("case_id", "unknown"))
        logger.info("Constructing memory for case %d/%d: %s", index, len(graphs), case_id)
        initial_memory = _load_initial_memory(case_id, initial_memory_dir)
        result = constructor.construct_case(
            graph=graph,
            initial_memory=initial_memory,
            memory_archive_dir=str(archive_dir),
        )
        result.memory_store.save(memory_dir / f"{case_id}.json")
        result.success_pool.save(memory_dir / f"{case_id}_pool.json")
        result.high_priority_buffer.save(memory_dir / f"{case_id}_buffer.json")
        (coverage_dir / f"{case_id}.json").write_text(
            json.dumps(
                {
                    "completion_status": result.completion_status,
                    **result.coverage_state,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if save_traces:
            (trace_dir / f"{case_id}.trace.json").write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        case_summary = summarize_case_traces(
            result.traces,
            result.memory_store,
            case_id=case_id,
            completion_status=result.completion_status,
        )
        case_summaries.append(case_summary)
        logger.info("Finished case %s: %s", case_id, case_summary)

    summary = {
        "case_count": len(case_summaries),
        "memory_dir": str(memory_dir),
        "cases": case_summaries,
        "coverage_dir": str(coverage_dir),
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
    completion_status: str = "incomplete",
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
        "completion_status": completion_status,
        "structural_coverage": (
            float(traces[-1].coverage.get("structural_coverage", 0.0)) if traces else 0.0
        ),
        "critical_coverage": (
            float(traces[-1].coverage.get("critical_coverage", 0.0)) if traces else 0.0
        ),
        "certification_passes": (
            int(traces[-1].coverage.get("consecutive_certification_passes", 0)) if traces else 0
        ),
        "attack_failure_errors": _top_errors(
            trace.error for trace in traces if trace.status == "attack_generation_failed"
        ),
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
    totals = {key: sum(int(item.get(key, 0)) for item in case_summaries) for key in keys}
    totals["done_cases"] = sum(item.get("completion_status") == "done" for item in case_summaries)
    totals["incomplete_cases"] = len(case_summaries) - totals["done_cases"]
    return totals


def _should_log_progress(episode: int, total: int, interval: int) -> bool:
    if total <= 0:
        return False
    if episode == 0 or episode == total - 1:
        return True
    if interval <= 0:
        return False
    return (episode + 1) % interval == 0


def _force_add_state(state: Dict[str, Any], tau: float) -> Dict[str, Any]:
    """Debug-only override that keeps memories independent instead of merging."""
    original_decision = state.get("decision")
    max_score = float(getattr(original_decision, "max_score", 0.0) or 0.0)
    similarity_report = getattr(original_decision, "similarity_report", None)
    forced = dict(state)
    forced["action"] = ADD_ACTION
    forced["selected_memory_ids"] = []
    forced["regression_questions"] = []
    forced["decision"] = RefactorActionDecision(
        action=ADD_ACTION,
        question=str(state.get("question", "")),
        tau=tau,
        max_score=max_score,
        selected_memory_ids=[],
        reason="Debug force-add override disabled the normal Add/Merge decision.",
        similarity_report=similarity_report,
    )
    return forced


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


def _build_coverage_candidates(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    entities = _entity_map(graph)
    edges = [
        edge
        for edge in graph.get("relationships", [])
        if _is_public_coverage_edge(edge, entities)
    ]
    if not edges:
        return []

    ordered_edges = _coverage_order(edges, graph)
    candidates = []
    seen_keys = set()
    for edge in ordered_edges:
        candidate = _coverage_candidate_from_edge(graph, edge, entities)
        if not candidate:
            continue
        key = (
            candidate["question"],
            candidate["answer"],
            tuple(_source_ids_from_golden_facts(candidate["golden_facts"])),
        )
        if key in seen_keys:
            continue
        candidates.append(candidate)
        seen_keys.add(key)
    return candidates


def _coverage_candidate_from_edge(
    graph: Dict[str, Any],
    edge: Dict[str, Any],
    entities: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    source = str(edge.get("source", ""))
    target = str(edge.get("target", ""))
    if not source or not target:
        return None

    if source == "USER":
        answer = target
        answer_entity = entities.get(target, {})
        question = _coverage_question_for_user_edge(edge, answer_entity)
    elif target == "USER":
        answer = source
        answer_entity = entities.get(source, {})
        question = _coverage_question_for_user_edge(edge, answer_entity, reverse=True)
    else:
        answer = target
        answer_entity = entities.get(target, {})
        question = (
            f"In this memory, {source} has a '{_relation_phrase(edge)}' relation "
            f"with what {_entity_label(answer_entity)}?"
        )

    route = GraphRoute(
        policy="coverage",
        nodes=[source, target],
        relationships=[edge],
        reason="Coverage probe over graph evidence.",
    )
    facts = route_golden_facts(graph, route)
    if not facts:
        return None
    return {
        "question": question,
        "answer": answer,
        "golden_facts": facts,
        "route": public_route_evidence(graph, route),
    }


def _coverage_question_variant(candidate: Dict[str, Any], variant: int) -> str:
    if variant == 0:
        return str(candidate["question"])
    relationships = candidate.get("route", {}).get("relationships", [])
    if not relationships:
        return str(candidate["question"])
    edge = relationships[0]
    source = str(edge.get("source", ""))
    target = str(edge.get("target", ""))
    relation = str(edge.get("description") or "related_to")
    if source == "USER":
        templates = [
            f"Which remembered entity has the '{relation}' relation from the user?",
            f"Identify what the user is linked to through '{relation}'.",
            f"According to memory, the user's '{relation}' relation points to what?",
        ]
    elif target == "USER":
        templates = [
            f"Which remembered entity has the '{relation}' relation to the user?",
            f"Identify what is linked to the user through '{relation}'.",
            f"According to memory, what points to the user through '{relation}'?",
        ]
    else:
        templates = [
            f"What does {source} connect to through '{relation}'?",
            f"Identify the target of {source}'s '{relation}' relation.",
            f"According to memory, {source} has '{relation}' with what?",
        ]
    del target
    return templates[(variant - 1) % len(templates)]


def _coverage_question_for_user_edge(
    edge: Dict[str, Any],
    answer_entity: Dict[str, Any],
    reverse: bool = False,
) -> str:
    label = _entity_label(answer_entity)
    relation = _relation_phrase(edge)
    if reverse:
        return f"In this memory, what {label} has a '{relation}' relation with the user?"
    return f"In this memory, the user has a '{relation}' relation with what {label}?"


def _coverage_order(edges: List[Dict[str, Any]], graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Interleave source sessions so a bounded run does not starve later facts."""
    source_order = {
        str(chunk.get("chunk_id", "")): int(chunk.get("order", index))
        for index, chunk in enumerate(graph.get("chunks", []))
    }

    def source_key(edge: Dict[str, Any]) -> str:
        source_ids = [str(item) for item in edge.get("source_ids", [])]
        return min(
            source_ids,
            key=lambda item: (source_order.get(item, 10**9), item),
            default="",
        )

    def edge_priority(edge: Dict[str, Any]) -> tuple:
        try:
            weight = float(edge.get("weight", 1.0) or 1.0)
        except (TypeError, ValueError):
            weight = 1.0
        concrete_rank = (
            0
            if edge.get("source") != "USER" and edge.get("target") != "USER"
            else 1
        )
        return (
            -weight,
            concrete_rank,
            str(edge.get("source", "")),
            str(edge.get("target", "")),
            str(edge.get("description", "")),
        )

    edges_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for edge in edges:
        edges_by_source.setdefault(source_key(edge), []).append(edge)
    for source_edges in edges_by_source.values():
        source_edges.sort(key=edge_priority)

    ordered_sources = sorted(
        edges_by_source,
        key=lambda item: (source_order.get(item, 10**9), item),
    )
    ordered_edges: List[Dict[str, Any]] = []
    depth = 0
    while True:
        added = False
        for source_id in ordered_sources:
            source_edges = edges_by_source[source_id]
            if depth < len(source_edges):
                ordered_edges.append(source_edges[depth])
                added = True
        if not added:
            break
        depth += 1
    return ordered_edges


def _is_public_coverage_edge(edge: Dict[str, Any], entities: Dict[str, Dict[str, Any]]) -> bool:
    if edge.get("description") == "target_answer":
        return False
    metadata = edge.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("source") == EVALUATOR_METADATA_SOURCE:
        return False
    return not (
        _is_evaluator_entity(entities.get(str(edge.get("source", "")), {}))
        or _is_evaluator_entity(entities.get(str(edge.get("target", "")), {}))
    )


def _is_evaluator_entity(entity: Dict[str, Any]) -> bool:
    metadata = entity.get("metadata", {})
    description = str(entity.get("description", ""))
    return (
        isinstance(metadata, dict)
        and metadata.get("source") == EVALUATOR_METADATA_SOURCE
    ) or "Memory fact recovered from evaluator metadata." in description


def _entity_map(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(entity.get("name", "")): entity for entity in graph.get("entities", [])}


def _entity_label(entity: Dict[str, Any]) -> str:
    raw = str(entity.get("entity_type") or entity.get("type") or "thing").strip()
    label = raw.split("/", 1)[0].casefold() if raw else "thing"
    return {
        "event": "event",
        "object": "item",
        "organization": "organization",
        "place": "place",
        "person": "person",
        "resource": "resource",
        "statistic": "detail",
        "time": "time",
        "interest": "interest",
        "interest/skill": "interest",
        "goal": "goal",
        "goal/intention": "goal",
    }.get(label, "thing")


def _relation_phrase(edge: Dict[str, Any]) -> str:
    relation = str(edge.get("description") or "related_to")
    relation = relation.replace("<SEP>", " or ")
    relation = relation.replace("_", " ")
    relation = " ".join(relation.split())
    return relation or "related to"


def _preferred_route_edge(edges: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not edges:
        return None
    for edge in edges:
        if edge.get("source") == "USER" or edge.get("target") == "USER":
            return edge
    return edges[0]


def _top_errors(errors: Any, limit: int = 3) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for error in errors:
        error = str(error or "").strip()
        if not error:
            continue
        counts[error] = counts.get(error, 0) + 1
    return [
        {"error": error, "count": count}
        for error, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


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
