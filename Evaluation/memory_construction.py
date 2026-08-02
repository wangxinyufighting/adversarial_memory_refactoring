"""Construct evaluation memories with a trained defender policy."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from case_graph.attacker import FrozenLLMAttacker
from case_graph.baseline import AnswerEquivalenceJudge
from case_graph.coverage import (
    CaseCoverageTracker,
    CoverageAwareRouteScheduler,
    CoverageUnit,
    relationship_unit_id,
)
from case_graph.defense import RetrievedMemoryAnswerAgent, SuccessPool
from case_graph.evidence import route_golden_facts
from case_graph.grpo_adapter import SYSTEM_PROMPT, build_user_prompt
from case_graph.llm import OpenAIChatClient
from case_graph.models import EVALUATOR_METADATA_SOURCE
from case_graph.pipeline import AlgorithmConfig, prepare_refactor_state
from case_graph.memory_evaluator import evaluate_completeness, evaluate_refactor_proposal
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
from case_graph.retriever import MemoryChunk, MemoryStore, build_memory_retriever
from case_graph.routing import (
    GraphRoute,
    RandomWalkRoutingPolicy,
    public_route_evidence,
    target_free_graph,
)

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
    dynamic_question_budget: bool = True
    questions_per_unit: float = 3.0
    hard_max_questions_per_case: int = 2000
    min_questions_per_case: int = 20
    coverage_threshold: float = 0.98
    critical_coverage_threshold: float = 1.0
    certification_questions: int = 60
    adaptive_stopping: bool = True
    qa_early_stopping: bool = False
    qa_stop_question_count: int = 50
    qa_stop_accuracy: float = 0.9
    qa_stop_check_interval: int = 20
    proposal_count: int = 1
    commit_threshold: float = 0.0
    seed: int = 42
    force_add: bool = False
    routing_max_steps: int = 3
    routing_min_nodes: int = 1
    routing_attempts: int = 8
    defender_max_output_tokens: int = 4096
    defender_proposal_retries: int = 2
    attacker_max_output_tokens: int = 700
    max_retries_per_unit: int = 3
    compositional_probes: bool = True
    trace_detail: str = "compact"
    case_workers: int = 1
    memory_save_interval: int = 0
    max_attack_failures: int = 20
    max_consecutive_proposal_failures: int = 10
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
    stop_reason: str = "question_budget_exhausted"
    question_budget: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "summary": summarize_case_traces(
                self.traces,
                self.memory_store,
                completion_status=self.completion_status,
                stop_reason=self.stop_reason,
                question_budget=self.question_budget,
                coverage_state=self.coverage_state,
            ),
            "final_memory": {"memories": [chunk.to_dict() for chunk in self.memory_store.chunks]},
            "success_pool": self.success_pool.to_dict(),
            "high_priority_buffer": self.high_priority_buffer.to_dict(),
            "traces": [trace.to_dict() for trace in self.traces],
            "coverage_state": self.coverage_state,
            "completion_status": self.completion_status,
            "stop_reason": self.stop_reason,
            "question_budget": self.question_budget,
        }


class DefenderCheckpointPolicy:
    """Inference-only wrapper around the trained defender checkpoint server."""

    def __init__(
        self,
        client: DefenderClient,
        max_output_tokens: int = 4096,
        retries: int = 2,
    ):
        self.client = client
        self.max_output_tokens = max_output_tokens
        self.retries = max(0, int(retries))

    def propose_from_state(self, state: Dict[str, Any]) -> RefactorProposal:
        started_at = time.monotonic()
        logger.info(
            "Calling defender policy: case=%s step=%s action=%s selected=%s",
            state.get("case_id", ""),
            state.get("step", ""),
            state.get("action", ""),
            state.get("selected_memory_ids", []),
        )
        response: Dict[str, Any] = {}
        chunks: List[MemoryChunk] = []
        errors: List[str] = []
        base_prompt = _evaluation_policy_prompt(state)
        for attempt in range(self.retries + 1):
            try:
                prompt = base_prompt if attempt == 0 else _repair_policy_prompt(base_prompt, errors[-1])
                response = self.client.complete_json(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    max_tokens=self.max_output_tokens,
                )
                chunks = _policy_chunks_from_response(response, state)
                if not chunks:
                    preview = json.dumps(response, ensure_ascii=False, default=str)[:1000]
                    raise ValueError(
                        "response contains no non-empty chunks; "
                        f"response_preview={preview}"
                    )
                break
            except Exception as exc:
                errors.append(str(exc))
                logger.warning(
                    "Defender proposal attempt %d/%d failed: case=%s step=%s error=%s",
                    attempt + 1,
                    self.retries + 1,
                    state.get("case_id", ""),
                    state.get("step", ""),
                    str(exc)[:1200],
                )
                if attempt >= self.retries:
                    raise ValueError(
                        "Defender proposal failed after "
                        f"{self.retries + 1} attempts: {' | '.join(errors)}"
                    ) from exc
        logger.info(
            "Defender policy returned: case=%s step=%s chunks=%d elapsed=%.1fs",
            state.get("case_id", ""),
            state.get("step", ""),
            len(chunks),
            time.monotonic() - started_at,
        )

        source_ids = _source_ids_from_state(state)
        for index, chunk in enumerate(chunks):
            chunk.content = _best_chunk_content(chunk, state)
            chunk.memory_id = _system_memory_id(state, index, chunk.content)
            metadata = _sanitize_policy_metadata(chunk.metadata, chunk.content)
            metadata["source"] = "defender_checkpoint_eval_construction"
            metadata["case_id"] = state.get("case_id", "")
            metadata["construction_question"] = state.get("question", "")
            metadata["source_ids"] = list(source_ids)
            metadata["construction_step"] = state.get("step", 0)
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


class CoverageCandidatesExhausted(RuntimeError):
    """Raised when every uncovered coverage unit reached its retry limit."""


class CoverageGraphAttacker:
    """Deterministic high-coverage probe generator for evaluation memory building."""

    def __init__(self, compositional_probes: bool = True):
        self._case_candidates: Dict[str, List[Dict[str, Any]]] = {}
        self._case_positions: Dict[str, int] = {}
        self._last_candidate: Dict[str, int] = {}
        self.compositional_probes = bool(compositional_probes)
        self._fallback = RouteEvidenceAttacker()

    def prepare(self, graph: Dict[str, Any], tracker: CaseCoverageTracker) -> None:
        case_id = str(graph.get("case_id", ""))
        candidates = self._case_candidates.get(case_id)
        if candidates is None:
            candidates = _build_coverage_candidates(
                graph,
                include_compositional=self.compositional_probes,
            )
            self._case_candidates[case_id] = candidates
        for candidate in candidates:
            unit = candidate.get("composite_unit")
            if isinstance(unit, CoverageUnit):
                tracker.add_unit(unit)

    def generate(
        self,
        graph: Dict[str, Any],
        route: Any,
        tracker: Optional[CaseCoverageTracker] = None,
        certification: bool = False,
        max_retries_per_unit: int = 3,
        seed: int = 0,
    ):
        from case_graph.attacker import AttackExample

        case_id = str(graph.get("case_id", ""))
        candidates = self._case_candidates.get(case_id)
        if candidates is None:
            candidates = _build_coverage_candidates(
                graph,
                include_compositional=self.compositional_probes,
            )
            self._case_candidates[case_id] = candidates

        if not candidates:
            if route is None:
                raise CoverageCandidatesExhausted(
                    "The graph contains no public relationship with source evidence."
                )
            return self._fallback.generate(graph, route)

        position = self._case_positions.get(case_id, 0)
        if tracker is None:
            candidate_index = position % len(candidates)
        else:
            candidate_index = self._select_candidate(
                case_id=case_id,
                candidates=candidates,
                tracker=tracker,
                certification=certification,
                max_retries_per_unit=max_retries_per_unit,
                seed=seed,
            )
        self._case_positions[case_id] = position + 1
        self._last_candidate[case_id] = candidate_index
        candidate = dict(candidates[candidate_index])
        attempts = _candidate_attempts(candidate, tracker) if tracker is not None else position
        candidate["question"] = _coverage_question_variant(
            candidate,
            variant=attempts % 4,
        )
        return AttackExample(
            case_id=case_id,
            question=candidate["question"],
            answer=candidate["answer"],
            golden_facts=candidate["golden_facts"],
            route=candidate["route"],
        )

    def _select_candidate(
        self,
        case_id: str,
        candidates: List[Dict[str, Any]],
        tracker: CaseCoverageTracker,
        certification: bool,
        max_retries_per_unit: int,
        seed: int,
    ) -> int:
        retry_limit = max(1, int(max_retries_per_unit))
        if certification:
            eligible = [
                index
                for index, candidate in enumerate(candidates)
                if _candidate_is_covered(candidate, tracker)
            ]
            if not eligible:
                raise CoverageCandidatesExhausted("No covered candidate is available for certification.")
            return min(
                eligible,
                key=lambda index: (
                    _candidate_certified_passes(candidates[index], tracker),
                    -_candidate_priority(candidates[index], tracker),
                    index,
                ),
            )

        last_index = self._last_candidate.get(case_id)
        if last_index is not None:
            last = candidates[last_index]
            if (
                not _candidate_is_covered(last, tracker)
                and 0 < _candidate_attempts(last, tracker) < retry_limit
            ):
                return last_index

        eligible = [
            index
            for index, candidate in enumerate(candidates)
            if (
                not _candidate_is_covered(candidate, tracker)
                and _candidate_attempts(candidate, tracker) < retry_limit
            )
        ]
        unresolved_critical = any(
            unit.critical and not tracker.statuses[unit_id].covered
            for unit_id, unit in tracker.units.items()
        )
        if unresolved_critical:
            eligible = [
                index
                for index in eligible
                if _candidate_has_uncovered_critical(candidates[index], tracker)
            ]
        if not eligible:
            raise CoverageCandidatesExhausted(
                f"All uncovered coverage candidates reached {retry_limit} attempts."
            )
        offset = seed % max(1, len(candidates))
        return max(
            eligible,
            key=lambda index: (
                _candidate_priority(candidates[index], tracker),
                -_candidate_attempts(candidates[index], tracker),
                -((index - offset) % len(candidates)),
            ),
        )


def _build_qa_stop_questions(
    graph: Dict[str, Any],
    config: MemoryConstructionConfig,
) -> List[Dict[str, Any]]:
    """Create a fixed QA bank from the original case and graph evidence."""

    requested = max(0, int(config.qa_stop_question_count))
    if not config.qa_early_stopping or requested == 0:
        return []
    questions: List[Dict[str, Any]] = []
    seen = set()
    target = graph.get("target") or {}
    if isinstance(target, dict):
        target_question = str(
            target.get("question") or graph.get("question") or ""
        ).strip()
        target_answer_value = (
            target.get("answer") if "answer" in target else graph.get("answer", "")
        )
        target_answer = str(target_answer_value).strip()
        if target_question and target_answer:
            questions.append({"question": target_question, "answer": target_answer})
            seen.add((target_question.casefold(), target_answer.casefold()))
    if len(questions) >= requested:
        return questions
    tracker = CaseCoverageTracker.from_graph(graph)
    generator = CoverageGraphAttacker(compositional_probes=config.compositional_probes)
    generator.prepare(graph, tracker)
    for index in range(max(requested * 4, requested + 10)):
        try:
            attack = generator.generate(
                graph,
                route=None,
                tracker=tracker,
                max_retries_per_unit=1,
                seed=config.seed + index,
            )
        except CoverageCandidatesExhausted:
            break
        payload = attack.to_dict()
        route = payload.get("route", {}) or {}
        unit_ids = [
            str(item)
            for item in route.get("coverage_unit_ids", [])
            if str(item) in tracker.units
        ] or tracker.unit_ids_for_route(route)
        tracker.record(unit_ids, success=True)
        key = (
            str(payload.get("question", "")).casefold(),
            str(payload.get("answer", "")).casefold(),
        )
        if key in seen or not all(key):
            continue
        seen.add(key)
        questions.append(
            {
                "question": str(payload["question"]),
                "answer": str(payload["answer"]),
            }
        )
        if len(questions) >= requested:
            break
    return questions


def _evaluate_qa_stop_questions(
    memory_store: MemoryStore,
    questions: List[Dict[str, Any]],
    config: MemoryConstructionConfig,
    answer_agent: RetrievedMemoryAnswerAgent,
    judge: AnswerEquivalenceJudge,
) -> Dict[str, Any]:
    retriever = build_memory_retriever(config.retriever_config, memory_store)
    correct = 0
    errors = 0
    for item in questions:
        try:
            hits = retriever.retrieve(
                item["question"],
                top_k=config.top_k,
                min_score=config.min_score,
            )
            answer_result = answer_agent.answer(item["question"], hits)
            judge_result = judge.judge(
                question=item["question"],
                gold_answer=item["answer"],
                candidate_answer=str(answer_result.get("answer", "")),
            )
            correct += int(bool(judge_result.get("correct", False)))
        except Exception:
            errors += 1
            logger.exception("QA early-stopping question failed.")
    total = len(questions)
    return {
        "correct": correct,
        "question_count": total,
        "accuracy": correct / total if total else 0.0,
        "errors": errors,
    }


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
        memory_store = copy.deepcopy(initial_memory) if initial_memory is not None else MemoryStore()
        _ensure_unique_memory_ids(memory_store, case_id)
        success_pool = SuccessPool()
        high_priority_buffer = HighPriorityBuffer()
        routing_policy = RandomWalkRoutingPolicy(
            seed=self.config.seed,
            max_steps=self.config.routing_max_steps,
            min_nodes=self.config.routing_min_nodes,
            attempts=self.config.routing_attempts,
        )
        coverage_tracker = CaseCoverageTracker.from_graph(safe_graph)
        if isinstance(self.attacker, CoverageGraphAttacker):
            self.attacker.prepare(safe_graph, coverage_tracker)
        qa_stop_questions = _build_qa_stop_questions(graph, self.config)
        qa_stop_state = {
            "enabled": bool(self.config.qa_early_stopping),
            "requested_question_count": max(0, int(self.config.qa_stop_question_count)),
            "question_count": len(qa_stop_questions),
            "target_accuracy": min(1.0, max(0.0, float(self.config.qa_stop_accuracy))),
            "correct": 0,
            "accuracy": 0.0,
            "errors": 0,
            "checks": 0,
            "last_checked_after_questions": -1,
        }
        if qa_stop_questions:
            logger.info(
                "Case %s QA early stopping enabled: questions=%d target_accuracy=%.3f",
                case_id,
                len(qa_stop_questions),
                qa_stop_state["target_accuracy"],
            )
        elif self.config.qa_early_stopping:
            logger.warning("Case %s has no valid QA early-stopping questions.", case_id)

        def qa_stop_reached(completed_questions: int, force: bool = False) -> bool:
            if not qa_stop_questions:
                return False
            minimum = max(0, int(self.config.min_questions_per_case))
            interval = max(1, int(self.config.qa_stop_check_interval))
            if completed_questions < minimum:
                return False
            if not force and completed_questions % interval:
                return False
            if qa_stop_state["last_checked_after_questions"] == completed_questions:
                return qa_stop_state["accuracy"] >= qa_stop_state["target_accuracy"]
            metrics = _evaluate_qa_stop_questions(
                memory_store,
                qa_stop_questions,
                self.config,
                self.answer_agent,
                self.judge,
            )
            qa_stop_state.update(metrics)
            qa_stop_state["checks"] += 1
            qa_stop_state["last_checked_after_questions"] = completed_questions
            logger.info(
                "Case %s QA early-stop check after %d questions: %d/%d (%.3f), target=%.3f",
                case_id,
                completed_questions,
                metrics["correct"],
                metrics["question_count"],
                metrics["accuracy"],
                qa_stop_state["target_accuracy"],
            )
            return metrics["accuracy"] >= qa_stop_state["target_accuracy"]

        route_scheduler = CoverageAwareRouteScheduler(
            routing_policy,
            candidate_attempts=max(8, self.config.routing_attempts * 2),
            random_exploration_ratio=0.15,
        )
        algorithm_config = self.config.algorithm_config(memory_archive_dir=memory_archive_dir)
        traces: List[ConstructionStepTrace] = []
        attack_failures = 0
        consecutive_proposal_failures = 0
        completion_status = "incomplete"
        stop_reason = "question_budget_exhausted"
        question_budget = _resolve_question_budget(
            self.config,
            coverage_tracker,
            adaptive_coverage=isinstance(self.attacker, CoverageGraphAttacker),
        )
        logger.info(
            "Starting case %s: requested_questions=%d resolved_budget=%d coverage_units=%d "
            "critical_units=%d initial_memory_chunks=%d",
            case_id,
            self.config.episodes_per_case,
            question_budget,
            len(coverage_tracker.units),
            coverage_tracker.critical_unit_count(),
            len(memory_store.chunks),
        )

        for episode in range(question_budget):
            if qa_stop_reached(episode):
                completion_status = "done"
                stop_reason = "qa_accuracy_reached"
                break
            seed = self.config.seed + episode * 1009
            route_payload: Dict[str, Any] = {}
            should_log = _should_log_progress(
                episode=episode,
                total=question_budget,
                interval=self.config.progress_log_interval,
            )
            if should_log:
                logger.info(
                    "Case %s episode %d/%d begin: memory_chunks=%d",
                    case_id,
                    episode + 1,
                    question_budget,
                    len(memory_store.chunks),
                )
            certification_phase = bool(
                self.config.adaptive_stopping
                and coverage_tracker.is_coverage_ready(
                    self.config.coverage_threshold,
                    self.config.critical_coverage_threshold,
                )
            )
            try:
                if isinstance(self.attacker, CoverageGraphAttacker):
                    attack = self.attacker.generate(
                        safe_graph,
                        route=None,
                        tracker=coverage_tracker,
                        certification=certification_phase,
                        max_retries_per_unit=self.config.max_retries_per_unit,
                        seed=seed,
                    )
                else:
                    route = route_scheduler.select_route(safe_graph, coverage_tracker, seed=seed)
                    route_payload = route.to_dict()
                    attack = self.attacker.generate(safe_graph, route)
                attack_dict = attack.to_dict()
                attack_dict["case_id"] = case_id
                attack_dict["route_evidence"] = attack_dict.get("route", {})
                route_payload = attack_dict.get("route", route_payload)
                coverage_unit_ids = [
                    str(item)
                    for item in (attack_dict.get("route", {}).get("coverage_unit_ids", []) or [])
                    if str(item) in coverage_tracker.units
                ]
                if not coverage_unit_ids:
                    coverage_unit_ids = coverage_tracker.unit_ids_for_route(route_payload)
                attack_dict["coverage_unit_ids"] = coverage_unit_ids
            except CoverageCandidatesExhausted as exc:
                stop_reason = "coverage_candidates_exhausted"
                logger.warning("Stopping case %s: %s", case_id, exc)
                break
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
                    stop_reason = "attack_failure_limit"
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
                    initial_success_validator=_initial_success_validator(
                        attack_dict,
                        self.config.reward_config,
                    ),
                    return_initial_success_state=True,
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
                coverage_tracker.record(
                    coverage_unit_ids,
                    success=False,
                    certification=certification_phase,
                )
                continue

            if state is None or state.get("initial_defense_success", False):
                coverage_tracker.record(
                    coverage_unit_ids,
                    success=True,
                    certification=certification_phase,
                )
                _mark_supported_sibling_units(
                    coverage_tracker,
                    memory_store,
                    _source_ids_from_golden_facts(attack_dict.get("golden_facts", [])),
                )
                traces.append(
                    ConstructionStepTrace(
                        case_id=case_id,
                        episode=episode,
                        status="initial_defense_success",
                        question=str(attack_dict.get("question", "")),
                        answer=str(attack_dict.get("answer", "")),
                        route=attack_dict.get("route", {}),
                        initial_defense=(state or {}).get("initial_defense", {}),
                        phase="certify" if certification_phase else "cover",
                        coverage=coverage_tracker.snapshot(),
                    )
                )
                if should_log:
                    logger.info(
                        "Case %s episode %d/%d initial defense success: memory_chunks=%d",
                        case_id,
                        episode + 1,
                        question_budget,
                        len(memory_store.chunks),
                    )
                if _certification_done(coverage_tracker, self.config, certification_phase):
                    completion_status = "done"
                    stop_reason = "certified"
                    logger.info(
                        "Case %s certified after %d questions: coverage=%.3f",
                        case_id,
                        episode + 1,
                        coverage_tracker.required_coverage(),
                    )
                    break
                continue

            state["coverage_unit_ids"] = coverage_unit_ids
            route_weight = coverage_tracker.route_weight(coverage_unit_ids)
            pending_weight = coverage_tracker.pending_weight(coverage_unit_ids)
            state["coverage_route_weight"] = route_weight
            state["coverage_pending_weight"] = pending_weight
            state["coverage_critical_pending_weight"] = coverage_tracker.critical_pending_weight(
                coverage_unit_ids
            )
            state["coverage_before"] = coverage_tracker.required_coverage()

            if self.config.force_add and state["action"] != ADD_ACTION:
                state = _force_add_state(state, tau=self.config.tau)

            sandbox_results: List[SandboxResult] = []
            proposal_errors: List[str] = []
            for rollout_index in range(max(1, self.config.proposal_count)):
                try:
                    proposal = self.defender_policy.propose_from_state(state)
                    if should_log:
                        logger.info(
                            "Running sandbox: case=%s episode=%d/%d rollout=%d/%d",
                            case_id,
                            episode + 1,
                            question_budget,
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
                    proposal_errors.append(f"rollout {rollout_index}: {exc}")

            proposal_failed = not sandbox_results
            if proposal_failed:
                consecutive_proposal_failures += 1
                logger.error(
                    "Defender proposal failed: case=%s episode=%d/%d consecutive=%d error=%s",
                    case_id,
                    episode + 1,
                    question_budget,
                    consecutive_proposal_failures,
                    " | ".join(proposal_errors)[:2000],
                )
            else:
                consecutive_proposal_failures = 0

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
            _ensure_unique_memory_ids(memory_store, case_id)
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
                and selected_judge.get("complete", False)
                and selected_judge.get("structured_complete", True)
            )
            if certification_phase and settlement.committed:
                coverage_tracker.reset_certification()
            coverage_tracker.record(
                coverage_unit_ids,
                success=coverage_success,
                certification=certification_phase,
            )
            if settlement.committed:
                _mark_supported_sibling_units(
                    coverage_tracker,
                    memory_store,
                    _source_ids_from_golden_facts(state.get("golden_facts", [])),
                )
            if should_log and settlement.selected_result is not None:
                selected_reward = (
                    settlement.selected_result.reward.reward
                    if settlement.selected_result is not None
                    else (max(settlement.all_rewards) if settlement.all_rewards else 0.0)
                )
                logger.info(
                    "Case %s episode %d/%d %s: reward=%.3f memory_chunks=%d",
                    case_id,
                    episode + 1,
                    question_budget,
                    "committed" if settlement.committed else "rolled back",
                    float(selected_reward),
                    len(memory_store.chunks),
                )
            trace_status = (
                "proposal_failed"
                if proposal_failed
                else ("committed" if settlement.committed else "rolled_back")
            )
            traces.append(
                ConstructionStepTrace(
                    case_id=case_id,
                    episode=episode,
                    status=trace_status,
                    question=state["question"],
                    answer=state["answer"],
                    route=attack_dict.get("route", {}),
                    initial_defense=state.get("initial_defense", {}),
                    decision=state["decision"].to_dict(),
                    settlement=_settlement_trace(settlement, self.config.trace_detail),
                    error=" | ".join(proposal_errors),
                    phase="repair" if certification_phase else "cover",
                    coverage=coverage_tracker.snapshot(),
                )
            )
            proposal_failure_limit = max(
                0,
                int(self.config.max_consecutive_proposal_failures),
            )
            if (
                proposal_failure_limit > 0
                and consecutive_proposal_failures >= proposal_failure_limit
            ):
                stop_reason = "proposal_failure_limit"
                logger.error(
                    "Stopping case %s after %d consecutive defender proposal failures.",
                    case_id,
                    consecutive_proposal_failures,
                )
                break
            if _certification_done(coverage_tracker, self.config, certification_phase):
                completion_status = "done"
                stop_reason = "certified"
                logger.info(
                    "Case %s certified after %d questions: coverage=%.3f",
                    case_id,
                    episode + 1,
                    coverage_tracker.required_coverage(),
                )
                break

        if completion_status != "done" and qa_stop_reached(len(traces), force=True):
            completion_status = "done"
            stop_reason = "qa_accuracy_reached"

        if completion_status != "done" and stop_reason == "question_budget_exhausted":
            if coverage_tracker.is_coverage_ready(
                self.config.coverage_threshold,
                self.config.critical_coverage_threshold,
            ):
                stop_reason = "certification_budget_exhausted"

        _finalize_evaluation_memory(memory_store)
        coverage_state = coverage_tracker.snapshot()
        if self.config.qa_early_stopping:
            coverage_state["qa_early_stopping"] = dict(qa_stop_state)

        return CaseConstructionResult(
            case_id=case_id,
            memory_store=memory_store,
            success_pool=success_pool,
            high_priority_buffer=high_priority_buffer,
            traces=traces,
            coverage_state=coverage_state,
            completion_status=completion_status,
            stop_reason=stop_reason,
            question_budget=question_budget,
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


def _initial_success_validator(
    attack: Dict[str, Any],
    reward_config: Optional[Dict[str, Any]],
) -> Callable[[str, str, List[Any], Dict[str, Any], Dict[str, Any]], Dict[str, Any]]:
    """Require an initially correct answer to be backed by complete retrieved facts."""

    def validate(
        question: str,
        gold_answer: str,
        hits: List[Any],
        answer_result: Dict[str, Any],
        judge_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        chunks = [
            MemoryChunk(
                memory_id=hit.memory_id,
                content=hit.content,
                linked_questions=list(hit.linked_questions),
                metadata=dict(hit.metadata),
            )
            for hit in hits
        ]
        proposal = RefactorProposal(
            action=ADD_ACTION,
            new_chunks=chunks,
            remove_memory_ids=[],
            metadata={"audit": "initial_retrieved_memory"},
        )
        state = {
            "question": question,
            "answer": gold_answer,
            "golden_facts": attack.get("golden_facts", []),
            "route_evidence": attack.get("route_evidence") or attack.get("route") or {},
            "current_memory": {"memories": []},
        }
        diagnostics = evaluate_completeness(
            proposal=proposal,
            state=state,
            retrieved_memories=hits,
            reward_config=reward_config,
        )
        diagnostics["answer_agent_answer"] = str(answer_result.get("answer", ""))
        diagnostics["answer_judge_correct"] = bool(judge_result.get("correct", False))
        return diagnostics

    return validate


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

    def construct_one(index: int, graph: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
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
                    "stop_reason": result.stop_reason,
                    "question_budget": result.question_budget,
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
            stop_reason=result.stop_reason,
            question_budget=result.question_budget,
            coverage_state=result.coverage_state,
        )
        logger.info("Finished case %s: %s", case_id, case_summary)
        return index, case_summary

    case_summaries: List[Optional[Dict[str, Any]]] = [None] * len(graphs)
    workers = max(1, int(constructor.config.case_workers))
    if workers == 1:
        for index, graph in enumerate(graphs, start=1):
            result_index, case_summary = construct_one(index, graph)
            case_summaries[result_index - 1] = case_summary
    else:
        logger.info("Constructing %d cases with %d workers", len(graphs), workers)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eval-memory") as executor:
            futures = {
                executor.submit(construct_one, index, graph): index
                for index, graph in enumerate(graphs, start=1)
            }
            for future in as_completed(futures):
                result_index, case_summary = future.result()
                case_summaries[result_index - 1] = case_summary

    completed_summaries = [item for item in case_summaries if item is not None]

    summary = {
        "case_count": len(completed_summaries),
        "memory_dir": str(memory_dir),
        "cases": completed_summaries,
        "coverage_dir": str(coverage_dir),
        "case_workers": workers,
        "totals": summarize_construction(completed_summaries),
    }
    (output_path / "construction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def strip_target_metadata(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Return the graph view allowed during memory construction."""
    return target_free_graph(graph)


def summarize_case_traces(
    traces: List[ConstructionStepTrace],
    memory_store: MemoryStore,
    case_id: str = "",
    completion_status: str = "incomplete",
    stop_reason: str = "question_budget_exhausted",
    question_budget: int = 0,
    coverage_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    memory_ids = [chunk.memory_id for chunk in memory_store.chunks]
    memory_payload = [chunk.to_dict() for chunk in memory_store.chunks]
    final_coverage = dict(
        coverage_state
        or (traces[-1].coverage if traces else {})
    )
    qa_stop = dict(final_coverage.get("qa_early_stopping", {}) or {})
    return {
        "case_id": case_id or (traces[0].case_id if traces else ""),
        "episodes": len(traces),
        "committed": sum(1 for trace in traces if trace.status == "committed"),
        "rolled_back": sum(1 for trace in traces if trace.status == "rolled_back"),
        "initial_defense_success": sum(1 for trace in traces if trace.status == "initial_defense_success"),
        "proposal_failed": sum(1 for trace in traces if trace.status == "proposal_failed"),
        "attack_generation_failed": sum(1 for trace in traces if trace.status == "attack_generation_failed"),
        "memory_chunks": len(memory_store.chunks),
        "unique_memory_ids": len(set(memory_ids)),
        "duplicate_memory_ids": len(memory_ids) - len(set(memory_ids)),
        "memory_chars": sum(len(chunk.content) for chunk in memory_store.chunks),
        "memory_payload_chars": len(json.dumps(memory_payload, ensure_ascii=False)),
        "completion_status": completion_status,
        "stop_reason": stop_reason,
        "question_budget": question_budget,
        "structural_coverage": float(final_coverage.get("structural_coverage", 0.0)),
        "required_coverage": float(final_coverage.get("required_coverage", 0.0)),
        "critical_coverage": float(final_coverage.get("critical_coverage", 0.0)),
        "certification_passes": int(
            final_coverage.get("consecutive_certification_passes", 0)
        ),
        "certification_resets": int(final_coverage.get("certification_resets", 0)),
        "qa_stop_enabled": bool(qa_stop.get("enabled", False)),
        "qa_stop_accuracy": float(qa_stop.get("accuracy", 0.0)),
        "qa_stop_correct": int(qa_stop.get("correct", 0)),
        "qa_stop_question_count": int(qa_stop.get("question_count", 0)),
        "qa_stop_checks": int(qa_stop.get("checks", 0)),
        "attack_failure_errors": _top_errors(
            trace.error for trace in traces if trace.status == "attack_generation_failed"
        ),
        "proposal_failure_errors": _top_errors(
            trace.error for trace in traces if trace.status == "proposal_failed"
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
        "memory_payload_chars",
        "question_budget",
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


def _resolve_question_budget(
    config: MemoryConstructionConfig,
    tracker: CaseCoverageTracker,
    adaptive_coverage: bool,
) -> int:
    requested = max(0, int(config.episodes_per_case))
    if not config.dynamic_question_budget or not adaptive_coverage:
        return requested
    required_units = tracker.critical_unit_count() or len(tracker.units)
    dynamic = (
        math.ceil(required_units * max(0.1, float(config.questions_per_unit)))
        + max(0, int(config.certification_questions))
    )
    resolved = max(requested, int(config.min_questions_per_case), dynamic)
    hard_max = int(config.hard_max_questions_per_case)
    if hard_max > 0 and resolved > hard_max:
        logger.warning(
            "Dynamic question budget %d exceeds hard cap %d; the case may remain incomplete.",
            resolved,
            hard_max,
        )
        resolved = hard_max
    return max(0, resolved)


def _certification_done(
    tracker: CaseCoverageTracker,
    config: MemoryConstructionConfig,
    certification_phase: bool,
) -> bool:
    return bool(
        certification_phase
        and tracker.consecutive_certification_passes
        >= max(0, int(config.certification_questions))
    )


def _settlement_trace(settlement: Any, detail: str) -> Dict[str, Any]:
    if str(detail or "compact").casefold() == "full":
        return settlement.to_dict()
    selected = settlement.selected_result
    selected_payload = None
    if selected is not None:
        current = selected.evaluation.current_test
        selected_payload = {
            "proposal": selected.proposal.to_dict(),
            "evaluation": {
                "current_test": {
                    "question": current.question,
                    "gold_answer": current.gold_answer,
                    "correct": current.correct,
                    "answer_result": current.answer_result,
                    "judge": current.judge,
                    "retrieved_memory_ids": [hit.memory_id for hit in current.retrieved_memories],
                },
                "regression_accuracy": selected.evaluation.regression_accuracy,
                "failed_regression_count": selected.evaluation.failed_regression_count,
            },
            "reward": selected.reward.to_dict(),
        }
    return {
        "committed": settlement.committed,
        "selected_result": selected_payload,
        "all_rewards": list(settlement.all_rewards),
        "reason": settlement.reason,
        "archived_memory_path": settlement.archived_memory_path,
    }


def _repair_policy_prompt(base_prompt: str, error: str) -> str:
    payload = json.loads(base_prompt)
    payload["repair"] = {
        "previous_error": str(error)[:500],
        "instruction": (
            "Return one valid JSON object with a non-empty chunks list. Each chunk must contain "
            "a concise content string. Do not use markdown fences or truncate JSON."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _evaluation_policy_prompt(state: Dict[str, Any]) -> str:
    payload = json.loads(build_user_prompt(state))
    requirements = list(payload.get("requirements", []))
    requirements.append(
        "Evaluation construction: compactly preserve every distinct personal, episodic, and temporal "
        "fact supported by golden_facts, not only the shortest phrase that answers the probe. Omit "
        "generic assistant knowledge and dialogue filler."
    )
    payload["requirements"] = requirements
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _policy_chunks_from_response(
    response: Dict[str, Any],
    state: Dict[str, Any],
) -> List[MemoryChunk]:
    if not isinstance(response, dict):
        return []
    items = response.get("chunks")
    if not isinstance(items, list):
        for key in ("result", "output", "data"):
            nested = response.get(key)
            if isinstance(nested, dict) and isinstance(nested.get("chunks"), list):
                items = nested["chunks"]
                break
    if not isinstance(items, list) and any(key in response for key in ("content", "text", "summary")):
        items = [response]
    chunks = []
    for index, item in enumerate(items or []):
        if isinstance(item, str):
            item = {"content": item}
        if not isinstance(item, dict):
            continue
        chunk = MemoryChunk.from_dict(item, fallback_id=f"{state.get('action', 'add')}_{index}")
        if chunk.content.strip() or any(
            str(chunk.metadata.get(key, "")).strip() for key in ("summary", "facts")
        ):
            chunks.append(chunk)
    return chunks


def _best_chunk_content(chunk: MemoryChunk, state: Dict[str, Any]) -> str:
    metadata = chunk.metadata or {}
    summary = metadata.get("summary")
    candidates: List[tuple[str, str]] = []
    if _looks_like_raw_dialogue(chunk.content):
        candidates.extend(("excerpt", text) for text in _dialogue_excerpts(chunk.content))
    else:
        candidates.append(("content", chunk.content))
    if summary:
        candidates.append(("summary", str(summary)))
    facts = metadata.get("facts")
    if isinstance(facts, list):
        fact_texts = []
        for fact in facts:
            if isinstance(fact, str):
                fact_texts.append(fact)
            elif isinstance(fact, dict):
                text = fact.get("text") or fact.get("fact") or fact.get("content")
                if text:
                    fact_texts.append(str(text))
        if fact_texts:
            combined_facts = " ".join(fact_texts)
            if len(combined_facts) <= 1200 or not any(text.strip() for _, text in candidates):
                candidates.append(("facts", combined_facts))
    candidates = [(source, " ".join(text.split())) for source, text in candidates if str(text).strip()]
    if not candidates:
        return ""
    answer = str(state.get("answer", ""))
    route = state.get("route_evidence") or {}
    relationships = route.get("relationships", []) if isinstance(route, dict) else []
    golden_text = " ".join(
        str(item.get("text") or item.get("content") or "")
        for item in state.get("golden_facts", [])
        if isinstance(item, dict)
    )
    golden_tokens = set(_compact_tokens(golden_text))

    def score(item: tuple[str, str]) -> tuple[float, int]:
        source, text = item
        text_tokens = set(_compact_tokens(text))
        value = 5.0 if _semantic_answer_present(answer, text) else 0.0
        for edge in relationships:
            if not isinstance(edge, dict):
                continue
            for phrase in (
                str(edge.get("source", "")),
                str(edge.get("description") or edge.get("relation") or ""),
                str(edge.get("target", "")),
            ):
                value += _alias_phrase_coverage(phrase, text)
        if golden_tokens and text_tokens:
            value += 2.0 * len(golden_tokens & text_tokens) / len(golden_tokens | text_tokens)
        if source == "summary":
            value += 0.2
        value -= max(0, len(text_tokens) - 120) / 200.0
        return value, -len(text)

    return max(candidates, key=score)[1]


def _looks_like_raw_dialogue(text: str) -> bool:
    """Detect long conversational transcripts that must not become memory chunks."""

    normalized = " ".join(str(text).split())
    if len(normalized) < 500:
        return False
    conversational_markers = re.findall(
        r"\b(?:i'm|i've|i'll|can you|could you|do you|did you|by the way|thanks)\b",
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized.count("?") >= 3 or len(conversational_markers) >= 5


def _dialogue_excerpts(text: str, max_sentences: int = 3, max_chars: int = 700) -> List[str]:
    """Build short contiguous candidates instead of persisting a full transcript."""

    normalized = " ".join(str(text).split())
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", normalized) if item.strip()]
    excerpts: List[str] = []
    for start in range(len(sentences)):
        for size in range(1, max_sentences + 1):
            excerpt = " ".join(sentences[start : start + size])
            if not excerpt or len(excerpt) > max_chars:
                break
            excerpts.append(excerpt)
    return excerpts


def _sanitize_policy_metadata(metadata: Dict[str, Any], content: str) -> Dict[str, Any]:
    """Keep retrieval keys compact and never persist a hidden raw-session copy."""

    metadata = dict(metadata or {})
    result: Dict[str, Any] = {}
    summary = " ".join(str(metadata.get("summary", "")).split())
    if summary and summary.casefold() != content.casefold() and len(summary) <= 1000:
        result["summary"] = summary
    keywords = metadata.get("keywords", [])
    if isinstance(keywords, str):
        keywords = re.split(r"[,;\n]+", keywords)
    if isinstance(keywords, list):
        compact_keywords = []
        for item in keywords:
            value = " ".join(str(item).split())
            if value and len(value) <= 80 and value not in compact_keywords:
                compact_keywords.append(value)
            if len(compact_keywords) >= 20:
                break
        if compact_keywords:
            result["keywords"] = compact_keywords
    return result


def _system_memory_id(state: Dict[str, Any], index: int, content: str) -> str:
    case_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(state.get("case_id", "case"))).strip("-")
    step = int(state.get("step", 0) or 0)
    digest = hashlib.sha1(
        f"{case_id}\0{step}\0{state.get('action', '')}\0{index}\0{content}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{case_id or 'case'}_m{step:05d}_{index}_{digest}"


def _ensure_unique_memory_ids(memory_store: MemoryStore, case_id: str) -> None:
    seen = set()
    for index, chunk in enumerate(memory_store.chunks):
        original = str(chunk.memory_id or "memory")
        if original and original not in seen:
            seen.add(original)
            continue
        digest = hashlib.sha1(f"{case_id}\0{index}\0{chunk.content}".encode("utf-8")).hexdigest()[:8]
        replacement = f"{original or 'memory'}__{index}_{digest}"
        suffix = 1
        while replacement in seen:
            suffix += 1
            replacement = f"{original or 'memory'}__{index}_{digest}_{suffix}"
        metadata = dict(chunk.metadata)
        metadata["original_memory_id"] = original
        chunk.metadata = metadata
        chunk.memory_id = replacement
        seen.add(replacement)


def _finalize_evaluation_memory(memory_store: MemoryStore) -> None:
    """Drop construction-only lineage from the fixed inference artifact."""

    for chunk in memory_store.chunks:
        chunk.linked_questions = []
        metadata = dict(chunk.metadata or {})
        for key in (
            "construction_question",
            "construction_step",
            "original_memory_id",
        ):
            metadata.pop(key, None)
        chunk.metadata = metadata


def _candidate_unit_ids(
    candidate: Dict[str, Any],
    tracker: CaseCoverageTracker,
) -> List[str]:
    unit_ids = candidate.get("unit_ids") or candidate.get("route", {}).get("coverage_unit_ids", [])
    return [str(unit_id) for unit_id in unit_ids if str(unit_id) in tracker.statuses]


def _candidate_attempts(candidate: Dict[str, Any], tracker: CaseCoverageTracker) -> int:
    statuses = [tracker.statuses[unit_id] for unit_id in _candidate_unit_ids(candidate, tracker)]
    return min((status.attempts for status in statuses), default=0)


def _candidate_certified_passes(candidate: Dict[str, Any], tracker: CaseCoverageTracker) -> int:
    statuses = [tracker.statuses[unit_id] for unit_id in _candidate_unit_ids(candidate, tracker)]
    return min((status.certified_passes for status in statuses), default=0)


def _candidate_is_covered(candidate: Dict[str, Any], tracker: CaseCoverageTracker) -> bool:
    unit_ids = _candidate_unit_ids(candidate, tracker)
    return bool(unit_ids and all(tracker.statuses[unit_id].covered for unit_id in unit_ids))


def _candidate_has_uncovered_critical(
    candidate: Dict[str, Any],
    tracker: CaseCoverageTracker,
) -> bool:
    return any(
        tracker.units[unit_id].critical and not tracker.statuses[unit_id].covered
        for unit_id in _candidate_unit_ids(candidate, tracker)
    )


def _mark_supported_sibling_units(
    tracker: CaseCoverageTracker,
    memory_store: MemoryStore,
    source_ids: List[str],
) -> List[str]:
    wanted = set(source_ids)
    if not wanted:
        return []
    evidence_parts = []
    for chunk in memory_store.chunks:
        metadata = chunk.metadata or {}
        chunk_sources = metadata.get("source_ids", [])
        if isinstance(chunk_sources, str):
            chunk_sources = [chunk_sources]
        if not wanted.intersection(str(item) for item in chunk_sources):
            continue
        evidence_parts.append(chunk.content)
        for key in ("summary", "keywords"):
            value = metadata.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        evidence_parts.append(
                            str(item.get("text") or item.get("fact") or item.get("content") or "")
                        )
                    else:
                        evidence_parts.append(str(item))
            elif value:
                evidence_parts.append(str(value))
    evidence = "\n".join(part for part in evidence_parts if str(part).strip())
    if not evidence:
        return []
    supported = []
    for unit_id, unit in tracker.units.items():
        if unit_id.startswith("aggregate-") or tracker.statuses[unit_id].covered:
            continue
        if not wanted.intersection(unit.source_ids):
            continue
        if _coverage_unit_supported(unit, evidence):
            supported.append(unit_id)
    tracker.mark_covered(supported)
    return supported


def _coverage_unit_supported(unit: CoverageUnit, evidence: str) -> bool:
    subject_score = _alias_phrase_coverage(unit.source, evidence)
    object_score = _alias_phrase_coverage(unit.target, evidence)
    relation_score = _alias_phrase_coverage(unit.relation, evidence)
    qualifier_scores = [
        _alias_phrase_coverage(qualifier, evidence)
        for qualifier in unit.qualifiers
        if str(qualifier).strip()
    ]
    qualifier_complete = not qualifier_scores or min(qualifier_scores) >= 0.5
    return bool(
        subject_score >= 0.8
        and object_score >= 0.8
        and relation_score >= 0.5
        and qualifier_complete
    )


def _candidate_priority(candidate: Dict[str, Any], tracker: CaseCoverageTracker) -> float:
    unit_ids = _candidate_unit_ids(candidate, tracker)
    return float(candidate.get("priority_boost", 0.0) or 0.0) + tracker.route_priority(unit_ids)


def _semantic_answer_present(answer: str, text: str) -> bool:
    answer_dates = set(_canonical_dates(answer))
    if answer_dates and answer_dates & set(_canonical_dates(text)):
        return True
    answer_tokens = _compact_tokens(answer)
    text_tokens = _compact_tokens(text)
    if not answer_tokens:
        return False
    if len(answer_tokens) == 1 and answer_tokens[0].isdigit():
        return answer_tokens[0] in text_tokens
    return " ".join(answer_tokens) in " ".join(text_tokens)


def _canonical_dates(text: str) -> List[str]:
    dates = []
    for match in _ISO_DATE_RE.finditer(str(text or "")):
        dates.append(f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}")
    for match in _MONTH_DATE_RE.finditer(str(text or "")):
        dates.append(
            f"{int(match.group(3)):04d}-{_MONTH_NAMES[match.group(1).casefold()]:02d}-{int(match.group(2)):02d}"
        )
    return dates


_ALIAS_STOPWORDS = {
    "dr",
    "doctor",
    "orthopedic",
    "physician",
    "primary",
    "surgeon",
    "the",
    "user",
}


def _alias_phrase_coverage(phrase: str, text: str) -> float:
    if str(phrase).strip().casefold() == "user":
        actual_tokens = set(_compact_tokens(text))
        return 1.0 if actual_tokens & {"i", "me", "my", "user", "we", "our"} else 0.0
    required = [token for token in _compact_tokens(phrase) if token not in _ALIAS_STOPWORDS]
    if not required:
        required = _compact_tokens(phrase)
    actual = set(_compact_tokens(text))
    if not required:
        return 1.0
    direct = len(set(required) & actual) / len(set(required))
    required_stems = {_light_stem(token) for token in required}
    actual_stems = {_light_stem(token) for token in actual}
    stemmed = len(required_stems & actual_stems) / len(required_stems)
    return max(direct, stemmed)


def _compact_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", str(text or "").casefold().replace("_", " "))


def _light_stem(token: str) -> str:
    token = str(token or "").casefold()
    if token in {"has", "had", "have"}:
        return "have"
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    return token


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


def _build_coverage_candidates(
    graph: Dict[str, Any],
    include_compositional: bool = True,
) -> List[Dict[str, Any]]:
    entities = _entity_map(graph)
    edges = [
        edge
        for edge in graph.get("relationships", [])
        if _is_public_coverage_edge(edge, entities)
    ]
    if not edges:
        return []

    ordered_edges = _coverage_order(edges, graph)
    candidates: List[Dict[str, Any]] = []
    if include_compositional:
        candidates.extend(_build_temporal_aggregate_candidates(graph, edges, entities))
        candidates.extend(_build_relation_aggregate_candidates(graph, edges, entities))
    seen_keys = set()
    prioritized_sources = set()
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
        source_ids = _source_ids_from_golden_facts(candidate["golden_facts"])
        if any(source_id not in prioritized_sources for source_id in source_ids):
            candidate["priority_boost"] = 3.0
            prioritized_sources.update(source_ids)
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
    route_evidence = public_route_evidence(graph, route)
    route_evidence["coverage_unit_ids"] = [relationship_unit_id(edge)]
    return {
        "question": question,
        "answer": answer,
        "golden_facts": facts,
        "route": route_evidence,
        "unit_ids": [relationship_unit_id(edge)],
        "probe_kind": "relationship",
    }


def _coverage_question_variant(candidate: Dict[str, Any], variant: int) -> str:
    if variant == 0:
        return str(candidate["question"])
    if candidate.get("probe_kind") == "temporal_aggregate_count":
        aggregate = candidate.get("route", {}).get("aggregate", {})
        label = str(aggregate.get("event_label", "events"))
        bucket = str(aggregate.get("time_bucket_label", aggregate.get("time_bucket", "")))
        templates = [
            f"Count the completed {label} recorded during {bucket}.",
            f"According to memory, what is the number of completed {label} in {bucket}?",
            f"How many distinct completed {label} took place in {bucket}?",
        ]
        return templates[(variant - 1) % len(templates)]
    if candidate.get("probe_kind") == "relation_aggregate_count":
        aggregate = candidate.get("route", {}).get("aggregate", {})
        label = str(aggregate.get("event_label", "items"))
        relation = str(aggregate.get("relation", "related to"))
        templates = [
            f"Count the distinct {label} recorded across sessions for the user's '{relation}' relation.",
            f"According to memory, how many different {label} are linked to the user by '{relation}'?",
            f"What is the number of distinct {label} associated with the user through '{relation}'?",
        ]
        return templates[(variant - 1) % len(templates)]
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


_ISO_DATE_RE = re.compile(r"\b(20\d{2})[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])\b")
_MONTH_NAMES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_DATE_RE = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*|\s+)(20\d{2})\b",
    re.IGNORECASE,
)
_MEDICAL_TERMS = {
    "appointment",
    "checkup",
    "clinic",
    "doctor",
    "dr",
    "medical",
    "orthopedic",
    "physician",
    "surgeon",
}
_PROSPECTIVE_TERMS = {
    "future",
    "going to",
    "intend to",
    "next appointment",
    "plan to",
    "planned",
    "scheduled for",
    "upcoming",
    "will see",
}
_COMPLETED_TERMS = {
    "attended",
    "completed",
    "conducted",
    "diagnosed",
    "follow-up appointment",
    "had an appointment",
    "met with",
    "occurred",
    "saw dr",
    "visited",
    "went to see",
}


def _build_temporal_aggregate_candidates(
    graph: Dict[str, Any],
    edges: List[Dict[str, Any]],
    entities: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build target-free count probes from completed events sharing a time bucket."""

    chunks = {
        str(item.get("chunk_id", "")): item
        for item in graph.get("chunks", [])
        if isinstance(item, dict)
    }
    events: Dict[tuple, Dict[str, Any]] = {}
    for edge in edges:
        event = _temporal_event_from_edge(edge, entities, chunks)
        if not event:
            continue
        key = (
            event["family"],
            event["date"],
            tuple(event["source_ids"]),
        )
        existing = events.get(key)
        if existing is None or float(edge.get("weight", 1.0) or 1.0) > existing["weight"]:
            events[key] = {**event, "edge": edge, "weight": float(edge.get("weight", 1.0) or 1.0)}

    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for event in events.values():
        grouped.setdefault((event["family"], event["date"][:7]), []).append(event)

    candidates: List[Dict[str, Any]] = []
    for (family, time_bucket), group in sorted(grouped.items()):
        distinct = _distinct_aggregate_events(group)
        if len(distinct) < 2:
            continue
        relationships = [item["edge"] for item in distinct]
        nodes = []
        for edge in relationships:
            for name in (str(edge.get("source", "")), str(edge.get("target", ""))):
                if name and name not in nodes:
                    nodes.append(name)
        route = GraphRoute(
            policy="coverage_compositional",
            nodes=nodes,
            relationships=relationships,
            reason="Target-free temporal aggregation probe over completed graph events.",
        )
        facts = route_golden_facts(graph, route)
        if not facts:
            continue
        label = _aggregate_family_label(family)
        bucket_label = _month_bucket_label(time_bucket)
        composite_id = "aggregate-" + hashlib.sha1(
            f"{family}\0{time_bucket}\0".encode("utf-8")
            + "\0".join(sorted(item["date"] + ":" + item["label"] for item in distinct)).encode("utf-8")
        ).hexdigest()[:16]
        relationship_ids = [relationship_unit_id(edge) for edge in relationships]
        unit_ids = [composite_id, *relationship_ids]
        evidence = public_route_evidence(graph, route)
        evidence.update(
            {
                "probe_kind": "temporal_aggregate_count",
                "coverage_unit_ids": unit_ids,
                "aggregate": {
                    "kind": "temporal_count",
                    "event_family": family,
                    "event_label": label,
                    "time_bucket": time_bucket,
                    "time_bucket_label": bucket_label,
                    "expected_count": len(distinct),
                    "events": [
                        {
                            "label": item["label"],
                            "date": item["date"],
                            "relation": _relation_phrase(item["edge"]),
                        }
                        for item in distinct
                    ],
                },
            }
        )
        candidates.append(
            {
                "question": f"How many completed {label} are recorded during {bucket_label}?",
                "answer": str(len(distinct)),
                "golden_facts": facts,
                "route": evidence,
                "unit_ids": unit_ids,
                "probe_kind": "temporal_aggregate_count",
                "priority_boost": 12.0,
                "composite_unit": CoverageUnit(
                    unit_id=composite_id,
                    source=label.upper(),
                    relation="count_in_time_bucket",
                    target=time_bucket,
                    kind="temporal_aggregate_count",
                    source_ids=_source_ids_from_golden_facts(facts),
                    qualifiers=[bucket_label],
                    critical=True,
                    weight=3.0,
                    priority=8.0,
                    critical_reason="compositional_temporal_count",
                ),
            }
        )
    return candidates


def _build_relation_aggregate_candidates(
    graph: Dict[str, Any],
    edges: List[Dict[str, Any]],
    entities: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Probe cross-session counts of distinct entities under one user relation."""

    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source == "USER" and target:
            direction = "from_user"
            entity_name = target
        elif target == "USER" and source:
            direction = "to_user"
            entity_name = source
        else:
            continue
        relation = _relation_phrase(edge)
        entity_label = _entity_label(entities.get(entity_name, {}))
        key = (direction, relation.casefold(), entity_label)
        grouped.setdefault(key, []).append(
            {
                "edge": edge,
                "entity_name": entity_name,
                "relation": relation,
                "entity_label": entity_label,
            }
        )

    candidates: List[Dict[str, Any]] = []
    for (direction, _, entity_label), items in sorted(grouped.items()):
        distinct: Dict[str, Dict[str, Any]] = {}
        source_ids = set()
        for item in items:
            distinct.setdefault(item["entity_name"].casefold(), item)
            source_ids.update(str(value) for value in item["edge"].get("source_ids", []) if value)
        # These probes are specifically for cross-session retrieval. A single
        # source session is already audited by session-aware sibling coverage.
        if len(distinct) < 2 or len(source_ids) < 2 or len(distinct) > 30:
            continue
        members = sorted(distinct.values(), key=lambda item: item["entity_name"].casefold())
        relationships = [item["edge"] for item in members]
        nodes = ["USER"]
        for item in members:
            if item["entity_name"] not in nodes:
                nodes.append(item["entity_name"])
        route = GraphRoute(
            policy="coverage_compositional",
            nodes=nodes,
            relationships=relationships,
            reason="Target-free cross-session relation-count probe.",
        )
        facts = route_golden_facts(graph, route)
        if not facts:
            continue
        relation = members[0]["relation"]
        plural_label = _pluralize_entity_label(entity_label)
        composite_id = "aggregate-" + hashlib.sha1(
            (
                f"relation_count\0{direction}\0{relation.casefold()}\0"
                + "\0".join(item["entity_name"].casefold() for item in members)
            ).encode("utf-8")
        ).hexdigest()[:16]
        relationship_ids = [relationship_unit_id(edge) for edge in relationships]
        unit_ids = [composite_id, *relationship_ids]
        evidence = public_route_evidence(graph, route)
        evidence.update(
            {
                "probe_kind": "relation_aggregate_count",
                "coverage_unit_ids": unit_ids,
                "aggregate": {
                    "kind": "relation_count",
                    "event_label": plural_label,
                    "relation": relation,
                    "direction": direction,
                    "expected_count": len(members),
                    "events": [
                        {
                            "label": item["entity_name"],
                            "relation": relation,
                        }
                        for item in members
                    ],
                },
            }
        )
        candidates.append(
            {
                "question": (
                    f"How many distinct {plural_label} are recorded across sessions for "
                    f"the user's '{relation}' relation?"
                ),
                "answer": str(len(members)),
                "golden_facts": facts,
                "route": evidence,
                "unit_ids": unit_ids,
                "probe_kind": "relation_aggregate_count",
                "priority_boost": 9.0,
                "composite_unit": CoverageUnit(
                    unit_id=composite_id,
                    source="USER",
                    relation=f"count_distinct_{relation}",
                    target=plural_label,
                    kind="relation_aggregate_count",
                    source_ids=_source_ids_from_golden_facts(facts),
                    qualifiers=[relation],
                    critical=True,
                    weight=2.5,
                    priority=7.0,
                    critical_reason="compositional_relation_count",
                ),
            }
        )
    return candidates


def _temporal_event_from_edge(
    edge: Dict[str, Any],
    entities: Dict[str, Dict[str, Any]],
    chunks: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    source = str(edge.get("source", ""))
    target = str(edge.get("target", ""))
    source_ids = [str(item) for item in edge.get("source_ids", []) if str(item)]
    context_parts = [
        source,
        target,
        str(edge.get("description") or edge.get("relation") or ""),
        str(entities.get(source, {}).get("description", "")),
        str(entities.get(target, {}).get("description", "")),
    ]
    context_parts.extend(str(chunks.get(source_id, {}).get("content", "")) for source_id in source_ids)
    context = " ".join(context_parts)
    date = _extract_date(context)
    if not date:
        return None
    relation = _relation_phrase(edge).casefold()
    normalized_context = " ".join(context.casefold().replace("_", " ").split())
    family = _event_family(relation, normalized_context)
    if not family:
        return None
    prospective = any(term in normalized_context for term in _PROSPECTIVE_TERMS)
    completed = any(term in normalized_context for term in _COMPLETED_TERMS)
    completed = completed or any(
        term in relation
        for term in ("attended on", "completed on", "diagnosed on", "occur on", "visited on")
    )
    if not completed or (prospective and not any(term in normalized_context for term in _COMPLETED_TERMS)):
        return None
    label = source if source != "USER" else target
    if family == "medical_appointment":
        doctor_match = re.search(r"\bdr\.?\s+([a-z][a-z'-]+)", context, re.IGNORECASE)
        if doctor_match:
            label = f"Dr. {doctor_match.group(1)} appointment"
    return {
        "family": family,
        "date": date,
        "label": label,
        "source_ids": source_ids,
    }


def _event_family(relation: str, context: str) -> str:
    tokens = set(re.findall(r"[a-z0-9]+", f"{relation} {context}"))
    if tokens & _MEDICAL_TERMS:
        return "medical_appointment"
    if any(term in relation for term in ("attend", "participated", "completed", "visit")):
        return "attended_event"
    if any(term in relation for term in ("bought", "purchase")):
        return "purchase"
    return ""


def _extract_date(text: str) -> str:
    match = _ISO_DATE_RE.search(str(text or ""))
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    match = _MONTH_DATE_RE.search(str(text or ""))
    if match:
        return f"{int(match.group(3)):04d}-{_MONTH_NAMES[match.group(1).casefold()]:02d}-{int(match.group(2)):02d}"
    return ""


def _distinct_aggregate_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    distinct: Dict[tuple, Dict[str, Any]] = {}
    for event in events:
        source_key = tuple(event.get("source_ids", [])) or (event.get("label", ""),)
        distinct.setdefault((event.get("date", ""), source_key), event)
    return sorted(distinct.values(), key=lambda item: (item["date"], item["label"]))


def _aggregate_family_label(family: str) -> str:
    return {
        "medical_appointment": "medical appointments or doctor visits",
        "attended_event": "attended events",
        "purchase": "purchases",
    }.get(family, "events")


def _pluralize_entity_label(label: str) -> str:
    label = str(label or "item").strip()
    if label.endswith("s"):
        return label
    if label.endswith("y") and len(label) > 1 and label[-2].casefold() not in "aeiou":
        return label[:-1] + "ies"
    if label.endswith(("ch", "sh", "x", "z")):
        return label + "es"
    return label + "s"


def _month_bucket_label(bucket: str) -> str:
    try:
        return datetime.strptime(bucket, "%Y-%m").strftime("%B %Y")
    except ValueError:
        return bucket


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
