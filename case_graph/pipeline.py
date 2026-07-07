import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .attack_routes_cli import generate_attacks
from .attacker import FrozenLLMAttacker
from .baseline import AnswerEquivalenceJudge, GoldenFactAnswerAgent, run_baseline_sanity_test
from .defense import RetrievedMemoryAnswerAgent, SuccessPool, run_initial_defense
from .route_cli import generate_routes
from .refactoring import (
    HighPriorityBuffer,
    PromptMemoryRefactoringPolicy,
    RefactorActionDecision,
    RegressionQuestion,
    SandboxResult,
    SettlementResult,
    SimilarityActionRouter,
    prepare_regression_questions,
    run_sandbox_refactor,
    settle_grpo_rollouts,
)
from .retriever import MemoryStore
from .verifier import GoldenFactVerifier


class MemoryRefactoringPolicy(Protocol):
    def propose(
        self,
        decision: RefactorActionDecision,
        memory_store: MemoryStore,
        golden_facts: List[Dict[str, Any]],
    ):
        ...


@dataclass(frozen=True)
class AlgorithmConfig:
    tau: float
    top_k: int = 5
    min_score: float = 0.0
    regression_sample_size: int = 3
    proposal_count: int = 1
    seed: int = 0
    commit_threshold: float = 0.0
    memory_archive_dir: Optional[str] = None
    exp_name: str = "default"


def prepare_refactor_state(
    attack: Dict[str, Any],
    memory_store: MemoryStore,
    success_pool: SuccessPool,
    answer_agent: RetrievedMemoryAnswerAgent,
    judge: AnswerEquivalenceJudge,
    config: AlgorithmConfig,
    step: int,
) -> Optional[Dict[str, Any]]:
    """Prepare one refactoring state from attack (used by both offline & online).

    Returns None if initial defense succeeds (no refactor needed).
    Returns state dict if refactor needed.
    """
    question = str(attack.get("question", ""))
    answer = str(attack.get("answer") or attack.get("gold_answer") or "")
    case_id = str(attack.get("case_id", ""))
    golden_facts = attack.get("golden_facts", [])

    if not question or not answer:
        raise ValueError("Each attack must contain question and answer.")

    # Initial defense check
    initial = run_initial_defense(
        question=question,
        gold_answer=answer,
        memory_store=memory_store,
        answer_agent=answer_agent,
        judge=judge,
        success_pool=success_pool,
        top_k=config.top_k,
        min_score=config.min_score,
    )

    if initial.correct:
        return None  # No refactor needed

    # Decide action (add/merge)
    decision = SimilarityActionRouter(config.tau, config.top_k).choose_action(
        question=question,
        memory_store=memory_store,
    )

    # Prepare regression questions
    regression_questions = prepare_regression_questions(
        decision=decision,
        memory_store=memory_store,
        success_pool=success_pool,
        sample_size=config.regression_sample_size,
        seed=config.seed + step,
    )

    # Build state
    return {
        "case_id": case_id,
        "step": step,
        "question": question,
        "answer": answer,
        "golden_facts": golden_facts,
        "current_memory": {"memories": [chunk.to_dict() for chunk in memory_store.chunks]},
        "action": decision.action,
        "selected_memory_ids": decision.selected_memory_ids,
        "regression_questions": [r.to_dict() for r in regression_questions],
        "top_k": config.top_k,
        "decision": decision,
        "initial_defense": initial.to_dict(),
    }


@dataclass(frozen=True)
class AlgorithmStepResult:
    case_id: str
    step: int
    question: str
    answer: str
    golden_facts: List[Dict[str, Any]]
    status: str
    memory_store: MemoryStore
    current_memory: MemoryStore
    initial_defense: Dict[str, Any]
    decision: Optional[RefactorActionDecision] = None
    regression_questions: List[RegressionQuestion] = field(default_factory=list)
    sandbox_results: List[SandboxResult] = field(default_factory=list)
    settlement: Optional[SettlementResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "step": self.step,
            "question": self.question,
            "answer": self.answer,
            "golden_facts": self.golden_facts,
            "status": self.status,
            "current_memory": {"memories": [chunk.to_dict() for chunk in self.current_memory.chunks]},
            "initial_defense": self.initial_defense,
            "decision": self.decision.to_dict() if self.decision else None,
            "regression_questions": [item.to_dict() for item in self.regression_questions],
            "sandbox_results": [item.to_dict() for item in self.sandbox_results],
            "settlement": self.settlement.to_dict() if self.settlement else None,
        }


@dataclass(frozen=True)
class AlgorithmRunResult:
    memory_store: MemoryStore
    success_pool: SuccessPool
    high_priority_buffer: HighPriorityBuffer
    steps: List[AlgorithmStepResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "success_pool": self.success_pool.to_dict(),
            "high_priority_buffer": self.high_priority_buffer.to_dict(),
            "final_memory": {"memories": [chunk.to_dict() for chunk in self.memory_store.chunks]},
        }


@dataclass(frozen=True)
class AttackPreparationResult:
    routes: List[Dict[str, Any]]
    attacks: List[Dict[str, Any]]
    oracle_discarded: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routes": self.routes,
            "attacks": self.attacks,
            "oracle_discarded": self.oracle_discarded,
        }


class MemoryRefactoringPipeline:
    """End-to-end controller for one online memory-evolution stream."""

    def __init__(
        self,
        config: AlgorithmConfig,
        answer_agent: RetrievedMemoryAnswerAgent,
        judge: AnswerEquivalenceJudge,
        policy: Optional[MemoryRefactoringPolicy] = None,
    ):
        self.config = config
        self.answer_agent = answer_agent
        self.judge = judge
        self.policy = policy or PromptMemoryRefactoringPolicy()

    def run_step(
        self,
        attack: Dict[str, Any],
        memory_store: MemoryStore,
        success_pool: SuccessPool,
        high_priority_buffer: HighPriorityBuffer,
        step: int,
    ) -> AlgorithmStepResult:
        """Run one memory refactoring step."""
        question = str(attack.get("question", ""))
        answer = str(attack.get("answer") or attack.get("gold_answer") or "")
        case_id = str(attack.get("case_id", ""))
        golden_facts = attack.get("golden_facts", [])

        # Prepare refactor state (reusable with online training)
        state = prepare_refactor_state(
            attack=attack,
            memory_store=memory_store,
            success_pool=success_pool,
            answer_agent=self.answer_agent,
            judge=self.judge,
            config=self.config,
            step=step,
        )

        if state is None:
            # Initial defense succeeded, no refactor needed
            return AlgorithmStepResult(
                case_id=case_id,
                step=step,
                question=question,
                answer=answer,
                golden_facts=golden_facts,
                status="initial_defense_success",
                memory_store=memory_store,
                current_memory=memory_store,
                initial_defense=state.get("initial_defense") if state else {},
            )

        # Run sandbox proposals
        decision = state["decision"]
        regression_questions_dict = state["regression_questions"]
        regression_questions = [
            RegressionQuestion(
                question=r["question"],
                answer=r["answer"],
                source_memory_ids=r.get("source_memory_ids", []),
            )
            for r in regression_questions_dict
        ]

        sandbox_results = self._run_sandboxes(
            attack=attack,
            decision=decision,
            memory_store=memory_store,
            question=question,
            answer=answer,
            regression_questions=regression_questions,
        )

        # Settle and commit/rollback
        settlement = settle_grpo_rollouts(
            memory_store=memory_store,
            sandbox_results=sandbox_results,
            question=question,
            answer=answer,
            success_pool=success_pool,
            high_priority_buffer=high_priority_buffer,
            memory_archive_dir=self.config.memory_archive_dir,
            case_id=case_id,
            step=step,
            exp_name=self.config.exp_name,
            commit_threshold=self.config.commit_threshold,
        )

        return AlgorithmStepResult(
            case_id=case_id,
            step=step,
            question=question,
            answer=answer,
            golden_facts=golden_facts,
            status="refactor_committed" if settlement.committed else "refactor_rolled_back",
            memory_store=settlement.memory_store,
            current_memory=memory_store,
            initial_defense=state["initial_defense"],
            decision=decision,
            regression_questions=regression_questions,
            sandbox_results=sandbox_results,
            settlement=settlement,
        )

    def run_stream(
        self,
        attacks: List[Dict[str, Any]],
        memory_store: MemoryStore,
        success_pool: Optional[SuccessPool] = None,
        high_priority_buffer: Optional[HighPriorityBuffer] = None,
    ) -> AlgorithmRunResult:
        success_pool = success_pool or SuccessPool()
        high_priority_buffer = high_priority_buffer or HighPriorityBuffer()
        steps = []
        current_memory = memory_store
        for step_index, attack in enumerate(attacks):
            result = self.run_step(
                attack=attack,
                memory_store=current_memory,
                success_pool=success_pool,
                high_priority_buffer=high_priority_buffer,
                step=step_index,
            )
            current_memory = result.memory_store
            steps.append(result)
        return AlgorithmRunResult(
            memory_store=current_memory,
            success_pool=success_pool,
            high_priority_buffer=high_priority_buffer,
            steps=steps,
        )

    def _run_sandboxes(
        self,
        attack: Dict[str, Any],
        decision: RefactorActionDecision,
        memory_store: MemoryStore,
        question: str,
        answer: str,
        regression_questions: List[RegressionQuestion],
    ) -> List[SandboxResult]:
        results = []
        golden_facts = attack.get("golden_facts", [])
        for _ in range(max(1, self.config.proposal_count)):
            proposal = self.policy.propose(decision, memory_store, golden_facts)
            results.append(
                run_sandbox_refactor(
                    memory_store=memory_store,
                    proposal=proposal,
                    current_question=question,
                    current_answer=answer,
                    regression_questions=regression_questions,
                    answer_agent=self.answer_agent,
                    judge=self.judge,
                    top_k=self.config.top_k,
                    min_score=self.config.min_score,
                )
            )
        return results


def load_attacks(path: str | Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return list(payload.get("attacks", []))


def graph_paths(input_path: str | Path) -> List[Path]:
    path = Path(input_path)
    if path.is_dir():
        return sorted(path.glob("*.case_graph.json"))
    return [path]


def load_graphs(input_path: str | Path) -> List[Dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in graph_paths(input_path)]


def prepare_attacks_from_graphs(
    graphs: List[Dict[str, Any]],
    policies: Dict[str, object],
    policy_names: List[str],
    attacker: FrozenLLMAttacker,
    oracle_answer_agent: GoldenFactAnswerAgent,
    oracle_judge: AnswerEquivalenceJudge,
    verifier: Optional[GoldenFactVerifier] = None,
    keep_failed_verification: bool = False,
    keep_failed_oracle: bool = False,
) -> AttackPreparationResult:
    """Run stage one attack generation and stage two oracle filtering."""

    routes = generate_routes(graphs, policies, policy_names)
    attacks = generate_attacks(
        routes,
        attacker=attacker,
        verifier=verifier,
        keep_failed_verification=keep_failed_verification,
    )
    passed, discarded = run_baseline_sanity_test(
        attacks,
        answer_agent=oracle_answer_agent,
        judge=oracle_judge,
        keep_failed=keep_failed_oracle,
    )
    return AttackPreparationResult(
        routes=routes,
        attacks=passed,
        oracle_discarded=discarded,
    )
