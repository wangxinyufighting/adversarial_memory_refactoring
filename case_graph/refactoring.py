import copy
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .baseline import AnswerEquivalenceJudge, JsonClient
from .defense import RetrievedMemoryAnswerAgent, SuccessPool
from .llm import OpenAIChatClient
from .retriever import (
    FrozenBM25Retriever,
    MemoryChunk,
    MemoryStore,
    RetrievalHit,
    build_memory_retriever,
)


ADD_ACTION = "add"
MERGE_ACTION = "merge"


@dataclass(frozen=True)
class SimilarityReport:
    """当前问题 Q 与旧记忆 Top-K 的相似度报告。"""

    question: str
    top_hits: List[RetrievalHit]
    max_score: float = 0.0

    @classmethod
    def from_hits(cls, question: str, hits: List[RetrievalHit]) -> "SimilarityReport":
        # Top-K 已经按分数降序排列，因此第一个分数就是最高相似度。
        max_score = hits[0].score if hits else 0.0
        return cls(question=question, top_hits=hits, max_score=max_score)

    def to_dict(self) -> Dict[str, object]:
        return {
            "question": self.question,
            "max_score": self.max_score,
            "top_hits": [hit.to_dict() for hit in self.top_hits],
        }


@dataclass(frozen=True)
class RefactorActionDecision:
    """阶段四的分流结果：只选择动作，不修改记忆库。"""

    action: str
    question: str
    tau: float
    max_score: float
    selected_memory_ids: List[str] = field(default_factory=list)
    reason: str = ""
    similarity_report: Optional[SimilarityReport] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "action": self.action,
            "question": self.question,
            "tau": self.tau,
            "max_score": self.max_score,
            "selected_memory_ids": self.selected_memory_ids,
            "reason": self.reason,
            "similarity_report": (
                self.similarity_report.to_dict() if self.similarity_report else None
            ),
        }


class SimilarityActionRouter:
    """用配置的冻结 Retriever 相似度分数决定 Add 或 Merge。"""

    def __init__(self, tau: float, top_k: int = 5, retriever_config: Optional[Dict[str, Any]] = None):
        self.tau = tau
        self.top_k = top_k
        self.retriever_config = dict(retriever_config or {})

    def compute_similarity(
        self,
        question: str,
        memory_store: MemoryStore,
        retriever: Optional[Any] = None,
    ) -> SimilarityReport:
        retriever = retriever or build_memory_retriever(self.retriever_config, memory_store)
        hits = retriever.retrieve(question, top_k=self.top_k)
        return SimilarityReport.from_hits(question=question, hits=hits)

    def choose_action(
        self,
        question: str,
        memory_store: MemoryStore,
        retriever: Optional[Any] = None,
    ) -> RefactorActionDecision:
        report = self.compute_similarity(question, memory_store, retriever)
        if report.max_score < self.tau:
            return RefactorActionDecision(
                action=ADD_ACTION,
                question=question,
                tau=self.tau,
                max_score=report.max_score,
                reason="最高相似度低于阈值，视为全新知识。",
                similarity_report=report,
            )

        # Merge 只选择达到阈值的旧 Chunk；真正合并由后续 Policy 执行。
        selected_ids = [hit.memory_id for hit in report.top_hits if hit.score >= self.tau]
        return RefactorActionDecision(
            action=MERGE_ACTION,
            question=question,
            tau=self.tau,
            max_score=report.max_score,
            selected_memory_ids=selected_ids,
            reason="存在相似旧记忆，交给重构策略做压缩合并。",
            similarity_report=report,
        )


@dataclass(frozen=True)
class RefactorProposal:
    """Policy 给出的具体编辑方案；沙盒只执行这个方案。"""

    action: str
    new_chunks: List[MemoryChunk]
    remove_memory_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "new_chunks": [chunk.to_dict() for chunk in self.new_chunks],
            "remove_memory_ids": self.remove_memory_ids,
            "metadata": self.metadata,
        }


class PromptMemoryRefactoringPolicy:
    """先用 prompt 版本实现 Policy，后续 GRPO 训练时替换采样逻辑即可。"""

    def __init__(self, client: Optional[JsonClient] = None, max_output_tokens: int = 800):
        self.client = client
        self.max_output_tokens = max_output_tokens

    def propose(
        self,
        decision: RefactorActionDecision,
        memory_store: MemoryStore,
        golden_facts: List[Dict[str, Any]],
    ) -> RefactorProposal:
        old_chunks = _find_chunks(memory_store, decision.selected_memory_ids)
        response = self._client().complete_json(
            system_prompt=(
                "You are a memory refactoring policy. Return JSON only with a chunks list. "
                "Each chunk needs memory_id and content."
            ),
            user_prompt=json.dumps(
                {
                    "action": decision.action,
                    "question": decision.question,
                    "golden_facts": golden_facts,
                    "old_chunks": [chunk.to_dict() for chunk in old_chunks],
                    "instruction": self._instruction(decision.action, len(old_chunks)),
                },
                ensure_ascii=False,
            ),
            max_tokens=self.max_output_tokens,
        )
        chunks = [
            MemoryChunk.from_dict(item, fallback_id=f"{decision.action}_{index}")
            for index, item in enumerate(response.get("chunks", []))
        ]
        if decision.action == ADD_ACTION:
            chunks = chunks[:1]
        elif old_chunks:
            chunks = chunks[: len(old_chunks)]
        return RefactorProposal(
            action=decision.action,
            new_chunks=chunks,
            remove_memory_ids=decision.selected_memory_ids if decision.action == MERGE_ACTION else [],
            metadata={"policy": "prompt"},
        )

    def _client(self) -> JsonClient:
        if self.client is None:
            self.client = OpenAIChatClient.from_env()
        return self.client

    def _instruction(self, action: str, old_chunk_count: int) -> str:
        if action == ADD_ACTION:
            return "Only use golden_facts to write one concise new memory chunk."
        return (
            "Merge old_chunks with golden_facts into concise new chunks. "
            f"Return no more than {old_chunk_count} chunks and preserve answer-critical facts."
        )


@dataclass(frozen=True)
class RegressionQuestion:
    question: str
    answer: str
    source_memory_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "source_memory_ids": self.source_memory_ids,
        }


@dataclass(frozen=True)
class QuestionTestResult:
    question: str
    gold_answer: str
    correct: bool
    retrieved_memories: List[RetrievalHit]
    answer_result: Dict[str, Any]
    judge: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "gold_answer": self.gold_answer,
            "correct": self.correct,
            "retrieved_memories": [hit.to_dict() for hit in self.retrieved_memories],
            "answer_result": self.answer_result,
            "judge": self.judge,
        }


@dataclass(frozen=True)
class SandboxEvaluation:
    current_test: QuestionTestResult
    regression_tests: List[QuestionTestResult] = field(default_factory=list)

    @property
    def regression_accuracy(self) -> float:
        if not self.regression_tests:
            return 1.0
        correct = sum(1 for result in self.regression_tests if result.correct)
        return correct / len(self.regression_tests)

    @property
    def failed_regression_count(self) -> int:
        return sum(1 for result in self.regression_tests if not result.correct)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_test": self.current_test.to_dict(),
            "regression_tests": [result.to_dict() for result in self.regression_tests],
            "regression_accuracy": self.regression_accuracy,
            "failed_regression_count": self.failed_regression_count,
        }


@dataclass(frozen=True)
class RewardWeights:
    current_correct: float = 3.0
    current_wrong: float = -3.0
    completeness: float = 2.0
    completeness_missing: float = 2.0
    grounded: float = 1.0
    regression_accuracy: float = 1.25
    regression_failure: float = 2.0
    chunk_count: float = 0.2
    tokens_per_100: float = 0.15
    duplicate: float = 0.5
    answer_only: float = 1.5
    raw_copy: float = 0.5

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "RewardWeights":
        config = dict(config or {})
        values = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in config:
                values[field_name] = float(config[field_name])
        return cls(**values)


@dataclass(frozen=True)
class RewardResult:
    reward: float
    parts: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {"reward": self.reward, "parts": self.parts}


@dataclass(frozen=True)
class SandboxResult:
    proposal: RefactorProposal
    temp_memory: MemoryStore
    evaluation: SandboxEvaluation
    reward: RewardResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "temp_memory": {"memories": [chunk.to_dict() for chunk in self.temp_memory.chunks]},
            "evaluation": self.evaluation.to_dict(),
            "reward": self.reward.to_dict(),
        }


@dataclass
class HighPriorityBuffer:
    """Questions that failed all sandbox proposals and should be retried later."""

    attacks: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "HighPriorityBuffer":
        buffer_path = Path(path)
        if not buffer_path.exists():
            return cls()
        payload = json.loads(buffer_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return cls(attacks=payload)
        return cls(attacks=list(payload.get("attacks", [])))

    def add(
        self,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.attacks.append(
            {
                "question": question,
                "answer": answer,
                "metadata": dict(metadata or {}),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"attacks": self.attacks}

    def save(self, path: str | Path) -> None:
        buffer_path = Path(path)
        buffer_path.parent.mkdir(parents=True, exist_ok=True)
        buffer_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class SettlementResult:
    """Commit or rollback decision after comparing multiple GRPO rollouts."""

    committed: bool
    memory_store: MemoryStore
    selected_result: Optional[SandboxResult]
    all_rewards: List[float]
    reason: str
    archived_memory_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "committed": self.committed,
            "selected_result": self.selected_result.to_dict() if self.selected_result else None,
            "all_rewards": self.all_rewards,
            "reason": self.reason,
            "archived_memory_path": self.archived_memory_path,
        }


def settle_grpo_rollouts(
    memory_store: MemoryStore,
    sandbox_results: List[SandboxResult],
    question: str,
    answer: str,
    success_pool: Optional[SuccessPool] = None,
    high_priority_buffer: Optional[HighPriorityBuffer] = None,
    memory_archive_dir: Optional[str | Path] = None,
    case_id: str = "",
    step: str | int = "",
    exp_name: str = "",
    commit_threshold: float = 0.0,
) -> SettlementResult:
    """Pick the best rollout for environment transition; GRPO still trains on all rewards."""

    if not sandbox_results:
        if high_priority_buffer is not None:
            high_priority_buffer.add(question, answer, {"reason": "no_sandbox_results"})
        return SettlementResult(
            committed=False,
            memory_store=memory_store,
            selected_result=None,
            all_rewards=[],
            reason="No sandbox result was available.",
        )

    selected = max(sandbox_results, key=lambda item: item.reward.reward)
    rewards = [item.reward.reward for item in sandbox_results]
    if selected.reward.reward <= commit_threshold:
        if high_priority_buffer is not None:
            high_priority_buffer.add(
                question,
                answer,
                {
                    "reason": "best_reward_not_positive",
                    "best_reward": selected.reward.reward,
                    "all_rewards": rewards,
                },
            )
        return SettlementResult(
            committed=False,
            memory_store=memory_store,
            selected_result=selected,
            all_rewards=rewards,
            reason="Rollback: the best rollout reward is not positive.",
        )

    committed_memory = copy.deepcopy(selected.temp_memory)
    archived_path = None
    if memory_archive_dir is not None:
        archived_path = archive_memory_snapshot(
            memory_store=memory_store,
            archive_dir=memory_archive_dir,
            case_id=case_id,
            step=step,
            exp_name=exp_name,
            metadata={
                "event": "pre_commit",
                "question": question,
                "answer": answer,
                "selected_reward": selected.reward.reward,
                "all_rewards": rewards,
            },
        )
    if success_pool is not None:
        success_pool.add(
            question=question,
            answer=answer,
            memory_ids=[chunk.memory_id for chunk in selected.proposal.new_chunks],
            metadata={
                "stage": "sandbox_commit",
                "reward": selected.reward.reward,
                "all_rewards": rewards,
            },
        )
    return SettlementResult(
        committed=True,
        memory_store=committed_memory,
        selected_result=selected,
        all_rewards=rewards,
        reason="Commit: the best rollout reward is positive.",
        archived_memory_path=archived_path,
    )


def archive_memory_snapshot(
    memory_store: MemoryStore,
    archive_dir: str | Path,
    case_id: str = "",
    step: str | int = "",
    exp_name: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a deepcopy of the old Mt before it is replaced by Mt+1."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name_parts = [
        _safe_name(exp_name or "exp"),
        _safe_name(case_id or "case"),
        f"step-{_safe_name(str(step or 'unknown'))}",
        timestamp,
    ]
    archive_path = Path(archive_dir) / ("__".join(name_parts) + ".memory.json")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = copy.deepcopy(memory_store)
    payload = {
        "metadata": {
            "case_id": case_id,
            "step": step,
            "exp_name": exp_name,
            "created_at_utc": timestamp,
            **dict(metadata or {}),
        },
        "memories": [chunk.to_dict() for chunk in snapshot.chunks],
    }
    archive_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(archive_path)


def build_sandbox_memory(
    memory_store: MemoryStore,
    proposal: RefactorProposal,
    current_question: str = "",
) -> MemoryStore:
    """复制 Mt，并在副本上执行 Add/Merge，得到带血缘的 Mtemp。"""

    temp_memory = copy.deepcopy(memory_store)
    remove_ids = set(proposal.remove_memory_ids)
    inherited_questions = _linked_questions_for_chunks(temp_memory, remove_ids)
    if current_question and current_question not in inherited_questions:
        inherited_questions.append(current_question)

    if remove_ids:
        temp_memory.chunks = [chunk for chunk in temp_memory.chunks if chunk.memory_id not in remove_ids]

    new_chunks = copy.deepcopy(proposal.new_chunks)
    if current_question:
        for chunk in new_chunks:
            # Add 绑定当前题；Merge 继承被替换旧 Chunk 的题目血缘并绑定当前题。
            source_questions = inherited_questions if proposal.action == MERGE_ACTION else [current_question]
            for question in source_questions:
                chunk.bind_question(question)
    temp_memory.chunks.extend(new_chunks)
    return temp_memory


def prepare_regression_questions(
    decision: RefactorActionDecision,
    memory_store: MemoryStore,
    success_pool: SuccessPool,
    sample_size: int = 3,
    seed: int = 0,
) -> List[RegressionQuestion]:
    """Add 随机抽全局成功题，Merge 精准抽命中旧 Chunk 绑定题。"""

    answer_by_question = {
        str(item.get("question", "")): str(item.get("answer", ""))
        for item in success_pool.successes
        if item.get("question")
    }

    if decision.action == ADD_ACTION:
        candidates = [
            RegressionQuestion(
                question=str(item.get("question", "")),
                answer=str(item.get("answer", "")),
                source_memory_ids=[str(memory_id) for memory_id in item.get("memory_ids", [])],
            )
            for item in success_pool.successes
            if item.get("question") and item.get("answer")
        ]
        sample_count = min(sample_size, len(candidates))
        return random.Random(seed).sample(candidates, sample_count)

    questions = _linked_questions_for_chunks(memory_store, decision.selected_memory_ids)
    return [
        RegressionQuestion(question=question, answer=answer_by_question[question])
        for question in questions
        if question in answer_by_question
    ]


def run_sandbox_evaluation(
    temp_memory: MemoryStore,
    current_question: str,
    current_answer: str,
    regression_questions: List[RegressionQuestion],
    answer_agent: RetrievedMemoryAnswerAgent,
    judge: AnswerEquivalenceJudge,
    top_k: int = 5,
    min_score: float = 0.0,
    retriever_config: Optional[Dict[str, Any]] = None,
) -> SandboxEvaluation:
    """在 Mtemp 上测试当前题和历史回归题。"""

    current_test = _test_question(
        temp_memory, current_question, current_answer, answer_agent, judge, top_k, min_score, retriever_config
    )
    regression_tests = [
        _test_question(
            temp_memory, item.question, item.answer, answer_agent, judge, top_k, min_score, retriever_config
        )
        for item in regression_questions
    ]
    return SandboxEvaluation(current_test=current_test, regression_tests=regression_tests)


def compute_reward(
    proposal: RefactorProposal,
    evaluation: SandboxEvaluation,
    weights: RewardWeights = RewardWeights(),
) -> RewardResult:
    """Reward compact but complete memory.

    The reward is gated: a chunk cannot get a high score merely because the
    answer string appears in retrieved memory.  Completeness diagnostics are
    optional for legacy callers; when absent, a correct current answer is
    treated as complete to preserve backwards compatibility.
    """

    diagnostics = evaluation.current_test.judge or {}
    complete = diagnostics.get("complete")
    if complete is None:
        complete = evaluation.current_test.correct
    grounded_score = _float_diagnostic(diagnostics, "grounding_score", default=1.0 if complete else 0.0)
    duplicate_score = _float_diagnostic(diagnostics, "duplicate_score", default=0.0)
    raw_copy_ratio = _float_diagnostic(diagnostics, "raw_copy_ratio", default=0.0)
    answer_only = bool(diagnostics.get("answer_only", False))
    current_part = weights.current_correct if evaluation.current_test.correct else weights.current_wrong
    completeness_part = weights.completeness if complete else -weights.completeness_missing
    grounded_part = weights.grounded * grounded_score
    regression_part = weights.regression_accuracy * evaluation.regression_accuracy
    failure_part = -weights.regression_failure * evaluation.failed_regression_count
    chunk_part = -weights.chunk_count * len(proposal.new_chunks)
    token_count = sum(len(_reward_tokens(chunk.content)) for chunk in proposal.new_chunks)
    length_part = -weights.tokens_per_100 * (token_count / 100.0)
    duplicate_part = -weights.duplicate * duplicate_score
    answer_only_part = -weights.answer_only if answer_only else 0.0
    raw_copy_part = -weights.raw_copy * raw_copy_ratio
    parts = {
        "current": current_part,
        "completeness": completeness_part,
        "grounded": grounded_part,
        "regression_reward": regression_part,
        "regression_failure": failure_part,
        "chunk_count": chunk_part,
        "length": length_part,
        "duplicate": duplicate_part,
        "answer_only": answer_only_part,
        "raw_copy": raw_copy_part,
    }
    reward = sum(parts.values())
    if not evaluation.current_test.correct:
        reward = min(reward, 0.0)
    elif not complete:
        reward = min(reward, 0.5)
    return RewardResult(reward=reward, parts=parts)


def run_sandbox_refactor(
    memory_store: MemoryStore,
    proposal: RefactorProposal,
    current_question: str,
    current_answer: str,
    regression_questions: List[RegressionQuestion],
    answer_agent: RetrievedMemoryAnswerAgent,
    judge: AnswerEquivalenceJudge,
    top_k: int = 5,
    min_score: float = 0.0,
    reward_weights: RewardWeights = RewardWeights(),
    retriever_config: Optional[Dict[str, Any]] = None,
) -> SandboxResult:
    temp_memory = build_sandbox_memory(memory_store, proposal, current_question=current_question)
    evaluation = run_sandbox_evaluation(
        temp_memory=temp_memory,
        current_question=current_question,
        current_answer=current_answer,
        regression_questions=regression_questions,
        answer_agent=answer_agent,
        judge=judge,
        top_k=top_k,
        min_score=min_score,
        retriever_config=retriever_config,
    )
    reward = compute_reward(proposal, evaluation, reward_weights)
    return SandboxResult(
        proposal=proposal,
        temp_memory=temp_memory,
        evaluation=evaluation,
        reward=reward,
    )


def _test_question(
    memory_store: MemoryStore,
    question: str,
    gold_answer: str,
    answer_agent: RetrievedMemoryAnswerAgent,
    judge: AnswerEquivalenceJudge,
    top_k: int,
    min_score: float,
    retriever_config: Optional[Dict[str, Any]] = None,
) -> QuestionTestResult:
    retriever = build_memory_retriever(retriever_config, memory_store)
    hits = retriever.retrieve(question, top_k=top_k, min_score=min_score)
    answer_result = answer_agent.answer(question, hits)
    judge_result = judge.judge(
        question=question,
        gold_answer=gold_answer,
        candidate_answer=str(answer_result.get("answer", "")),
    )
    return QuestionTestResult(
        question=question,
        gold_answer=gold_answer,
        correct=bool(judge_result.get("correct", False)),
        retrieved_memories=hits,
        answer_result=answer_result,
        judge=judge_result,
    )


def _float_diagnostic(diagnostics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(diagnostics.get(key, default))
    except (TypeError, ValueError):
        return default


def _reward_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", str(text or "").casefold())


def _find_chunks(memory_store: MemoryStore, memory_ids: Iterable[str]) -> List[MemoryChunk]:
    wanted = set(memory_ids)
    return [chunk for chunk in memory_store.chunks if chunk.memory_id in wanted]


def _linked_questions_for_chunks(memory_store: MemoryStore, memory_ids: Iterable[str]) -> List[str]:
    questions: List[str] = []
    for chunk in _find_chunks(memory_store, memory_ids):
        for question in chunk.linked_questions:
            if question not in questions:
                questions.append(question)
    return questions


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    return safe.strip("-_") or "unknown"
