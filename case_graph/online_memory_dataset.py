"""Online memory dataset for GRPO training.

Generates memory refactoring states on-demand during training by:
1. Selecting a case graph (round-robin)
2. Generating an adversarial attack from the graph
3. Preparing a refactoring state if initial defense fails
4. Returning verl-compatible training row
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.utils.data

from .attacker import FrozenLLMAttacker
from .baseline import AnswerEquivalenceJudge
from .coverage import CaseCoverageTracker, CoverageAwareRouteScheduler
from .defense import RetrievedMemoryAnswerAgent, SuccessPool
from .grpo_adapter import build_verl_row_online, compute_score
from .llm import OpenAIChatClient
from .pipeline import AlgorithmConfig, prepare_refactor_state
from .refactoring import HighPriorityBuffer, RefactorProposal, build_sandbox_memory
from .retriever import MemoryChunk, MemoryStore, retriever_config_from_mapping
from .routing import RandomWalkRoutingPolicy

logger = logging.getLogger(__name__)


def build_attacker_from_config(config: Dict[str, Any]) -> FrozenLLMAttacker:
    """Create the frozen attacker, honoring online GRPO config before env defaults."""
    model = config.get("attacker_llm")
    api_base = config.get("attacker_api_base")
    api_key = config.get("attacker_api_key")

    if model or api_base or api_key:
        client = OpenAIChatClient(
            model=str(model or os.environ.get("LOCAL_MODEL") or "local-model"),
            api_key=str(
                api_key
                or os.environ.get("LOCAL_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or "dummy-key"
            ),
            base_url=str(api_base or os.environ.get("LOCAL_API_BASE_URL") or "http://localhost:8000/v1").rstrip("/"),
            timeout=int(config.get("attacker_timeout", os.environ.get("CASE_GRAPH_TIMEOUT", 120))),
        )
        logger.info("Using configured attacker LLM %s at %s", client.model, client.base_url)
        return FrozenLLMAttacker(
            client=client,
            max_output_tokens=int(config.get("attacker_max_output_tokens", 700)),
        )

    return FrozenLLMAttacker(
        max_output_tokens=int(config.get("attacker_max_output_tokens", 700)),
    )


@dataclass
class CaseMemoryState:
    """Per-case persistent memory state across episodes."""

    case_id: str
    graph: Dict[str, Any]
    memory_store: MemoryStore
    success_pool: SuccessPool
    high_priority_buffer: HighPriorityBuffer
    episode_count: int = 0
    coverage_tracker: Optional[CaseCoverageTracker] = None
    ready: bool = False
    completed: bool = False
    incomplete: bool = False


class OnlineMemoryEnvironment:
    """Manages memory states for all cases and generates episodes on-demand."""

    def __init__(
        self,
        graphs: List[Dict[str, Any]],
        routing_policy: RandomWalkRoutingPolicy,
        attacker: FrozenLLMAttacker,
        answer_agent: RetrievedMemoryAnswerAgent,
        judge: AnswerEquivalenceJudge,
        config: AlgorithmConfig,
        initial_memory_dir: Optional[str] = None,
    ):
        self.graphs = graphs
        self.routing_policy = routing_policy
        self.route_scheduler = CoverageAwareRouteScheduler(
            routing_policy,
            candidate_attempts=max(4, int(config.coverage_route_attempts)),
            random_exploration_ratio=config.coverage_random_exploration_ratio,
        )
        self.attacker = attacker
        self.answer_agent = answer_agent
        self.judge = judge
        self.config = config

        # Initialize per-case memory states
        self.case_states: Dict[str, CaseMemoryState] = {}
        for graph in graphs:
            case_id = str(graph.get("case_id", "unknown"))
            self.case_states[case_id] = self._init_case_state(graph, initial_memory_dir)

    def _init_case_state(
        self, graph: Dict[str, Any], initial_memory_dir: Optional[str]
    ) -> CaseMemoryState:
        """Initialize memory state for a case."""
        case_id = str(graph.get("case_id", "unknown"))

        if initial_memory_dir:
            memory_path = Path(initial_memory_dir) / f"{case_id}.json"
            if memory_path.exists():
                memory_store = MemoryStore.load(memory_path)
            else:
                memory_store = MemoryStore()
        else:
            memory_store = MemoryStore()

        return CaseMemoryState(
            case_id=case_id,
            graph=graph,
            memory_store=memory_store,
            success_pool=SuccessPool(),
            high_priority_buffer=HighPriorityBuffer(),
            episode_count=0,
            coverage_tracker=CaseCoverageTracker.from_graph(graph),
        )

    def generate_episode(
        self, case_id: str, episode: int, seed: int
    ) -> Optional[Dict[str, Any]]:
        """Generate one training state from a case.

        Returns None if initial defense succeeds (no refactor needed).
        """
        case_state = self.case_states[case_id]

        try:
            # 1. Generate attack from graph
            route = self.route_scheduler.select_route(
                case_state.graph,
                tracker=case_state.coverage_tracker,
                seed=seed,
            )
            attack = self.attacker.generate(case_state.graph, route)
            coverage_unit_ids = (
                case_state.coverage_tracker.unit_ids_for_route(route)
                if case_state.coverage_tracker is not None
                else []
            )
            attack_dict = {
                "case_id": case_id,
                "question": attack.question,
                "answer": attack.answer,
                "golden_facts": attack.golden_facts,
                "route_evidence": attack.route,
            }

        except Exception as e:
            logger.warning(f"Attack generation failed for case {case_id} episode {episode}: {e}")
            return None

        # 2. Prepare refactor state (includes initial defense check)
        try:
            state = prepare_refactor_state(
                attack=attack_dict,
                memory_store=case_state.memory_store,
                success_pool=case_state.success_pool,
                answer_agent=self.answer_agent,
                judge=self.judge,
                config=self.config,
                step=episode,
            )
        except ValueError as exc:
            logger.warning(
                "Skipping invalid attack for case %s episode %s: %s",
                case_id,
                episode,
                exc,
            )
            return None

        if state is None:
            # Initial defense succeeded, no refactor needed
            if case_state.coverage_tracker is not None:
                certification = case_state.coverage_tracker.is_coverage_ready(
                    self.config.coverage_threshold,
                    self.config.critical_coverage_threshold,
                )
                case_state.coverage_tracker.record(
                    coverage_unit_ids,
                    success=True,
                    certification=certification,
                )
                case_state.ready = (
                    case_state.coverage_tracker.is_coverage_ready(
                        self.config.coverage_threshold,
                        self.config.critical_coverage_threshold,
                    )
                    and case_state.coverage_tracker.consecutive_certification_passes
                    >= self.config.training_probe_window
                )
            return None

        # 3. Add UID for GRPO grouping
        state["uid"] = f"{case_id}_ep{episode}"
        state["episode"] = episode
        state["coverage_unit_ids"] = coverage_unit_ids
        if case_state.coverage_tracker is not None:
            route_weight = case_state.coverage_tracker.route_weight(coverage_unit_ids)
            pending_weight = case_state.coverage_tracker.pending_weight(coverage_unit_ids)
            state["coverage_route_weight"] = route_weight
            state["coverage_pending_weight"] = pending_weight
            state["coverage_critical_pending_weight"] = pending_weight
            state["coverage_before"] = case_state.coverage_tracker.structural_coverage()

        return state

    def commit_memory_update(
        self,
        uid: str,
        best_proposal: RefactorProposal,
        current_question: str,
        current_answer: str = "",
        coverage_unit_ids: Optional[List[str]] = None,
        coverage_success: bool = True,
    ) -> None:
        """Apply winning proposal to M_t after training step."""
        # Parse case_id from uid
        case_id = uid.split("_ep")[0]
        case_state = self.case_states.get(case_id)

        if case_state is None:
            logger.warning(f"Cannot commit: case {case_id} not found in environment")
            return

        # Build sandbox memory (apply proposal)
        temp_memory = build_sandbox_memory(
            case_state.memory_store, best_proposal, current_question=current_question
        )

        # Commit to M_t
        case_state.memory_store = temp_memory

        # Update success pool with new chunks
        if current_question:
            case_state.success_pool.add(
                question=current_question,
                answer=current_answer,
                memory_ids=[chunk.memory_id for chunk in best_proposal.new_chunks],
                metadata={"committed_from_online_training": True},
            )
        if case_state.coverage_tracker is not None and coverage_unit_ids:
            case_state.coverage_tracker.record(coverage_unit_ids, success=coverage_success)

    def record_rollback(
        self,
        uid: str,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Track failed/rolled-back online updates for later replay."""
        case_id = uid.split("_ep")[0]
        case_state = self.case_states.get(case_id)
        if case_state is None:
            logger.warning(f"Cannot record rollback: case {case_id} not found in environment")
            return
        case_state.high_priority_buffer.add(question, answer, metadata)


class OnlineMemoryDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that generates memory refactoring states on-demand.

    Compatible with verl's data loading pipeline.
    """

    def __init__(
        self,
        graph_files: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        initial_memory_dir: Optional[str] = None,
        # verl standard arguments (ignored, but accepted for compatibility)
        data_files: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        processor: Optional[Any] = None,
        max_samples: int = -1,
    ):
        # If called by verl with data_files, load config from the file
        if data_files is not None and graph_files is None:
            # verl passes the config file path as data_files[0]
            if isinstance(data_files, str):
                config_file = Path(data_files)
            elif isinstance(data_files, list) and len(data_files) > 0:
                config_file = Path(data_files[0])
                
            if config_file.exists():
                logger.info(f"Loading dataset config from {config_file}")
                with open(config_file) as f:
                    dataset_config = json.load(f)
                graph_files = dataset_config["graph_files"]
                config = dataset_config["config"]
                initial_memory_dir = dataset_config.get("initial_memory_dir")
            else:
                raise ValueError(f"Config file not found: {config_file}")

        if graph_files is None or config is None:
            raise ValueError("graph_files and config must be provided")

        self.config = config
        self.tokenizer = tokenizer

        # Load and validate graphs
        self.graphs = self._load_and_validate_graphs(graph_files)
        if not self.graphs:
            raise ValueError("No valid graphs found")

        # Initialize environment
        self.env = OnlineMemoryEnvironment(
            graphs=self.graphs,
            routing_policy=RandomWalkRoutingPolicy(
                seed=config.get("seed", 42),
                max_steps=config.get("routing_max_steps", 4),
                min_nodes=config.get("routing_min_nodes", 1),
                attempts=config.get("routing_attempts", 12),
            ),
            attacker=build_attacker_from_config(config),
            answer_agent=RetrievedMemoryAnswerAgent(),
            judge=AnswerEquivalenceJudge(
                use_llm=_config_bool(config.get("initial_defense_judge_use_llm"), False)
            ),
            config=AlgorithmConfig(
                tau=config.get("tau", 0.55),
                top_k=config.get("top_k", 8),
                top_k_points=config.get("top_k_points", config.get("retriever_top_k_points", 32)),
                regression_sample_size=config.get("regression_sample_size", 12),
                seed=config.get("seed", 42),
                commit_threshold=config.get("commit_threshold", 1.0),
                retriever_config=retriever_config_from_mapping(config),
                reward_config=dict(config.get("reward", config.get("reward_config", {})) or {}),
                initial_defense_use_llm=_config_bool(config.get("initial_defense_use_llm"), False),
                coverage_threshold=float(config.get("coverage_threshold", 0.98)),
                critical_coverage_threshold=float(config.get("critical_coverage_threshold", 1.0)),
                training_probe_window=int(config.get("training_probe_window", 12)),
                coverage_route_attempts=int(config.get("coverage_route_attempts", 16)),
                coverage_random_exploration_ratio=float(
                    config.get("coverage_random_exploration_ratio", 0.15)
                ),
            ),
            initial_memory_dir=initial_memory_dir,
        )

        # Episode tracking
        self.case_rotation = 0
        self.global_seed = config.get("seed", 42)
        self.episodes_per_case = config.get(
            "max_questions_per_case", config.get("episodes_per_case", 200)
        )
        self.max_attack_attempts = config.get("max_attack_attempts", 20)
        self.routing_seed_offset = config.get("routing_seed_offset", 0)
        self.commit_threshold = config.get("commit_threshold", 1.0)
        self.output_dir = Path(config["output_dir"]) if config.get("output_dir") else None
        self.checkpoint_interval = int(config.get("checkpoint_interval", 500))
        self.memory_save_interval = int(config.get("memory_save_interval", 1))
        self.batch_commit_count = 0
        self.memory_trajectory_enabled = _config_bool(config.get("memory_trajectory_enabled", True))
        self.memory_trajectory_dir = str(config.get("memory_trajectory_dir", "memory_trajectory"))
        self.trajectory_step = 0

    def _load_and_validate_graphs(self, graph_files: List[str]) -> List[Dict[str, Any]]:
        """Load graphs and filter out invalid ones."""
        graphs = []
        for file_path in graph_files:
            try:
                graph = json.loads(Path(file_path).read_text(encoding="utf-8"))
                if self._validate_graph(graph):
                    graphs.append(graph)
                else:
                    logger.warning(f"Skipping invalid graph: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load graph {file_path}: {e}")
        return graphs

    @staticmethod
    def _validate_graph(graph: Dict[str, Any]) -> bool:
        """Check if graph has sufficient structure for attack generation."""
        entities = len(graph.get("entities", []))
        relationships = len(graph.get("relationships", []))
        return entities >= 3 and relationships >= 2

    def __len__(self):
        """Virtual length for dataloader."""
        return len(self.graphs) * self.episodes_per_case

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate one state on-demand.

        Uses round-robin case selection and retries if initial defense succeeds.
        """
        total_attempts = max(1, len(self.graphs) * self.max_attack_attempts)
        last_case_id = "unknown"

        for attempt in range(total_attempts):
            active_graphs = [
                graph
                for graph in self.graphs
                if not self.env.case_states[str(graph.get("case_id", "unknown"))].completed
                and not self.env.case_states[str(graph.get("case_id", "unknown"))].incomplete
            ]
            if not active_graphs:
                raise RuntimeError("All online memory cases are complete or exhausted.")

            # Round-robin selection over unresolved cases.
            case_idx = self.case_rotation % len(active_graphs)
            self.case_rotation += 1

            case_graph = active_graphs[case_idx]
            case_id = str(case_graph.get("case_id", "unknown"))
            last_case_id = case_id
            case_state = self.env.case_states[case_id]
            if case_state.episode_count >= self.episodes_per_case:
                case_state.incomplete = True
                continue
            episode = case_state.episode_count
            case_state.episode_count += 1

            seed = (
                self.global_seed
                + self.routing_seed_offset
                + idx
                + episode * 1000
                + attempt
            )
            state = self.env.generate_episode(case_id, episode, seed)

            if state is not None:
                # Convert to verl format
                return build_verl_row_online(state, index=idx)

        raise RuntimeError(
            f"Failed to generate a refactor state after {total_attempts} attempts; "
            f"last case was {last_case_id}."
        )

    def commit_memory_update(
        self,
        uid: str,
        best_proposal: RefactorProposal,
        current_question: str,
        current_answer: str = "",
    ) -> None:
        """Commit best proposal to M_t after training step."""
        self.env.commit_memory_update(uid, best_proposal, current_question, current_answer)

    def on_batch_end(self, batch: Any, tokenizer: Optional[Any] = None) -> None:
        """Commit or roll back the best rollout for each online state.

        verl calls this hook after policy update. The hook reads rollout text,
        rewards, and original state metadata from the training batch/TransferQueue,
        then applies only the best positive proposal to the persistent per-case
        memory store.
        """
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            logger.warning("Skipping online memory commit: tokenizer is unavailable")
            return

        records = self._extract_batch_records(batch, tokenizer)
        if not records:
            logger.warning("Skipping online memory commit: no rollout records found")
            return

        self._ensure_initial_trajectory_snapshot()

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            uid = record.get("uid")
            if uid:
                grouped.setdefault(uid, []).append(record)

        for uid, items in grouped.items():
            best = max(items, key=lambda item: item["reward"])
            state = best["state"]
            question = str(state.get("question", ""))
            answer = str(state.get("answer", ""))
            rewards = [item["reward"] for item in items]

            if best["reward"] <= self.commit_threshold:
                self.env.record_rollback(
                    uid,
                    question,
                    answer,
                    {
                        "reason": "best_reward_not_positive",
                        "best_reward": best["reward"],
                        "all_rewards": rewards,
                    },
                )
                self._record_coverage_result(state, success=False)
                logger.info("Rolled back %s: best reward %.3f", uid, best["reward"])
                self._save_memory_trajectory_step(
                    event="rollback",
                    uid=uid,
                    state=state,
                    rewards=rewards,
                    best_reward=best["reward"],
                    reason="best_reward_not_positive",
                )
                continue

            try:
                proposal = self._proposal_from_rollout(best["solution"], state)
            except Exception as exc:
                self.env.record_rollback(
                    uid,
                    question,
                    answer,
                    {
                        "reason": "proposal_parse_failed",
                        "best_reward": best["reward"],
                        "all_rewards": rewards,
                        "error": str(exc),
                    },
                )
                self._record_coverage_result(state, success=False)
                logger.warning("Failed to parse winning proposal for %s: %s", uid, exc)
                self._save_memory_trajectory_step(
                    event="rollback",
                    uid=uid,
                    state=state,
                    rewards=rewards,
                    best_reward=best["reward"],
                    reason="proposal_parse_failed",
                    error=str(exc),
                )
                continue

            live_reward = self._live_reward(best["solution"], state)
            if live_reward is not None and live_reward <= self.commit_threshold:
                self.env.record_rollback(
                    uid,
                    question,
                    answer,
                    {
                        "reason": "live_memory_revalidation_failed",
                        "rollout_reward": best["reward"],
                        "live_reward": live_reward,
                    },
                )
                self._record_coverage_result(state, success=False)
                self._save_memory_trajectory_step(
                    event="rollback",
                    uid=uid,
                    state=state,
                    rewards=rewards,
                    best_reward=live_reward,
                    reason="live_memory_revalidation_failed",
                )
                continue

            coverage_unit_ids = [str(item) for item in state.get("coverage_unit_ids", [])]
            commit_kwargs = {"current_answer": answer}
            if coverage_unit_ids:
                commit_kwargs.update(
                    coverage_unit_ids=coverage_unit_ids,
                    coverage_success=True,
                )
            self.env.commit_memory_update(uid, proposal, question, **commit_kwargs)
            logger.info(
                "Committed %s: reward %.3f, chunks=%d",
                uid,
                best["reward"],
                len(proposal.new_chunks),
            )
            self._save_memory_trajectory_step(
                event="commit",
                uid=uid,
                state=state,
                rewards=rewards,
                best_reward=best["reward"],
                proposal=proposal,
                reason="best_reward_positive",
            )

        self.batch_commit_count += 1
        for case_state in self.env.case_states.values():
            if case_state.ready:
                case_state.completed = True
        self._maybe_save_online_memory_checkpoint()

    def _record_coverage_result(self, state: Dict[str, Any], success: bool) -> None:
        case_id = str(state.get("case_id", ""))
        case_state = self.env.case_states.get(case_id)
        if case_state is None or case_state.coverage_tracker is None:
            return
        case_state.coverage_tracker.record(
            [str(item) for item in state.get("coverage_unit_ids", [])],
            success=success,
        )

    def _live_reward(self, solution: str, state: Dict[str, Any]) -> Optional[float]:
        if "current_memory" not in state or "reward_config" not in state:
            return None
        case_id = str(state.get("case_id", ""))
        case_state = self.env.case_states.get(case_id)
        if case_state is None or not isinstance(case_state.memory_store, MemoryStore):
            return None
        selected_ids = [str(item) for item in state.get("selected_memory_ids", [])]
        live_ids = {chunk.memory_id for chunk in case_state.memory_store.chunks}
        if state.get("action") == "merge" and not set(selected_ids).issubset(live_ids):
            return float("-inf")
        live_state = dict(state)
        live_state["current_memory"] = {
            "memories": [chunk.to_dict() for chunk in case_state.memory_store.chunks]
        }
        if case_state.coverage_tracker is not None:
            unit_ids = [str(item) for item in state.get("coverage_unit_ids", [])]
            live_state["coverage_route_weight"] = case_state.coverage_tracker.route_weight(unit_ids)
            live_state["coverage_pending_weight"] = case_state.coverage_tracker.pending_weight(unit_ids)
            live_state["coverage_critical_pending_weight"] = live_state["coverage_pending_weight"]
            live_state["coverage_before"] = case_state.coverage_tracker.structural_coverage()
        payload = compute_score(
            data_source="memory_refactor_online",
            solution_str=solution,
            ground_truth=live_state,
        )
        return float(payload.get("score", 0.0))

    def should_stop_training(self) -> bool:
        return bool(self.env.case_states) and all(
            state.completed or state.incomplete for state in self.env.case_states.values()
        )

    def on_train_end(self) -> None:
        """Persist final online memory state when the trainer exits."""
        if self.output_dir is not None:
            self._ensure_initial_trajectory_snapshot()
            self._save_online_memory_checkpoint("final")

    def _extract_batch_records(self, batch: Any, tokenizer: Any) -> List[Dict[str, Any]]:
        if hasattr(batch, "partition_id") and hasattr(batch, "keys"):
            try:
                import transfer_queue as tq

                fields = ["responses", "response_mask", "rm_scores", "reward_model", "extra_info"]
                data = tq.kv_batch_get(
                    keys=batch.keys,
                    partition_id=batch.partition_id,
                    select_fields=fields,
                )
                return self._records_from_fields(data, len(batch.keys), tokenizer, keys=batch.keys)
            except Exception as exc:
                logger.warning("Could not read online batch from TransferQueue: %s", exc)
                return []

        if hasattr(batch, "batch") and hasattr(batch, "non_tensor_batch"):
            fields = {
                "responses": batch.batch.get("responses"),
                "response_mask": batch.batch.get("response_mask"),
                "rm_scores": batch.batch.get("rm_scores"),
                "reward_model": batch.non_tensor_batch.get("reward_model"),
                "extra_info": batch.non_tensor_batch.get("extra_info"),
            }
            return self._records_from_fields(fields, len(batch), tokenizer)

        logger.warning("Unsupported online batch type for commit hook: %s", type(batch))
        return []

    def _records_from_fields(
        self,
        fields: Dict[str, Any],
        count: int,
        tokenizer: Any,
        keys: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        responses = _tensor_rows(fields.get("responses"), count)
        response_masks = _tensor_rows(fields.get("response_mask"), count)
        rewards = _tensor_rows(fields.get("rm_scores"), count)
        reward_models = _value_rows(fields.get("reward_model"), count)
        extra_infos = _value_rows(fields.get("extra_info"), count)

        records: List[Dict[str, Any]] = []
        for index in range(count):
            reward_model = _unwrap_value(reward_models[index])
            ground_truth = None
            if isinstance(reward_model, dict):
                ground_truth = reward_model.get("ground_truth")
            elif isinstance(reward_model, str):
                ground_truth = reward_model
            if not ground_truth:
                continue

            try:
                state = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
            except json.JSONDecodeError:
                logger.warning("Skipping rollout with invalid ground_truth JSON")
                continue
            if not isinstance(state, dict):
                continue

            extra_info = _unwrap_value(extra_infos[index])
            if not isinstance(extra_info, dict):
                extra_info = {}

            uid = str(extra_info.get("uid") or state.get("uid") or "")
            if not uid and keys:
                uid = str(keys[index]).rsplit("_", 2)[0]

            records.append(
                {
                    "uid": uid,
                    "state": state,
                    "solution": _decode_response(
                        tokenizer,
                        responses[index],
                        response_masks[index],
                    ),
                    "reward": _sum_reward(rewards[index]),
                }
            )

        return records

    @staticmethod
    def _proposal_from_rollout(rollout_text: str, state: Dict[str, Any]) -> RefactorProposal:
        try:
            payload = json.loads(str(rollout_text or "").strip())
        except json.JSONDecodeError:
            start = str(rollout_text or "").find("{")
            end = str(rollout_text or "").rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Cannot parse JSON from rollout")
            payload = json.loads(str(rollout_text)[start : end + 1])

        chunks = [
            MemoryChunk.from_dict(item, fallback_id=f"{state.get('action', 'add')}_{index}")
            for index, item in enumerate(payload.get("chunks", []))
        ]
        if not chunks:
            raise ValueError("No chunks in rollout output")

        action = state.get("action", "add")
        if action == "add":
            chunks = chunks[:1]
            remove_ids: List[str] = []
        else:
            selected_ids = [str(item) for item in state.get("selected_memory_ids", [])]
            chunks = chunks[: max(1, len(selected_ids))]
            remove_ids = selected_ids

        return RefactorProposal(
            action=action,
            new_chunks=chunks,
            remove_memory_ids=remove_ids,
            metadata={"source": "online_grpo_commit"},
        )

    def save_memory_states(self, output_dir: str) -> None:
        """Save all case memory states to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for case_id, state in self.env.case_states.items():
            memory_path = output_path / f"{case_id}_ep{state.episode_count}.json"
            state.memory_store.save(memory_path)

            pool_path = output_path / f"{case_id}_ep{state.episode_count}_pool.json"
            state.success_pool.save(pool_path)

            buffer_path = output_path / f"{case_id}_ep{state.episode_count}_buffer.json"
            state.high_priority_buffer.save(buffer_path)

        logger.info(f"Saved memory states for {len(self.env.case_states)} cases to {output_dir}")

    def _maybe_save_online_memory_checkpoint(self) -> None:
        if self.output_dir is None:
            return
        if self.memory_save_interval > 0 and self.batch_commit_count % self.memory_save_interval == 0:
            self._save_online_memory_checkpoint("latest")
        if self.checkpoint_interval > 0 and self.batch_commit_count % self.checkpoint_interval == 0:
            self._save_online_memory_checkpoint(f"step{self.batch_commit_count}")

    def _save_online_memory_checkpoint(self, suffix: str) -> None:
        assert self.output_dir is not None
        checkpoint_dir = self.output_dir / f"checkpoint_{suffix}"
        memory_dir = checkpoint_dir / "memory_states"
        self.save_memory_states(str(memory_dir))
        metadata = {
            "batch_commit_count": self.batch_commit_count,
            "num_graphs": len(self.graphs),
            "case_episodes": {
                case_id: state.episode_count
                for case_id, state in self.env.case_states.items()
            },
        }
        metadata_path = checkpoint_dir / "training_metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ensure_initial_trajectory_snapshot(self) -> None:
        if self.trajectory_step == 0:
            self._save_memory_trajectory_step(
                event="initial",
                reason="before_first_online_update",
            )

    def _save_memory_trajectory_step(
        self,
        event: str,
        uid: str = "",
        state: Optional[Dict[str, Any]] = None,
        rewards: Optional[List[float]] = None,
        best_reward: Optional[float] = None,
        proposal: Optional[RefactorProposal] = None,
        reason: str = "",
        error: str = "",
    ) -> None:
        """Persist a full all-case M_t snapshot for post-hoc debugging."""
        if self.output_dir is None or not self.memory_trajectory_enabled:
            return

        state = state or {}
        rewards = rewards or []
        step_dir = self.output_dir / self.memory_trajectory_dir / f"step_{self.trajectory_step:06d}"
        memory_dir = step_dir / "memory_states"
        self.save_memory_states(str(memory_dir))

        case_id = str(state.get("case_id") or _case_id_from_uid(uid) or "")
        event_payload = {
            "trajectory_step": self.trajectory_step,
            "event": event,
            "reason": reason,
            "error": error,
            "uid": uid,
            "case_id": case_id,
            "question": str(state.get("question", "")),
            "answer": str(state.get("answer", "")),
            "action": str(state.get("action", "")),
            "selected_memory_ids": [str(item) for item in state.get("selected_memory_ids", [])],
            "batch_commit_count": self.batch_commit_count,
            "best_reward": best_reward,
            "all_rewards": [float(item) for item in rewards],
            "proposal": proposal.to_dict() if proposal is not None else None,
            "case_episodes": {
                case_id: case_state.episode_count
                for case_id, case_state in self.env.case_states.items()
            },
            "memory_chunk_counts": {
                case_id: len(case_state.memory_store.chunks)
                for case_id, case_state in self.env.case_states.items()
            },
            "coverage_states": {
                case_id: case_state.coverage_tracker.snapshot()
                for case_id, case_state in self.env.case_states.items()
                if case_state.coverage_tracker is not None
            },
            "snapshot_dir": str(step_dir),
            "memory_states_dir": str(memory_dir),
        }
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "event.json").write_text(
            json.dumps(event_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        index_path = self.output_dir / self.memory_trajectory_dir / "index.jsonl"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_payload, ensure_ascii=False) + "\n")

        logger.info(
            "Saved memory trajectory step %06d (%s) to %s",
            self.trajectory_step,
            event,
            step_dir,
        )
        self.trajectory_step += 1


def _tensor_rows(value: Any, count: int) -> List[Any]:
    value = _unwrap_value(value)
    if value is None:
        return [None] * count
    if isinstance(value, torch.Tensor):
        if value.is_nested:
            try:
                return list(value.unbind())
            except RuntimeError:
                padded = value.to_padded_tensor(0)
                return [padded[index] for index in range(count)]
        if value.dim() == 0:
            return [value] * count
        if value.dim() == 1 and count == 1:
            return [value]
        return [row for row in value]
    return _value_rows(value, count)


def _value_rows(value: Any, count: int) -> List[Any]:
    value = _unwrap_value(value)
    if value is None:
        return [None] * count
    if isinstance(value, list | tuple):
        return [_unwrap_value(item) for item in value]
    if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
        return [_unwrap_value(value[index]) for index in range(count)]
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [_unwrap_value(item) for item in converted]
    return [value] * count


def _unwrap_value(value: Any) -> Any:
    if type(value).__module__.startswith("numpy"):
        if getattr(value, "shape", None) == ():
            try:
                return value.item()
            except (AttributeError, ValueError):
                return value
        return value
    if hasattr(value, "data") and not isinstance(value, torch.Tensor):
        data = value.data
        if isinstance(data, memoryview):
            return value
        return data
    return value


def _decode_response(tokenizer: Any, response: Any, response_mask: Any = None) -> str:
    token_ids = _to_list(response)
    mask = _to_list(response_mask)
    if mask:
        token_ids = [
            token_id
            for token_id, keep in zip(token_ids, mask)
            if int(keep) != 0
        ]
    return tokenizer.decode([int(token_id) for token_id in token_ids], skip_special_tokens=True)


def _sum_reward(reward_row: Any) -> float:
    values = _to_list(reward_row)
    if not values:
        return 0.0
    # The first reward channel is the scalar score returned by compute_score.
    # Remaining channels are diagnostics; summing them changes the commit
    # decision and can reward metadata instead of behavior.
    first = float(values[0])
    if first != 0.0 or len(values) == 1:
        return first
    nonzero = [float(value) for value in values if float(value) != 0.0]
    if len(nonzero) == 1:
        return nonzero[0]
    return first


def _case_id_from_uid(uid: str) -> str:
    return str(uid or "").split("_ep", 1)[0]


def _config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() not in {"0", "false", "no", "off", ""}


def _to_list(value: Any) -> List[Any]:
    value = _unwrap_value(value)
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, list | tuple):
        return list(value)
    return [value]
