"""Online memory dataset for GRPO training.

Generates memory refactoring states on-demand during training by:
1. Selecting a case graph (round-robin)
2. Generating an adversarial attack from the graph
3. Preparing a refactoring state if initial defense fails
4. Returning verl-compatible training row
"""

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch.utils.data

from .attacker import FrozenLLMAttacker
from .baseline import AnswerEquivalenceJudge
from .defense import RetrievedMemoryAnswerAgent, SuccessPool
from .grpo_adapter import build_verl_row_online
from .pipeline import AlgorithmConfig, prepare_refactor_state
from .refactoring import HighPriorityBuffer, RefactorProposal, build_sandbox_memory
from .retriever import MemoryStore
from .routing import RandomWalkRoutingPolicy

logger = logging.getLogger(__name__)


@dataclass
class CaseMemoryState:
    """Per-case persistent memory state across episodes."""

    case_id: str
    graph: Dict[str, Any]
    memory_store: MemoryStore
    success_pool: SuccessPool
    high_priority_buffer: HighPriorityBuffer
    episode_count: int = 0


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
            # Note: RandomWalkRoutingPolicy uses its own seed, we can't pass per-episode seed
            route = self.routing_policy.select_route(case_state.graph)
            attack = self.attacker.generate(case_state.graph, route)
            attack_dict = {
                "case_id": case_id,
                "question": attack.question,
                "answer": attack.answer,
                "golden_facts": attack.golden_facts,
            }

        except Exception as e:
            logger.warning(f"Attack generation failed for case {case_id} episode {episode}: {e}")
            return None

        # 2. Prepare refactor state (includes initial defense check)
        state = prepare_refactor_state(
            attack=attack_dict,
            memory_store=case_state.memory_store,
            success_pool=case_state.success_pool,
            answer_agent=self.answer_agent,
            judge=self.judge,
            config=self.config,
            step=episode,
        )

        if state is None:
            # Initial defense succeeded, no refactor needed
            return None

        # 3. Add UID for GRPO grouping
        state["uid"] = f"{case_id}_ep{episode}"
        state["episode"] = episode

        return state

    def commit_memory_update(
        self, uid: str, best_proposal: RefactorProposal, current_question: str
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
                answer="",  # Answer not needed for success pool tracking
                memory_ids=[chunk.memory_id for chunk in best_proposal.new_chunks],
                metadata={"committed_from_online_training": True},
            )


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

        # Load and validate graphs
        self.graphs = self._load_and_validate_graphs(graph_files)
        if not self.graphs:
            raise ValueError("No valid graphs found")

        # Initialize environment
        self.env = OnlineMemoryEnvironment(
            graphs=self.graphs,
            routing_policy=RandomWalkRoutingPolicy(
                seed=config.get("seed", 42),
                max_steps=config.get("routing_max_steps", 3),
                min_nodes=config.get("routing_min_nodes", 1),
                attempts=config.get("routing_attempts", 8),
            ),
            attacker=FrozenLLMAttacker(),
            answer_agent=RetrievedMemoryAnswerAgent(),
            judge=AnswerEquivalenceJudge(),
            config=AlgorithmConfig(
                tau=config.get("tau", 0.7),
                top_k=config.get("top_k", 5),
                regression_sample_size=config.get("regression_sample_size", 3),
                seed=config.get("seed", 42),
                commit_threshold=config.get("commit_threshold", 0.0),
            ),
            initial_memory_dir=initial_memory_dir,
        )

        # Episode tracking
        self.case_rotation = 0
        self.global_seed = config.get("seed", 42)
        self.episodes_per_case = config.get("episodes_per_case", 100)
        self.max_attack_attempts = config.get("max_attack_attempts", 10)

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
        # Round-robin case selection
        case_idx = self.case_rotation % len(self.graphs)
        self.case_rotation += 1

        case_graph = self.graphs[case_idx]
        case_id = str(case_graph.get("case_id", "unknown"))
        case_state = self.env.case_states[case_id]
        episode = case_state.episode_count
        case_state.episode_count += 1

        # Generate episode (may return None if initial defense succeeds)
        for attempt in range(self.max_attack_attempts):
            seed = self.global_seed + idx + attempt * 1000
            state = self.env.generate_episode(case_id, episode + attempt, seed)

            if state is not None:
                # Convert to verl format
                return build_verl_row_online(state)

        # All attempts failed or succeeded in initial defense
        # Return to next case
        logger.warning(
            f"Failed to generate refactor state for case {case_id} after {self.max_attack_attempts} attempts"
        )
        return self.__getitem__((idx + 1) % len(self))

    def commit_memory_update(
        self, uid: str, best_proposal: RefactorProposal, current_question: str
    ) -> None:
        """Commit best proposal to M_t after training step."""
        self.env.commit_memory_update(uid, best_proposal, current_question)

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
