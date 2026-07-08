"""Online dataset for training an attacker policy with fixed graph routing."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch.utils.data

from .attacker_grpo_adapter import build_attacker_verl_row
from .evidence import route_golden_facts
from .retriever import FrozenBM25Retriever, MemoryStore
from .routing import (
    HeuristicRoutingPolicy,
    RandomWalkRoutingPolicy,
    public_route_evidence,
)

logger = logging.getLogger(__name__)


@dataclass
class AttackerCaseState:
    case_id: str
    graph: Dict[str, Any]
    memory_store: MemoryStore
    episode_count: int = 0


class OnlineAttackerDataset(torch.utils.data.Dataset):
    """Generate attacker GRPO states on demand.

    The graph routing policy is fixed. The trainable policy only generates the
    attack question and answer from public route evidence plus behavior-level
    memory observations.
    """

    def __init__(
        self,
        graph_files: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        initial_memory_dir: Optional[str] = None,
        data_files: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        processor: Optional[Any] = None,
        max_samples: int = -1,
    ):
        del tokenizer, processor, max_samples
        if data_files is not None and graph_files is None:
            if isinstance(data_files, str):
                config_file = Path(data_files)
            elif isinstance(data_files, list) and len(data_files) > 0:
                config_file = Path(data_files[0])
            else:
                raise ValueError("data_files must contain an attacker dataset config path")
            if not config_file.exists():
                raise ValueError(f"Config file not found: {config_file}")
            dataset_config = json.loads(config_file.read_text(encoding="utf-8"))
            graph_files = dataset_config["graph_files"]
            config = dataset_config["config"]
            initial_memory_dir = dataset_config.get("initial_memory_dir")

        if graph_files is None or config is None:
            raise ValueError("graph_files and config must be provided")

        self.config = config
        self.graphs = self._load_and_validate_graphs(graph_files)
        if not self.graphs:
            raise ValueError("No valid graphs found")

        self.routing_policy = self._build_routing_policy(config)
        self.global_seed = int(config.get("seed", 42))
        self.routing_seed_offset = int(config.get("routing_seed_offset", 10000))
        self.episodes_per_case = int(config.get("attacker_episodes_per_case", config.get("episodes_per_case", 100)))
        self.memory_context_limit = int(config.get("attacker_memory_context_limit", 12))
        self.top_k = int(config.get("top_k", 5))
        self.case_rotation = 0
        self.case_states = {
            str(graph.get("case_id", f"case_{index}")): AttackerCaseState(
                case_id=str(graph.get("case_id", f"case_{index}")),
                graph=graph,
                memory_store=self._load_initial_memory(graph, initial_memory_dir),
            )
            for index, graph in enumerate(self.graphs)
        }
        self.recent_attacks = self._load_recent_attacks(config.get("mistake_book_path"))

    def __len__(self) -> int:
        return len(self.graphs) * self.episodes_per_case

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_idx = self.case_rotation % len(self.graphs)
        self.case_rotation += 1
        graph = self.graphs[case_idx]
        case_id = str(graph.get("case_id", f"case_{case_idx}"))
        case_state = self.case_states[case_id]
        episode = case_state.episode_count
        case_state.episode_count += 1
        seed = self.global_seed + self.routing_seed_offset + idx + episode * 1000
        route = self._select_route(graph, seed)
        route_evidence = public_route_evidence(graph, route)
        golden_facts = route_golden_facts(graph, route)
        state = {
            "case_id": case_id,
            "uid": f"{case_id}_att_ep{episode}",
            "episode": episode,
            "step": episode,
            "route": route_evidence,
            "golden_facts": golden_facts,
            "current_memory": self._memory_context(case_state.memory_store, route_evidence),
            "memory_observation": self._memory_observation(case_state.memory_store, route_evidence),
            "recent_attacks": self.recent_attacks,
            "top_k": self.top_k,
        }
        return build_attacker_verl_row(state, index=idx)

    @staticmethod
    def _load_and_validate_graphs(graph_files: List[str]) -> List[Dict[str, Any]]:
        graphs = []
        for file_path in graph_files:
            try:
                graph = json.loads(Path(file_path).read_text(encoding="utf-8"))
                if len(graph.get("entities", [])) >= 3 and len(graph.get("relationships", [])) >= 2:
                    graphs.append(graph)
                else:
                    logger.warning("Skipping invalid graph for attacker training: %s", file_path)
            except Exception as exc:
                logger.warning("Failed to load graph %s: %s", file_path, exc)
        return graphs

    @staticmethod
    def _build_routing_policy(config: Dict[str, Any]) -> Any:
        policy_name = str(config.get("routing_policy", "random_walk"))
        if policy_name == "heuristic":
            return HeuristicRoutingPolicy()
        if policy_name != "random_walk":
            logger.warning("Unsupported attacker routing policy %s; using random_walk", policy_name)
        return RandomWalkRoutingPolicy(
            seed=int(config.get("seed", 42)),
            max_steps=int(config.get("routing_max_steps", 3)),
            min_nodes=int(config.get("routing_min_nodes", 1)),
            attempts=int(config.get("routing_attempts", 8)),
        )

    @staticmethod
    def _load_initial_memory(graph: Dict[str, Any], initial_memory_dir: Optional[str]) -> MemoryStore:
        if not initial_memory_dir:
            return MemoryStore()
        case_id = str(graph.get("case_id", "unknown"))
        memory_path = Path(initial_memory_dir) / f"{case_id}.json"
        if memory_path.exists():
            return MemoryStore.load(memory_path)
        candidates = sorted(Path(initial_memory_dir).glob(f"{case_id}*.json"))
        for candidate in reversed(candidates):
            if not candidate.name.endswith("_pool.json") and not candidate.name.endswith("_buffer.json"):
                return MemoryStore.load(candidate)
        return MemoryStore()

    def _select_route(self, graph: Dict[str, Any], seed: int):
        try:
            return self.routing_policy.select_route(graph, seed=seed)
        except TypeError:
            return self.routing_policy.select_route(graph)

    def _memory_context(self, memory_store: MemoryStore, route: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not memory_store.chunks or self.memory_context_limit <= 0:
            return []
        query = _route_query(route)
        hits = FrozenBM25Retriever(memory_store).retrieve(
            query,
            top_k=self.memory_context_limit,
            min_score=0.0,
        )
        if hits:
            return [hit.to_dict() for hit in hits]
        return [chunk.to_dict() for chunk in memory_store.chunks[: self.memory_context_limit]]

    def _memory_observation(self, memory_store: MemoryStore, route: Dict[str, Any]) -> Dict[str, Any]:
        query = _route_query(route)
        hits = FrozenBM25Retriever(memory_store).retrieve(query, top_k=self.top_k, min_score=0.0)
        return {
            "observation_type": "retrieval_only",
            "query": query[:500],
            "memory_count": len(memory_store.chunks),
            "retrieved_memory_ids": [hit.memory_id for hit in hits],
            "retrieved_count": len(hits),
            "note": "Behavior-level retrieval observation only; no defender logits, gradients, or training batch.",
        }

    @staticmethod
    def _load_recent_attacks(path: Optional[str]) -> List[Dict[str, Any]]:
        if not path:
            return []
        attack_path = Path(path)
        if not attack_path.exists():
            return []
        payload = json.loads(attack_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload[-20:]
        items = payload.get("attacks") or payload.get("mistakes") or []
        return list(items)[-20:]


def _route_query(route: Dict[str, Any]) -> str:
    parts: List[str] = []
    for entity in route.get("entities", []):
        parts.append(str(entity.get("name", "")))
        parts.append(str(entity.get("description", "")))
    for relationship in route.get("relationships", []):
        parts.append(str(relationship.get("source", "")))
        parts.append(str(relationship.get("target", "")))
        parts.append(str(relationship.get("description", "")))
    return " ".join(part for part in parts if part).strip()
