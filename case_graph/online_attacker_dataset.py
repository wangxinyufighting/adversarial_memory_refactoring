"""Dataset for training attacker with frozen defender memory states."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.utils.data

from .evidence import route_golden_facts
from .routing import (
    HeuristicRoutingPolicy,
    RandomWalkRoutingPolicy,
    public_route_evidence,
)

logger = logging.getLogger(__name__)


class OnlineAttackerDataset(torch.utils.data.Dataset):
    """Dataset that generates attacker training states from case graphs.

    Each state consists of:
    - Case graph with entities and relationships
    - Current defender memory state M_t
    - Route evidence for question generation
    """

    def __init__(
        self,
        graph_files: Optional[List[str]] = None,
        defender_memory_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        # verl standard arguments (ignored, but accepted for compatibility)
        data_files: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        processor: Optional[Any] = None,
        max_samples: int = -1,
        **kwargs,
    ):
        logger.info(
            f"OnlineAttackerDataset.__init__ called with data_files={data_files}, graph_files={graph_files}"
        )

        # Parse dataset config from data_files if provided (verl pattern)
        if data_files and not graph_files:
            logger.info(f"Loading dataset config from data_files: {data_files}")
            dataset_config = self._load_dataset_config(data_files)
            graph_files = dataset_config.get("graph_files", [])
            defender_memory_dir = dataset_config.get("defender_memory_dir")
            config = dataset_config.get("config", config or {})
            logger.info(
                f"Extracted {len(graph_files) if graph_files else 0} graph files from config"
            )

        self.config = config or {}
        self.episodes_per_case = self.config.get("episodes_per_case", 100)
        self.routing_policy = self.config.get("routing_policy", "random_walk")
        self.routing_attempts = self.config.get("routing_attempts", 12)

        # Load case graphs
        self.graphs = self._load_graphs(graph_files or [])
        logger.info(f"Loaded {len(self.graphs)} case graphs for attacker training")

        if len(self.graphs) == 0:
            logger.error(f"CRITICAL: No graphs loaded! graph_files={graph_files}")

        # Load defender memory states (frozen from previous round)
        self.memory_states = self._load_memory_states(defender_memory_dir)
        logger.info(f"Loaded {len(self.memory_states)} defender memory states")

        # Track episode counts
        self.case_episode_counts = {g["case_id"]: 0 for g in self.graphs}

    def _load_dataset_config(self, data_files: Union[List[str], str]) -> Dict[str, Any]:
        """Load dataset config from JSON file (verl pattern).

        Args:
            data_files: List of paths or single path string, typically containing one JSON config file

        Returns:
            Dictionary with graph_files, defender_memory_dir, and config
        """
        if not data_files:
            logger.warning("_load_dataset_config called with empty data_files")
            return {}

        # Handle both string and list inputs
        if isinstance(data_files, str):
            config_path = Path(data_files)
        elif isinstance(data_files, list):
            config_path = Path(data_files[0])
        else:
            logger.error(f"Unexpected data_files type: {type(data_files)}")
            return {}

        logger.info(f"Attempting to load dataset config from: {config_path}")

        if not config_path.exists():
            logger.error(f"Dataset config file {config_path} not found")
            return {}

        with open(config_path) as f:
            config_data = json.load(f)
            logger.info(f"Loaded config with keys: {list(config_data.keys())}")
            if "graph_files" in config_data:
                logger.info(
                    f"Config contains {len(config_data['graph_files'])} graph files"
                )
            return config_data

    def _load_graphs(self, graph_files: List[str]) -> List[Dict]:
        """Load case graph files."""
        graphs = []
        for path in graph_files:
            file_path = Path(path)
            if file_path.is_file():
                with open(file_path) as f:
                    graphs.append(json.load(f))
            elif file_path.is_dir():
                for graph_file in file_path.glob("*.case_graph.json"):
                    with open(graph_file) as f:
                        graphs.append(json.load(f))
        return graphs

    def _load_memory_states(self, memory_dir: Optional[str]) -> Dict[str, Dict]:
        """Load frozen defender memory states."""
        if not memory_dir:
            # No memory states - attacker trains with empty memory baseline
            return {g["case_id"]: {"chunks": []} for g in self.graphs}

        memory_states = {}
        memory_path = Path(memory_dir)

        if not memory_path.exists():
            logger.warning(
                f"Memory directory {memory_dir} not found, using empty states"
            )
            return {g["case_id"]: {"chunks": []} for g in self.graphs}

        for memory_file in memory_path.glob("*.json"):
            case_id = memory_file.stem
            with open(memory_file) as f:
                memory_states[case_id] = json.load(f)

        return memory_states

    def __len__(self) -> int:
        """Total number of training episodes."""
        return len(self.graphs) * self.episodes_per_case

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate one attacker training state.

        Returns state dict compatible with verl dataset format.
        """
        # Map index to case and episode
        case_idx = idx % len(self.graphs)
        episode = idx // len(self.graphs)

        graph = self.graphs[case_idx]
        case_id = graph["case_id"]
        memory_state = self.memory_states.get(case_id, {"chunks": []})

        # Generate route for question generation
        route, golden_facts = self._generate_route(graph)

        # Build state for attacker
        state = {
            "case_id": case_id,
            "episode": episode,
            "graph": graph,
            "memory_state": memory_state,
            "route": route,
            "golden_facts": golden_facts,
        }

        # Format as verl row
        row = self._build_verl_row(state, idx)

        # verl expects raw_prompt to be set (normally done by RLHFDataset.__getitem__)
        # raw_prompt should be the messages list from the "prompt" field
        row["raw_prompt"] = row["prompt"]

        # verl expects index at top level (extracted from extra_info by RLHFDataset)
        row["index"] = row.get("extra_info", {}).get("index", idx)

        # verl expects tools_kwargs and interaction_kwargs at top level
        row["tools_kwargs"] = row.get("extra_info", {}).get("tools_kwargs", {})
        row["interaction_kwargs"] = row.get("extra_info", {}).get(
            "interaction_kwargs", {}
        )

        # verl requires dummy_tensor to ensure DataProto.batch is not empty
        import torch

        row["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        return row

    def _generate_route(
        self, graph: Dict
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Generate public route evidence plus evaluator-only golden facts."""
        if self.routing_policy == "random_walk":
            # Use random walk routing policy
            policy = RandomWalkRoutingPolicy(
                seed=0,
                max_steps=self.config.get("routing_max_steps", 4),
                min_nodes=self.config.get("routing_min_nodes", 1),
                attempts=self.routing_attempts,
            )
            route = policy.select_route(graph)
            evidence = public_route_evidence(graph, route)
            return evidence, route_golden_facts(graph, route)
        elif self.routing_policy == "heuristic":
            policy = HeuristicRoutingPolicy()
            route = policy.select_route(graph)
            evidence = public_route_evidence(graph, route)
            return evidence, route_golden_facts(graph, route)

        # Fallback: empty route
        return {"nodes": [], "relationships": []}, []

    def _build_verl_row(self, state: Dict, index: int) -> Dict[str, Any]:
        """Build verl-compatible training row."""
        # Build attacker prompt
        system_prompt = (
            "You are an attacker generating questions to test memory systems. "
            "Generate a challenging question and its answer based on the provided route evidence. "
            "Return JSON with fields: question, answer."
        )

        user_prompt = self._build_user_prompt(state)

        return {
            "data_source": "attacker_cotrain",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "reward_model": {
                "ground_truth": json.dumps(
                    {
                        "case_id": state["case_id"],
                        "episode": state["episode"],
                        "route": state["route"],
                        "golden_facts": state["golden_facts"],
                        "memory_state": state["memory_state"],
                    },
                    ensure_ascii=False,
                )
            },
            "extra_info": {
                "index": index,
                "case_id": state["case_id"],
                "episode": state["episode"],
            },
        }

    def _build_user_prompt(self, state: Dict) -> str:
        """Build user prompt for attacker."""
        route = state["route"]
        graph = state["graph"]

        # Extract entities from route nodes
        entity_names = route.get("nodes", [])
        entities = [
            e for e in graph.get("entities", []) if e.get("name") in entity_names
        ]

        # Build prompt
        prompt_parts = []
        prompt_parts.append("## Graph Route Evidence")

        for entity in entities:
            prompt_parts.append(f"Entity: {entity.get('name', 'Unknown')}")
            if entity.get("description"):
                prompt_parts.append(f"  Description: {entity['description']}")
            if entity.get("entity_type"):
                prompt_parts.append(f"  Type: {entity['entity_type']}")

        # Add relationships from route
        route_relationships = route.get("relationships", [])
        if route_relationships:
            prompt_parts.append("\n## Relationships")
            for rel in route_relationships:
                source = rel.get("source", "")
                target = rel.get("target", "")
                rel_type = rel.get("description") or rel.get("type", "")
                prompt_parts.append(f"  {source} --[{rel_type}]--> {target}")

        prompt_parts.append("\n## Task")
        prompt_parts.append(
            "Generate a challenging question and answer based on this evidence."
        )
        prompt_parts.append('Return JSON format: {"question": "...", "answer": "..."}')

        return "\n".join(prompt_parts)
