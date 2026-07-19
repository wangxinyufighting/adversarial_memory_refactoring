"""Dataset for training attacker with frozen defender memory states."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .routing import random_walk_route
from .models import CaseGraph

logger = logging.getLogger(__name__)


class OnlineAttackerDataset:
    """Dataset that generates attacker training states from case graphs.

    Each state consists of:
    - Case graph with entities and relationships
    - Current defender memory state M_t
    - Route evidence for question generation
    """

    def __init__(
        self,
        graph_files: List[str],
        defender_memory_dir: Optional[str],
        config: Dict[str, Any],
    ):
        self.config = config
        self.episodes_per_case = config.get("episodes_per_case", 100)
        self.routing_policy = config.get("routing_policy", "random_walk")
        self.routing_attempts = config.get("routing_attempts", 12)

        # Load case graphs
        self.graphs = self._load_graphs(graph_files)
        logger.info(f"Loaded {len(self.graphs)} case graphs for attacker training")

        # Load defender memory states (frozen from previous round)
        self.memory_states = self._load_memory_states(defender_memory_dir)
        logger.info(f"Loaded {len(self.memory_states)} defender memory states")

        # Track episode counts
        self.case_episode_counts = {g["case_id"]: 0 for g in self.graphs}

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
            logger.warning(f"Memory directory {memory_dir} not found, using empty states")
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
        route = self._generate_route(graph)

        # Build state for attacker
        state = {
            "case_id": case_id,
            "episode": episode,
            "graph": graph,
            "memory_state": memory_state,
            "route": route,
            "golden_facts": self._extract_golden_facts(graph, route),
        }

        # Format as verl row
        return self._build_verl_row(state, idx)

    def _generate_route(self, graph: Dict) -> Dict[str, Any]:
        """Generate attack route from graph."""
        if self.routing_policy == "random_walk":
            # Use random walk routing
            for attempt in range(self.routing_attempts):
                route = random_walk_route(
                    graph,
                    min_nodes=self.config.get("routing_min_nodes", 1),
                    max_steps=self.config.get("routing_max_steps", 4),
                    seed_offset=attempt,
                )
                if route and len(route.entity_ids) >= 1:
                    return route.to_dict()

            # Fallback: simple route with first entity
            entities = graph.get("entities", [])
            if entities:
                return {
                    "entity_ids": [entities[0]["entity_id"]],
                    "path": [entities[0]["entity_id"]],
                    "source_ids": entities[0].get("source_ids", []),
                }

        return {"entity_ids": [], "path": [], "source_ids": []}

    def _extract_golden_facts(self, graph: Dict, route: Dict) -> List[Dict]:
        """Extract golden facts from route source IDs."""
        source_ids = set(route.get("source_ids", []))
        chunks = graph.get("chunks", [])

        return [
            chunk for chunk in chunks
            if chunk.get("chunk_id") in source_ids
        ]

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
                "ground_truth": json.dumps({
                    "case_id": state["case_id"],
                    "episode": state["episode"],
                    "route": state["route"],
                    "golden_facts": state["golden_facts"],
                    "memory_state": state["memory_state"],
                }, ensure_ascii=False)
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

        # Extract entities and relationships from route
        entity_ids = route.get("entity_ids", [])
        entities = [
            e for e in graph.get("entities", [])
            if e["entity_id"] in entity_ids
        ]

        # Build prompt
        prompt_parts = []
        prompt_parts.append("## Graph Route Evidence")

        for entity in entities:
            prompt_parts.append(f"Entity: {entity['entity_name']}")
            if entity.get("description"):
                prompt_parts.append(f"  Description: {entity['description']}")

        # Add relationships if available
        relationships = graph.get("relationships", [])
        route_relationships = [
            r for r in relationships
            if r["source_id"] in entity_ids or r["target_id"] in entity_ids
        ]

        if route_relationships:
            prompt_parts.append("\n## Relationships")
            for rel in route_relationships[:5]:  # Limit to top 5
                prompt_parts.append(
                    f"{rel['source_name']} --[{rel['relationship_type']}]--> {rel['target_name']}"
                )

        prompt_parts.append("\n## Task")
        prompt_parts.append(
            "Generate one challenging question that tests memory of this evidence. "
            "The question should be specific and grounded in the evidence above. "
            "Return JSON: {\"question\": \"...\", \"answer\": \"...\"}"
        )

        return "\n".join(prompt_parts)
