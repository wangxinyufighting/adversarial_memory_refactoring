"""CLI for memory certification."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from case_graph.certification import batch_certify_memories
from case_graph.retriever import retriever_config_from_mapping

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify memory coverage for evaluation.")
    parser.add_argument("--memory-dir", required=True, help="Directory with <case_id>.json memories")
    parser.add_argument("--graphs", required=True, help="CaseGraph directory or file")
    parser.add_argument("--output", required=True, help="Output certification report JSON")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top-K")
    parser.add_argument("--training-config", default="configs/online_grpo.yaml")
    return parser.parse_args()


def load_case_graphs(path: str) -> Dict[str, Dict[str, Any]]:
    """Load case graphs from directory or file."""
    graph_path = Path(path)
    if graph_path.is_dir():
        paths = sorted(graph_path.glob("*.case_graph.json"))
    else:
        paths = [graph_path]

    graphs = {}
    for item in paths:
        graph = json.loads(item.read_text(encoding="utf-8"))
        if not isinstance(graph, dict):
            continue
        case_id = str(graph.get("case_id") or item.name.replace(".case_graph.json", ""))
        if case_id:
            graphs[case_id] = graph
    return graphs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    config_path = Path(args.training_config)
    if config_path.exists():
        training_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        logger.warning("Training config not found, using defaults: %s", args.training_config)
        training_config = {}

    retriever_config = retriever_config_from_mapping({
        "retriever": training_config.get("retriever", {}),
    })

    logger.info("Loading case graphs from: %s", args.graphs)
    graphs = load_case_graphs(args.graphs)
    logger.info("Loaded %d case graphs", len(graphs))

    logger.info("Certifying memories in: %s", args.memory_dir)
    results = batch_certify_memories(
        memory_dir=args.memory_dir,
        graphs=graphs,
        retriever_config=retriever_config,
        top_k=args.top_k,
    )

    certified = sum(1 for r in results.values() if r.get("certified"))
    total = len(results)
    mean_coverage = sum(r.get("source_coverage", 0.0) for r in results.values()) / total if total else 0.0

    summary = {
        "total_cases": total,
        "certified_cases": certified,
        "certification_rate": certified / total if total else 0.0,
        "mean_source_coverage": mean_coverage,
        "retriever_config": retriever_config,
        "top_k": args.top_k,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Certification summary:")
    logger.info("  Total cases: %d", total)
    logger.info("  Certified: %d (%.1f%%)", certified, 100 * summary["certification_rate"])
    logger.info("  Mean source coverage: %.3f", mean_coverage)
    logger.info("Results saved to: %s", args.output)


if __name__ == "__main__":
    main()
