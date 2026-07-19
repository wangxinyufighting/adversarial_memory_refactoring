"""CLI entry point for Co-Training V2."""

import argparse
import logging
import sys
import yaml
from pathlib import Path

from case_graph.cotrain_v2_orchestrator import CoTrainV2Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Co-Training V2: Adversarial Attacker and Defender Training"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to cotrain_v2.yaml config file",
    )
    parser.add_argument(
        "--train-graphs",
        type=str,
        required=True,
        help="Directory or file pattern for training case graphs",
    )
    parser.add_argument(
        "--val-graphs",
        type=str,
        required=True,
        help="Directory or file pattern for validation case graphs",
    )
    parser.add_argument(
        "--attacker-init",
        type=str,
        required=True,
        help="Initial attacker model path",
    )
    parser.add_argument(
        "--defender-init",
        type=str,
        required=True,
        help="Initial defender model path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and results",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Expand graph paths
    train_graphs = _expand_graph_paths(args.train_graphs)
    val_graphs = _expand_graph_paths(args.val_graphs)

    logger.info(f"Found {len(train_graphs)} training graphs")
    logger.info(f"Found {len(val_graphs)} validation graphs")

    # Create orchestrator
    orchestrator = CoTrainV2Orchestrator(
        config=config,
        attacker_init_path=args.attacker_init,
        defender_init_path=args.defender_init,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        output_dir=args.output_dir,
    )

    # Run co-training
    try:
        orchestrator.run_cotrain()
        logger.info("Co-training completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Co-training failed: {e}", exc_info=True)
        return 1


def _expand_graph_paths(path_pattern: str) -> list:
    """Expand path pattern to list of graph files."""
    path = Path(path_pattern)

    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        return sorted(str(p) for p in path.glob("*.case_graph.json"))
    else:
        # Try glob pattern
        parent = path.parent
        if parent.exists():
            return sorted(str(p) for p in parent.glob(path.name))

    return []


if __name__ == "__main__":
    sys.exit(main())
