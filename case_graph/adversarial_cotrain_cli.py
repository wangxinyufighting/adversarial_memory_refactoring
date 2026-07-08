"""CLI for alternating attacker/defender GRPO co-training."""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from .adversarial_cotrain_trainer import AdversarialCoTrainingTrainer
from .online_memory_cli import list_graph_files, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alternating attacker/defender GRPO co-training")
    parser.add_argument("--graphs", required=True, help="CaseGraph JSON file or directory")
    parser.add_argument("--attacker-model-path", required=True, help="Initial attacker HF model path")
    parser.add_argument("--defender-model-path", required=True, help="Initial defender HF model path")
    parser.add_argument("--output-dir", required=True, help="Co-training output directory")
    parser.add_argument("--config", default="configs/adversarial_cotrain.yaml")
    parser.add_argument("--initial-memory-dir")
    parser.add_argument("--cotrain-rounds", type=int)
    parser.add_argument("--attacker-api-base", help="OpenAI-compatible API for defender-time attack generation")
    parser.add_argument(
        "--attacker-served-model",
        help=(
            "Model name served by the attacker API. If omitted, the latest attacker "
            "checkpoint path is passed as the model name after each attacker phase."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.cotrain_rounds is not None:
        config["cotrain_rounds"] = args.cotrain_rounds
    if args.attacker_api_base:
        config["attacker_api_base"] = args.attacker_api_base
    if args.attacker_served_model:
        config["attacker_served_model"] = args.attacker_served_model

    graph_files = list_graph_files(args.graphs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "cotrain_config.yaml"
    config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
    (output_dir / "cotrain_inputs.json").write_text(
        json.dumps(
            {
                "graphs": graph_files,
                "attacker_model_path": args.attacker_model_path,
                "defender_model_path": args.defender_model_path,
                "initial_memory_dir": args.initial_memory_dir,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        trainer = AdversarialCoTrainingTrainer(
            config=config,
            attacker_model_path=args.attacker_model_path,
            defender_model_path=args.defender_model_path,
            graph_files=graph_files,
            output_dir=str(output_dir),
            initial_memory_dir=args.initial_memory_dir,
        )
        trainer.train()
    except Exception as exc:
        logger.error("Adversarial co-training failed: %s", exc)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
