"""CLI for alternating attacker/defender GRPO co-training."""

import argparse
import json
import logging
import sys
from urllib.parse import urlparse
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
            "Model name served by the attacker API, e.g. attacker-current."
        ),
    )
    parser.add_argument(
        "--manage-attacker-server",
        dest="manage_attacker_server",
        action="store_true",
        default=None,
        help="Merge attacker checkpoints and manage the vLLM attacker server.",
    )
    parser.add_argument(
        "--no-manage-attacker-server",
        dest="manage_attacker_server",
        action="store_false",
        help="Do not start or stop the attacker server from this process.",
    )
    parser.add_argument("--attacker-server-host", help="Host passed to vLLM --host")
    parser.add_argument("--attacker-server-port", type=int, help="Port passed to vLLM --port")
    parser.add_argument("--attacker-server-dtype", help="vLLM dtype for the attacker server")
    parser.add_argument("--attacker-server-tp", type=int, help="vLLM tensor parallel size")
    parser.add_argument(
        "--attacker-server-gpu-memory-utilization",
        type=float,
        help="vLLM GPU memory utilization for the attacker server",
    )
    parser.add_argument(
        "--attacker-server-startup-timeout",
        type=float,
        help="Seconds to wait for attacker /v1/models to become healthy",
    )
    return parser.parse_args()


def _attacker_server_config(config: dict) -> dict:
    server_config = config.get("attacker_server")
    if not isinstance(server_config, dict):
        server_config = {}
        config["attacker_server"] = server_config
    return server_config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.cotrain_rounds is not None:
        config["cotrain_rounds"] = args.cotrain_rounds
    if args.attacker_api_base:
        config["attacker_api_base"] = args.attacker_api_base
        parsed = urlparse(args.attacker_api_base)
        if parsed.hostname:
            _attacker_server_config(config)["request_host"] = parsed.hostname
        if parsed.port:
            _attacker_server_config(config)["port"] = parsed.port
    if args.attacker_served_model:
        config["attacker_served_model"] = args.attacker_served_model
        _attacker_server_config(config)["served_model_name"] = args.attacker_served_model
    if args.manage_attacker_server is not None:
        config["manage_attacker_server"] = args.manage_attacker_server
        _attacker_server_config(config)["enabled"] = args.manage_attacker_server
    if args.attacker_server_host:
        _attacker_server_config(config)["host"] = args.attacker_server_host
    if args.attacker_server_port is not None:
        _attacker_server_config(config)["port"] = args.attacker_server_port
        config["attacker_api_base"] = f"http://localhost:{args.attacker_server_port}/v1"
    if args.attacker_server_dtype:
        _attacker_server_config(config)["dtype"] = args.attacker_server_dtype
    if args.attacker_server_tp is not None:
        _attacker_server_config(config)["tensor_parallel_size"] = args.attacker_server_tp
    if args.attacker_server_gpu_memory_utilization is not None:
        _attacker_server_config(config)["gpu_memory_utilization"] = args.attacker_server_gpu_memory_utilization
    if args.attacker_server_startup_timeout is not None:
        _attacker_server_config(config)["startup_timeout"] = args.attacker_server_startup_timeout

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
