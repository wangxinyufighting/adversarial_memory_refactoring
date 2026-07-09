"""Alternating attacker/defender GRPO co-training."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .attacker_grpo_trainer import AttackerGRPOTrainer
from .attacker_server_manager import AttackerServerHandle, AttackerServerManager
from .online_memory_trainer import OnlineMemoryTrainer

logger = logging.getLogger(__name__)


class AdversarialCoTrainingTrainer:
    """Train attacker and defender in alternating phases.

    The attacker and defender never share a checkpoint directory. The attacker
    receives behavior-level memory observations only; it does not receive
    defender logits, gradients, or training batches.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        attacker_model_path: str,
        defender_model_path: str,
        graph_files: List[str],
        output_dir: str,
        initial_memory_dir: Optional[str] = None,
    ):
        self.config = dict(config)
        self.attacker_model_path = attacker_model_path
        self.defender_model_path = defender_model_path
        self.graph_files = graph_files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.initial_memory_dir = initial_memory_dir
        self.rounds = int(self.config.get("cotrain_rounds", 1))
        self.round_records: List[Dict[str, Any]] = []
        self.attacker_server = AttackerServerManager(self.config, self.output_dir)

    def train(self) -> None:
        current_memory_dir = self.initial_memory_dir
        attacker_resume: Optional[str] = self.config.get("attacker_resume_from_path")
        defender_resume: Optional[str] = self.config.get("defender_resume_from_path")
        attacker_served_model = self.config.get("attacker_served_model") or self.attacker_model_path

        try:
            for round_index in range(self.rounds):
                logger.info("=" * 60)
                logger.info("Adversarial co-training round %d/%d", round_index + 1, self.rounds)
                logger.info("=" * 60)

                self.attacker_server.stop()
                attacker_dir = self.output_dir / f"attacker_ckpt_t{round_index:03d}"
                defender_dir = self.output_dir / f"defender_ckpt_t{round_index:03d}"

                attacker_config = self._role_config("attacker")
                attacker_config["role"] = "attacker"
                if attacker_resume:
                    attacker_config["resume_from_path"] = attacker_resume
                attacker_trainer = AttackerGRPOTrainer(
                    config=attacker_config,
                    model_path=self.attacker_model_path,
                    graph_files=self.graph_files,
                    output_dir=str(attacker_dir),
                    initial_memory_dir=current_memory_dir,
                )
                attacker_trainer.train()
                latest_attacker_ckpt = _latest_verl_checkpoint(attacker_dir / "verl_checkpoints")
                if latest_attacker_ckpt:
                    attacker_resume = str(latest_attacker_ckpt)
                    attacker_served_model = str(
                        self.config.get("attacker_served_model") or latest_attacker_ckpt
                    )

                attacker_server_handle = self._serve_attacker_for_defender(
                    latest_attacker_ckpt=latest_attacker_ckpt,
                    attacker_resume=attacker_resume,
                    round_index=round_index,
                )
                if attacker_server_handle is not None:
                    attacker_served_model = attacker_server_handle.served_model_name

                defender_config = self._role_config("defender")
                defender_config["role"] = "defender"
                defender_config["attacker_llm"] = attacker_served_model
                if attacker_server_handle is not None:
                    defender_config["attacker_api_base"] = attacker_server_handle.api_base
                if defender_resume:
                    defender_config["resume_from_path"] = defender_resume
                defender_trainer = OnlineMemoryTrainer(
                    config=defender_config,
                    model_path=self.defender_model_path,
                    graph_files=self.graph_files,
                    output_dir=str(defender_dir),
                    initial_memory_dir=current_memory_dir,
                )
                defender_trainer.train()
                if not self._server_keep_alive_after_defender():
                    self.attacker_server.stop()
                latest_defender_ckpt = _latest_verl_checkpoint(defender_dir / "verl_checkpoints")
                if latest_defender_ckpt:
                    defender_resume = str(latest_defender_ckpt)

                latest_memory_dir = _latest_memory_dir(defender_dir)
                if latest_memory_dir:
                    current_memory_dir = str(latest_memory_dir)

                record = {
                    "round": round_index,
                    "attacker_dir": str(attacker_dir),
                    "defender_dir": str(defender_dir),
                    "attacker_resume_from_path": attacker_resume,
                    "defender_resume_from_path": defender_resume,
                    "attacker_served_model": attacker_served_model,
                    "attacker_server": _server_handle_dict(attacker_server_handle),
                    "memory_dir_for_next_round": current_memory_dir,
                    "checkpoint_policy": "separate_attacker_and_defender",
                }
                self.round_records.append(record)
                self._write_manifest()
        finally:
            if not self._server_keep_alive_after_train():
                self.attacker_server.stop()

    def _serve_attacker_for_defender(
        self,
        latest_attacker_ckpt: Optional[Path],
        attacker_resume: Optional[str],
        round_index: int,
    ) -> Optional[AttackerServerHandle]:
        if not self.attacker_server.enabled:
            return None
        model_path = latest_attacker_ckpt or Path(attacker_resume or self.attacker_model_path)
        return self.attacker_server.serve(model_path, round_index)

    def _server_keep_alive_after_defender(self) -> bool:
        server_config = self.config.get("attacker_server") or {}
        return _as_bool(server_config.get("keep_alive_after_defender", False))

    def _server_keep_alive_after_train(self) -> bool:
        server_config = self.config.get("attacker_server") or {}
        return _as_bool(server_config.get("keep_alive_after_train", False))

    def _role_config(self, role: str) -> Dict[str, Any]:
        role_key = f"{role}_training"
        config = {
            key: value
            for key, value in self.config.items()
            if key not in {"attacker_training", "defender_training"}
        }
        role_overrides = self.config.get(role_key, {})
        if isinstance(role_overrides, dict):
            config.update(role_overrides)
        return config

    def _write_manifest(self) -> None:
        manifest = {
            "rounds": self.round_records,
            "attacker_checkpoint_prefix": "attacker_ckpt_t",
            "defender_checkpoint_prefix": "defender_ckpt_t",
            "attacker_visibility": (
                "route evidence, current memory view, retrieval observation, and mistake examples only"
            ),
            "forbidden_attacker_visibility": [
                "defender policy logits",
                "defender hidden states",
                "defender gradients",
                "defender GRPO advantages",
                "defender training batch tensors",
            ],
        }
        (self.output_dir / "cotrain_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _latest_verl_checkpoint(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [path for path in root.glob("global_step_*") if path.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: _step_number(path.name), reverse=True)[0]


def _latest_memory_dir(root: Path) -> Optional[Path]:
    candidates = [
        path
        for path in root.glob("checkpoint_*/memory_states")
        if path.is_dir()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (_memory_step(path), path.stat().st_mtime), reverse=True)[0]


def _memory_step(path: Path) -> int:
    match = re.search(r"checkpoint_step(\d+)", str(path.parent))
    if match:
        return int(match.group(1))
    if path.parent.name == "checkpoint_final":
        return 10**12
    if path.parent.name == "checkpoint_latest":
        return 10**11
    return -1


def _step_number(name: str) -> int:
    match = re.search(r"global_step_(\d+)", name)
    return int(match.group(1)) if match else -1


def _server_handle_dict(handle: Optional[AttackerServerHandle]) -> Optional[Dict[str, Any]]:
    if handle is None:
        return None
    return {
        "source_model_path": handle.source_model_path,
        "served_model_path": handle.served_model_path,
        "served_model_name": handle.served_model_name,
        "api_base": handle.api_base,
        "pid": handle.pid,
        "log_path": handle.log_path,
        "merged": handle.merged,
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "on"}
