"""Manage a defender OpenAI-compatible server for evaluation construction."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DefenderServerHandle:
    source_model_path: str
    served_model_path: str
    served_model_name: str
    api_base: str
    pid: Optional[int]
    log_path: Optional[str]
    merged: bool


class DefenderServerManager:
    """Merge a verl defender checkpoint and serve it with vLLM."""

    def __init__(
        self,
        output_dir: str | Path,
        served_model_name: str = "defender-current",
        api_base: str = "http://localhost:8004/v1",
        host: str = "0.0.0.0",
        port: int = 8004,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.6,
        startup_timeout: float = 600,
        health_check_interval: float = 5,
        shutdown_timeout: float = 30,
        checkpoint_backend: str = "fsdp",
        checkpoint_subdir: str = "actor",
        python_executable: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.served_model_name = served_model_name
        self.api_base = api_base.rstrip("/")
        self.host = host
        self.port = int(port)
        self.dtype = dtype
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.startup_timeout = float(startup_timeout)
        self.health_check_interval = float(health_check_interval)
        self.shutdown_timeout = float(shutdown_timeout)
        self.checkpoint_backend = checkpoint_backend
        self.checkpoint_subdir = checkpoint_subdir
        self.python_executable = python_executable or sys.executable
        self.export_dir = self.output_dir / "defender_server_exports"
        self.log_dir = self.output_dir / "defender_server_logs"
        self.pid_file = self.output_dir / "defender_server.pid"
        self.process: Optional[subprocess.Popen[Any]] = None
        self.log_file = None

    def serve(self, model_path: str | Path) -> DefenderServerHandle:
        source_path = Path(model_path)
        served_path, merged = self._prepare_model(source_path)
        self.restart(served_path)
        return DefenderServerHandle(
            source_model_path=str(source_path),
            served_model_path=str(served_path),
            served_model_name=self.served_model_name,
            api_base=self.api_base,
            pid=self.process.pid if self.process else None,
            log_path=str(self.log_dir / "defender_server.log"),
            merged=merged,
        )

    def restart(self, model_path: Path) -> None:
        self.stop()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "defender_server.log"
        self.log_file = log_path.open("a", encoding="utf-8")
        cmd = self._server_command(model_path)
        logger.info("Starting defender server: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            env=os.environ.copy(),
        )
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(self.process.pid), encoding="utf-8")
        self._wait_until_ready()

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            _terminate_process(self.process, self.shutdown_timeout)
        elif self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text(encoding="utf-8").strip())
            except ValueError:
                pid = None
            if pid is not None and _pid_alive(pid):
                _terminate_pid(pid, self.shutdown_timeout)

        self.process = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        if self.pid_file.exists():
            self.pid_file.unlink()

    def _prepare_model(self, model_path: Path) -> tuple[Path, bool]:
        if _is_hf_model_dir(model_path):
            return model_path, False
        checkpoint_dir = self._checkpoint_merge_dir(model_path)
        target_dir = self.export_dir / f"{model_path.name}_hf"
        marker = target_dir / "source_checkpoint.txt"
        if (
            _is_hf_model_dir(target_dir)
            and marker.exists()
            and marker.read_text(encoding="utf-8").strip() == str(checkpoint_dir.resolve())
        ):
            return target_dir, True

        target_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python_executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            self.checkpoint_backend,
            "--local_dir",
            str(checkpoint_dir),
            "--target_dir",
            str(target_dir),
        ]
        logger.info("Merging defender checkpoint: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=str(Path.cwd()), env=os.environ.copy(), check=True)
        if not _is_hf_model_dir(target_dir):
            raise RuntimeError(f"Merged defender model is not a HuggingFace model: {target_dir}")
        marker.write_text(str(checkpoint_dir.resolve()), encoding="utf-8")
        return target_dir, True

    def _checkpoint_merge_dir(self, checkpoint_path: Path) -> Path:
        if self.checkpoint_subdir:
            candidate = checkpoint_path / self.checkpoint_subdir
            if candidate.exists():
                return candidate
        actor = checkpoint_path / "actor"
        return actor if actor.exists() else checkpoint_path

    def _server_command(self, model_path: Path) -> List[str]:
        return [
            self.python_executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model_path),
            "--served-model-name",
            self.served_model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]

    def _wait_until_ready(self) -> None:
        url = f"{self.api_base}/models"
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    "Defender server exited before becoming ready. "
                    f"See log: {self.log_file.name if self.log_file else 'unknown'}"
                )
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                model_ids = _model_ids(payload)
                if self.served_model_name in model_ids:
                    logger.info("Defender server ready at %s", self.api_base)
                    return
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                pass
            time.sleep(self.health_check_interval)
        raise TimeoutError(f"Defender server did not become ready within {self.startup_timeout:.0f}s at {url}")


def _is_hf_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")) or any(path.glob("model*.bin"))


def _model_ids(payload: Dict[str, Any]) -> List[str]:
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    return [str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate_pid(pid: int, timeout: float) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _terminate_process(process: subprocess.Popen[Any], timeout: float) -> None:
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
