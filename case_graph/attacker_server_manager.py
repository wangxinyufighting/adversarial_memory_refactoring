"""Manage the OpenAI-compatible attacker server used during co-training."""

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


@dataclass
class AttackerServerHandle:
    """Metadata for the attacker endpoint used by one defender phase."""

    source_model_path: str
    served_model_path: str
    served_model_name: str
    api_base: str
    pid: Optional[int]
    log_path: Optional[str]
    merged: bool


class AttackerServerManager:
    """Export attacker checkpoints, serve them with vLLM, and check readiness."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str | Path,
        python_executable: Optional[str] = None,
    ):
        self.config = dict(config)
        self.server_config = dict(self.config.get("attacker_server") or {})
        self.output_dir = Path(output_dir)
        self.python_executable = python_executable or sys.executable
        self.process: Optional[subprocess.Popen[Any]] = None
        self.log_file = None

        enabled_value = self.config.get(
            "manage_attacker_server",
            self.server_config.get("enabled", False),
        )
        self.enabled = _as_bool(enabled_value)
        self.host = str(self.server_config.get("host", "0.0.0.0"))
        self.request_host = str(self.server_config.get("request_host", "localhost"))
        self.port = int(self.server_config.get("port", 8003))
        self.api_base = str(
            self.config.get("attacker_api_base")
            or self.server_config.get("api_base")
            or f"http://{self.request_host}:{self.port}/v1"
        ).rstrip("/")
        self.served_model_name = str(
            self.server_config.get("served_model_name")
            or self.config.get("attacker_served_model")
            or "attacker-current"
        )
        self.export_root = Path(
            self.server_config.get("export_dir")
            or self.output_dir / "attacker_server_exports"
        )
        self.log_dir = Path(
            self.server_config.get("log_dir")
            or self.output_dir / "attacker_server_logs"
        )
        self.pid_file = Path(
            self.server_config.get("pid_file")
            or self.output_dir / "attacker_server.pid"
        )

    def serve(
        self,
        model_path: str | Path,
        round_index: int,
    ) -> AttackerServerHandle:
        """Serve a trained attacker, or validate the configured external endpoint."""
        source_model_path = str(Path(model_path))
        if not self.enabled:
            external_model = str(
                self.config.get("attacker_llm") or self.served_model_name
            )
            self._wait_until_ready(expected_model=external_model, managed=False)
            return AttackerServerHandle(
                source_model_path=source_model_path,
                served_model_path=source_model_path,
                served_model_name=external_model,
                api_base=self.api_base,
                pid=None,
                log_path=None,
                merged=False,
            )

        served_model_path, merged = self.export_model(
            model_path,
            round_index,
            export_name="attacker",
        )
        self.restart(served_model_path, round_index)
        return AttackerServerHandle(
            source_model_path=source_model_path,
            served_model_path=str(served_model_path),
            served_model_name=self.served_model_name,
            api_base=self.api_base,
            pid=self.process.pid if self.process else None,
            log_path=str(self._log_path(round_index)),
            merged=merged,
        )

    def export_model(
        self,
        model_path: str | Path,
        round_index: int,
        export_name: str,
    ) -> tuple[Path, bool]:
        """Convert a training checkpoint into a model loadable next round."""
        return self._prepare_model(
            Path(model_path),
            round_index,
            export_name=export_name,
        )

    def restart(self, model_path: Path, round_index: int) -> None:
        """Stop the prior managed process and start vLLM for ``model_path``."""
        self.stop()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_path(round_index)
        self.log_file = log_path.open("a", encoding="utf-8")
        cmd = self._server_command(model_path)
        logger.info("Starting attacker server: %s", " ".join(cmd))
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                cwd=str(Path.cwd()),
                env=self._server_env(),
            )
            self.pid_file.parent.mkdir(parents=True, exist_ok=True)
            self.pid_file.write_text(str(self.process.pid), encoding="utf-8")
            self._wait_until_ready(
                expected_model=self.served_model_name,
                managed=True,
            )
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """Stop the vLLM process started by this manager."""
        if not self.enabled:
            return
        if self.process and self.process.poll() is None:
            logger.info("Stopping managed attacker server pid=%s", self.process.pid)
            _terminate_process(
                self.process,
                timeout=float(self.server_config.get("shutdown_timeout", 30)),
            )
        elif self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text(encoding="utf-8").strip())
            except ValueError:
                pid = None
            if pid is not None and _pid_alive(pid):
                logger.info("Stopping stale managed attacker server pid=%s", pid)
                _terminate_pid(
                    pid,
                    timeout=float(self.server_config.get("shutdown_timeout", 30)),
                )

        self.process = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        if self.pid_file.exists():
            self.pid_file.unlink()

    def _prepare_model(
        self,
        model_path: Path,
        round_index: int,
        export_name: str,
    ) -> tuple[Path, bool]:
        if _is_hf_model_dir(model_path):
            return model_path, False

        merge_enabled = _as_bool(self.server_config.get("merge_checkpoints", True))
        if not merge_enabled:
            return model_path, False

        checkpoint_dir = self._checkpoint_merge_dir(model_path)
        target_dir = self.export_root / f"round_{round_index:03d}_{export_name}_hf"
        marker_path = target_dir / "source_checkpoint.txt"
        if (
            _as_bool(self.server_config.get("reuse_export_if_exists", True))
            and _is_hf_model_dir(target_dir)
            and marker_path.exists()
            and marker_path.read_text(encoding="utf-8").strip()
            == str(checkpoint_dir.resolve())
        ):
            return target_dir, True

        target_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._merge_command(checkpoint_dir, target_dir)
        logger.info("Merging %s checkpoint: %s", export_name, " ".join(cmd))
        subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            env=os.environ.copy(),
            check=True,
        )
        if not _is_hf_model_dir(target_dir):
            raise RuntimeError(
                "Model checkpoint merge completed, but the export at "
                f"{target_dir} is not a HuggingFace model directory."
            )
        marker_path.write_text(str(checkpoint_dir.resolve()), encoding="utf-8")
        return target_dir, True

    def _checkpoint_merge_dir(self, checkpoint_path: Path) -> Path:
        subdir = self.server_config.get("checkpoint_subdir", "actor")
        if subdir:
            candidate = checkpoint_path / str(subdir)
            if candidate.exists():
                return candidate
        if (checkpoint_path / "actor").exists():
            return checkpoint_path / "actor"
        return checkpoint_path

    def _merge_command(self, checkpoint_dir: Path, target_dir: Path) -> List[str]:
        backend = str(self.server_config.get("checkpoint_backend", "fsdp"))
        cmd = [
            self.python_executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            backend,
            "--local_dir",
            str(checkpoint_dir),
            "--target_dir",
            str(target_dir),
        ]
        if _as_bool(self.server_config.get("trust_remote_code", False)):
            cmd.append("--trust-remote-code")
        if _as_bool(self.server_config.get("use_cpu_initialization", False)):
            cmd.append("--use_cpu_initialization")
        return cmd

    def _server_command(self, model_path: Path) -> List[str]:
        cmd = [
            self.python_executable,
            "-m",
            str(
                self.server_config.get(
                    "server_module", "vllm.entrypoints.openai.api_server"
                )
            ),
            "--model",
            str(model_path),
            "--served-model-name",
            self.served_model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            str(
                self.server_config.get(
                    "dtype", self.config.get("rollout_dtype", "bfloat16")
                )
            ),
            "--tensor-parallel-size",
            str(
                self.server_config.get(
                    "tensor_parallel_size", self.config.get("rollout_tp", 1)
                )
            ),
            "--gpu-memory-utilization",
            str(
                self.server_config.get(
                    "gpu_memory_utilization",
                    self.config.get("rollout_gpu_memory_utilization", 0.6),
                )
            ),
        ]
        max_model_len = self.server_config.get("max_model_len")
        if max_model_len is not None:
            cmd.extend(["--max-model-len", str(max_model_len)])
        if _as_bool(self.server_config.get("trust_remote_code", False)):
            cmd.append("--trust-remote-code")
        extra_args = self.server_config.get("extra_args", [])
        if isinstance(extra_args, str):
            extra_args = extra_args.split()
        cmd.extend(str(item) for item in extra_args)
        return cmd

    def _server_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        configured_env = self.server_config.get("env", {})
        if isinstance(configured_env, dict):
            env.update({str(key): str(value) for key, value in configured_env.items()})

        cuda_devices = os.environ.get("ATTACKER_SERVER_CUDA_VISIBLE_DEVICES")
        if not cuda_devices:
            cuda_devices = self.server_config.get("cuda_visible_devices")
        if str(cuda_devices).strip().casefold() == "auto":
            cuda_devices = self._auto_cuda_visible_devices()
        if cuda_devices not in (None, ""):
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
            logger.info("Attacker server CUDA_VISIBLE_DEVICES=%s", cuda_devices)
        return env

    def _auto_cuda_visible_devices(self) -> str:
        visible_devices = _visible_cuda_devices()
        training_gpus = int(self.config.get("training_n_gpus_per_node", 1))
        server_gpus = int(self.server_config.get("tensor_parallel_size", 1))
        start = training_gpus
        selected = visible_devices[start : start + server_gpus]
        if len(selected) != server_gpus:
            raise RuntimeError(
                "The managed attacker server needs "
                f"{server_gpus} GPU(s) in addition to the {training_gpus} GPU(s) "
                f"reserved for training, but only {len(visible_devices)} visible "
                "GPU(s) were found. Set ATTACKER_SERVER_CUDA_VISIBLE_DEVICES to "
                "dedicated GPU IDs, reduce training GPUs, or disable the managed "
                "server and provide a reachable attacker_api_base."
            )
        return ",".join(selected)

    def _wait_until_ready(
        self,
        expected_model: Optional[str],
        managed: bool,
    ) -> None:
        timeout = float(
            self.server_config.get(
                "startup_timeout" if managed else "external_check_timeout",
                600 if managed else 15,
            )
        )
        interval = float(self.server_config.get("health_check_interval", 5))
        deadline = time.time() + timeout
        models_url = f"{self.api_base}/models"
        last_error: Optional[Exception] = None

        while time.time() < deadline:
            if managed and self.process and self.process.poll() is not None:
                raise RuntimeError(
                    "Attacker server exited before becoming ready. See log: "
                    f"{self.log_file.name if self.log_file else 'unknown'}"
                )
            try:
                request = urllib.request.Request(
                    models_url,
                    headers=self._request_headers(),
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                model_ids = _model_ids(payload)
                if expected_model and model_ids and expected_model not in model_ids:
                    last_error = RuntimeError(
                        f"endpoint serves {model_ids}, not {expected_model}"
                    )
                else:
                    logger.info("Attacker server ready at %s", self.api_base)
                    return
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
            time.sleep(interval)

        mode = "managed" if managed else "external"
        detail = f": {last_error}" if last_error else ""
        raise RuntimeError(
            f"The {mode} attacker API is not ready at {models_url}{detail}"
        )

    def _request_headers(self) -> Dict[str, str]:
        api_key = str(self.config.get("attacker_api_key") or "").strip()
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _log_path(self, round_index: int) -> Path:
        return self.log_dir / f"attacker_server_round_{round_index:03d}.log"


def _visible_cuda_devices() -> List[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured and configured.strip() not in {"-1", ""}:
        return [item.strip() for item in configured.split(",") if item.strip()]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _is_hf_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    patterns = ("*.safetensors", "pytorch_model*.bin", "model*.bin")
    return any(any(path.glob(pattern)) for pattern in patterns)


def _model_ids(payload: Dict[str, Any]) -> List[str]:
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    return [
        str(item.get("id"))
        for item in data
        if isinstance(item, dict) and item.get("id")
    ]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "on"}


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
