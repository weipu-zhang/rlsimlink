"""Environment-specific runtime launcher for starting RL servers."""

from __future__ import annotations

import os
import subprocess
from configparser import ConfigParser
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

from rlsimlink.utils import Colors, print_log

DEFAULT_DOCKER_WORKSPACE = "/root/rlsimlink"
DEFAULT_DOCKER_HASH = "manual"


@dataclass(frozen=True)
class EnvBackendConfig:
    """Configuration for an environment backend."""

    name: str
    backend: str
    conda_env: Optional[str] = None
    docker_image: Optional[str] = None
    container_name: Optional[str] = None
    hash_code: Optional[str] = None
    workspace_dir: Optional[str] = None
    auto_install: bool = True

    def resolved_conda_env(self, fallback: str) -> str:
        return self.conda_env or fallback

    def resolved_workspace(self) -> str:
        return self.workspace_dir or DEFAULT_DOCKER_WORKSPACE

    def resolved_container(self, env_type: str, identifier: Optional[str] = None) -> str:
        if self.container_name:
            return self.container_name
        tag = _infer_tag_from_image(self.docker_image, env_type)
        name_prefix = identifier or self.hash_code or DEFAULT_DOCKER_HASH
        return f"{name_prefix}-rlsimlink-{tag}"


def _infer_tag_from_image(image: Optional[str], fallback: str) -> str:
    """Derive a docker tag from an image string."""

    if not image:
        return fallback

    if "@" in image:
        image = image.split("@", 1)[0]

    repository, sep, tag = image.rpartition(":")
    if not sep:
        return fallback
    return tag or fallback


def _candidate_registry_paths() -> list[Path]:
    """Possible locations for the env-registry configuration."""

    candidates = []
    xdg_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_home:
        candidates.append(Path(xdg_home).expanduser() / "rlsimlink" / "env-registry.conf")
    candidates.append(Path.home() / ".config" / "rlsimlink" / "env-registry.conf")
    candidates.append(Path(__file__).resolve().parents[2] / "env-registry.conf")

    unique_paths: list[Path] = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def _load_registry_from(path: Path) -> Dict[str, EnvBackendConfig]:
    parser = ConfigParser()
    parser.read(path, encoding="utf-8")

    registry: Dict[str, EnvBackendConfig] = {}
    for section in parser.sections():
        backend = parser.get(section, "backend", fallback="").strip().lower()
        if not backend:
            raise ValueError(f"Section '{section}' missing backend in {path}")

        registry[section.lower()] = EnvBackendConfig(
            name=section,
            backend=backend,
            conda_env=parser.get(section, "conda_env", fallback=None),
            docker_image=parser.get(section, "docker_image", fallback=None),
            container_name=parser.get(section, "container_name", fallback=None),
            hash_code=parser.get(section, "hash_code", fallback=None),
            workspace_dir=parser.get(section, "workspace_dir", fallback=None),
            auto_install=parser.getboolean(section, "auto_install", fallback=True),
        )

    if not registry:
        raise ValueError(f"No environments defined in registry file {path}")

    return registry


@lru_cache(maxsize=1)
def _load_env_registry() -> Tuple[Dict[str, EnvBackendConfig], Path]:
    """Load environment backend definitions from disk."""

    for candidate in _candidate_registry_paths():
        if candidate.exists():
            registry = _load_registry_from(candidate)
            return registry, candidate

    search_paths = ", ".join(str(p) for p in _candidate_registry_paths())
    raise FileNotFoundError(f"env-registry.conf not found. Looked in: {search_paths}")


def _get_backend_config(env_type: str) -> EnvBackendConfig:
    registry, source = _load_env_registry()
    key = env_type.lower()
    try:
        return registry[key]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            f"env_type '{env_type}' is not configured in {source}. " "Add a matching section to select a backend."
        ) from exc


def _ensure_container_running(container_name: str):
    status_cmd = [
        "docker",
        "container",
        "inspect",
        "-f",
        "{{.State.Status}}",
        container_name,
    ]
    result = subprocess.run(status_cmd, capture_output=True, text=True, check=False)
    status = result.stdout.strip()
    if result.returncode != 0 or status != "running":
        detail = result.stderr.strip() or status or "unknown"
        raise RuntimeError(
            f"Container '{container_name}' is not running (details: {detail}). "
            "Start it via `rlsimlink docker start <env_type>` before launching an environment."
        )


def _is_container_running(container_name: str) -> bool:
    status_cmd = [
        "docker",
        "container",
        "inspect",
        "-f",
        "{{.State.Status}}",
        container_name,
    ]
    result = subprocess.run(status_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return False
    return result.stdout.strip() == "running"


def _start_container(container_name: str, image: Optional[str], workspace_dir: str):
    if not image:
        raise RuntimeError("docker_image must be specified in env-registry for docker backends.")

    if _is_container_running(container_name):
        raise RuntimeError(
            f"Container '{container_name}' is already running. Please stop it before launching a new environment."
        )

    repo_root = Path(__file__).resolve().parents[2]
    shm_dir = Path("/dev/shm/rlsimlink")
    shm_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "docker",
        "run",
        "--rm",
        "-dit",
        "--name",
        container_name,
        "--hostname",
        os.environ.get("HOSTNAME", "rlsimlink-container"),
        "--network=host",
        "--shm-size=8g",
        "--mount",
        f"type=bind,source={repo_root},target={workspace_dir}",
        "--mount",
        f"type=bind,source={shm_dir},target=/dev/shm/rlsimlink",
    ]

    # Enable GPUs if available (matching the docker image expectation for CUDA)
    command.append("--gpus=all")

    command.append(image)

    print_log(
        "INFO",
        f"Starting docker container '{container_name}' from image '{image}'",
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"Failed to start container '{container_name}': {detail}")
    print_log("SUCCESS", f"Container '{container_name}' started successfully.")


class EnvServerLauncher:
    """Launches environment servers based on env_type."""

    def __init__(self, env_type: str):
        self.env_type = env_type
        self.process: Optional[subprocess.Popen] = None
        self.socket_id: Optional[str] = None
        self.socket_path: Optional[str] = None
        self._backend_config = _get_backend_config(env_type)

    def _start_via_conda(self, socket_id: str):
        env_name = self._backend_config.resolved_conda_env(self.env_type)
        command = ["conda", "run", "-n", env_name, "rlsimlink", "serve", "--socket-id", socket_id]

        print_log(
            "LINK",
            f"Launching {Colors.PURPLE}{self.env_type}{Colors.ENDC} server via conda env '{env_name}'",
        )

        # Conda launcher spawns a child process we need to track so close() can terminate it.
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _run_docker_install(self, container_name: str, workspace_dir: str, env_name: str):
        install_cmd = [
            "docker",
            "exec",
            container_name,
            "bash",
            "-c",
            f"cd {workspace_dir} && pip install -e .",
        ]
        print_log(
            "INFO",
            f"Installing rlsimlink inside container '{container_name}' env '{env_name}'...",
        )
        result = subprocess.run(install_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            error_detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"Failed to install package in container '{container_name}' env '{env_name}': {error_detail}"
            )
        print_log("SUCCESS", f"Package installed inside container '{container_name}'")

    def _start_via_docker(self, socket_id: str):
        container_name = self._backend_config.resolved_container(self.env_type, identifier=socket_id)
        workspace_dir = self._backend_config.resolved_workspace()
        env_name = self._backend_config.resolved_conda_env(self.env_type)
        docker_image = self._backend_config.docker_image

        _start_container(container_name, docker_image, workspace_dir)
        _ensure_container_running(container_name)

        if self._backend_config.auto_install:
            self._run_docker_install(container_name, workspace_dir, env_name)

        server_cmd = [
            "docker",
            "exec",
            "-d",
            container_name,
            "bash",
            "-c",
            f"cd {workspace_dir} && rlsimlink serve --socket-id {socket_id}",
        ]

        print_log(
            "LINK",
            f"Launching {Colors.PURPLE}{self.env_type}{Colors.ENDC} server inside container '{container_name}'",
        )

        # docker exec -d returns immediately; we shell out to track completion errors.
        result = subprocess.run(server_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"Failed to start server inside container '{container_name}': {detail}")

        # Store a dummy Popen-like sentinel so stop() no-ops for docker backends.
        self.process = None

    def start(self, socket_id: str):
        """Start the server process if not already running."""
        backend = self._backend_config.backend
        if backend == "conda" and self.process and self.process.poll() is None:
            return

        self.socket_id = socket_id

        if backend == "conda":
            self._start_via_conda(socket_id)
        elif backend == "docker":
            self._start_via_docker(socket_id)
        else:
            raise ValueError(f"Unsupported backend '{backend}' for env_type {self.env_type}")

    def stop(self):
        """Stop the launched server process."""
        if self.process is None:
            return

        if self.process.poll() is None:
            print_log("INFO", f"Stopping auto-launched {self.env_type} server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print_log("WARN", "Server did not exit in time, killing process.")
                self.process.kill()

        self.process = None
        self.socket_id = None
        self.socket_path = None
