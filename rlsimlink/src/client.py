"""Client helper that talks to an RL environment server over a Unix socket."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rlsimlink.utils import Colors, print_log, set_log_socket_path
from .common import ActionSpace, SocketManager
from .env_runtime import EnvServerLauncher
from .socket_paths import extract_socket_id, generate_socket_id, resolve_socket_path


class RLEnv:
    """Gymnasium-like interface that forwards operations to a remote RL server."""

    def __init__(
        self,
        env_type: str,
        env_name: str,
        socket_path: Optional[str] = None,
        seed: Optional[int] = None,
        socket_id: Optional[str] = None,
        socket_wait_timeout: float = 15.0,
        **kwargs,
    ):
        """Initialize RL environment client.

        Args:
            env_type: Environment type (e.g., "atari", "dmc", "metaworld")
            env_name: Environment name (e.g., "BoxingNoFrameskip-v4")
            socket_path: Optional explicit Unix socket path
            seed: Random seed forwarded to the server when reset() is first called
            socket_id: Optional socket identifier (maps to /dev/shm/rlsimlink/<id>/socket)
            socket_wait_timeout: Seconds to wait while polling for the socket
            **kwargs: Additional environment-specific arguments (image_size, etc.)
        """
        print_log("INFO", f"Initializing RLEnv: {Colors.PURPLE}{env_type}/{env_name}{Colors.ENDC}")

        self.env_type = env_type
        self.env_name = env_name

        if socket_path is None and socket_id is None:
            socket_id = generate_socket_id()

        if socket_path is None:
            resolved_path = resolve_socket_path(socket_id, create_parent=True)
            socket_path = str(resolved_path)
        elif socket_id is not None:
            expected_path = resolve_socket_path(socket_id, create_parent=False)
            if expected_path != Path(socket_path).expanduser().resolve():
                raise ValueError("socket_id and socket_path refer to different locations.")

        self.socket_path = str(Path(socket_path).expanduser().resolve())
        self.socket_id = socket_id or extract_socket_id(self.socket_path)
        self.seed = seed
        self.env_kwargs = kwargs
        self.socket_wait_timeout = socket_wait_timeout
        self._initialized = False
        self._server_started = False
        self._launcher = EnvServerLauncher(env_type)
        self._socket_manager = SocketManager(self.socket_path, self.socket_wait_timeout, log_fn=print_log)

        print_log("INFO", f"Using socket path: {Colors.ORANGE}{self.socket_path}{Colors.ENDC}")

        # Initialize logger for client with socket path
        set_log_socket_path(self.socket_path, log_type="client")

        # Always start the server when creating an RLEnv client
        self._start_server()

        # Connect to socket
        self._connect()

        # Get action space information and create ActionSpace object
        self.action_space_info = self._get_action_space()
        self.action_space = ActionSpace(self.action_space_info, expand_dim=False)

    def _start_server(self):
        """Start the environment server through the launcher."""
        try:
            self._launcher.start(self.socket_path)
            self._server_started = True
        except Exception as exc:
            print_log("ERROR", f"Failed to auto-start server: {exc}")
            raise

    def _connect(self):
        """Wait for the socket to be ready, then connect."""
        self._socket_manager.connect()

    def _get_action_space(self) -> Dict[str, Any]:
        """Get action space information from the server."""
        print_log("INFO", f"Getting action space for: {Colors.BOLD}{self.env_type}/{self.env_name}{Colors.ENDC}")

        message = {
            "operation": "get_action_space",
            "env_type": self.env_type,
            "env_name": self.env_name,
            "seed": self.seed,
        }

        # Add environment-specific kwargs
        for key, value in self.env_kwargs.items():
            message[key] = value

        response = self._send_message(message)

        if response.get("status") != "ok":
            print_log("ERROR", f"Failed to get action space: {response.get('message')}")
            raise RuntimeError(f"Failed to get action space: {response.get('message')}")

        action_space_info = response.get("action_space", {})
        print_log("INFO", f"Action space retrieved: {action_space_info}")

        return action_space_info

    def _send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send message to server and receive response."""
        if self._socket_manager.socket is None:
            raise RuntimeError("Socket not connected")

        self._socket_manager.send_json(message)
        response = self._socket_manager.receive_json()

        if response is None:
            raise RuntimeError("Failed to receive response from server")

        return response

    def _serialize_action(self, action: Any) -> Any:
        """Serialize action to JSON-serializable format."""
        if isinstance(action, np.ndarray):
            # Convert numpy array to list
            return action.tolist()
        elif isinstance(action, (int, np.integer)):
            # Convert single integer to list
            return [int(action)]
        elif isinstance(action, list):
            # Already a list, ensure all elements are JSON-serializable
            return [int(x) if isinstance(x, (np.integer, np.floating)) else x for x in action]
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

    def _deserialize_observation(self, obs_data: Any) -> np.ndarray:
        """Deserialize observation from JSON to numpy array."""
        if isinstance(obs_data, list):
            return np.array(obs_data, dtype=np.float32)
        else:
            return np.array(obs_data, dtype=np.float32)

    def _load_observation_from_shm(self, obs_path: Optional[str]) -> np.ndarray:
        """Load observation saved in shared memory."""
        if not obs_path:
            raise RuntimeError("Observation path missing in response")

        obs_file = Path(obs_path)
        if not obs_file.exists():
            raise RuntimeError(f"Observation file not found at {obs_path}")

        with open(obs_file, "rb") as f:
            return np.load(f, allow_pickle=False)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        print_log("INFO", f"Resetting environment: {Colors.BOLD}{self.env_type}/{self.env_name}{Colors.ENDC}")

        message = {
            "operation": "reset",
            "env_type": self.env_type,
            "env_name": self.env_name,
            "seed": self.seed,
            "kwargs": kwargs,
        }

        # Add environment-specific kwargs
        for key, value in self.env_kwargs.items():
            message[key] = value

        response = self._send_message(message)

        if response.get("status") != "ok":
            print_log("ERROR", f"Failed to reset environment: {response.get('message')}")
            raise RuntimeError(f"Failed to reset environment: {response.get('message')}")

        obs_path = response.get("observation_path")
        obs = (
            self._load_observation_from_shm(obs_path)
            if obs_path is not None
            else self._deserialize_observation(response.get("observation"))
        )
        info = response.get("info", {})
        self._initialized = True

        print_log("INFO", f"Environment reset successfully (obs shape: {obs.shape})")

        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step in the environment."""
        if not self._initialized:
            print_log("ERROR", "Environment not initialized")
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Serialize action to JSON-serializable format
        serialized_action = self._serialize_action(action)

        message = {
            "operation": "step",
            "env_type": self.env_type,
            "env_name": self.env_name,
            "action": serialized_action,
        }
        response = self._send_message(message)

        if response.get("status") != "ok":
            print_log("ERROR", f"Failed to step environment: {response.get('message')}")
            raise RuntimeError(f"Failed to step environment: {response.get('message')}")

        obs_path = response.get("observation_path")
        obs = (
            self._load_observation_from_shm(obs_path)
            if obs_path is not None
            else self._deserialize_observation(response.get("observation"))
        )
        reward = float(response.get("reward", 0.0))
        terminated = bool(response.get("terminated", False))
        truncated = bool(response.get("truncated", False))
        info = response.get("info", {})

        # Color code the reward output
        reward_color = Colors.OKGREEN if reward > 0 else Colors.FAIL if reward < 0 else Colors.ENDC
        status_str = f"reward={reward_color}{reward:.3f}{Colors.ENDC}"

        if terminated:
            status_str += f" {Colors.WARNING}[TERMINATED]{Colors.ENDC}"
        if truncated:
            status_str += f" {Colors.WARNING}[TRUNCATED]{Colors.ENDC}"

        print_log("STEP", f"action={action} {status_str}")

        return obs, reward, terminated, truncated, info

    def close(self):
        """Close the environment and socket connection."""
        self._socket_manager.close()

        if self._server_started:
            self._launcher.stop()
            self._server_started = False

        # Clean up socket file if it exists
        if self.socket_path and os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
                print_log("INFO", f"Socket file removed: {self.socket_path}")
            except OSError as e:
                print_log("WARNING", f"Failed to remove socket file {self.socket_path}: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @property
    def socket(self):
        """Expose the underlying socket managed by SocketManager."""
        return self._socket_manager.socket
