"""Client helper that talks to an RL environment server over a Unix socket."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rlsimlink.utils import Colors, print_log, set_log_socket_path
from .common import ActionSpace, SocketManager
from .env_runtime import EnvServerLauncher
from .socket_paths import generate_socket_id, resolve_socket_path


class RLEnv:
    """Gymnasium-like interface that forwards operations to a remote RL server."""

    def __init__(
        self,
        env_type: str,
        env_name: str,
        seed: Optional[int] = None,
        socket_wait_timeout: float = 15.0,
        **kwargs,
    ):
        """Initialize RL environment client.

        Args:
            env_type: Environment type (e.g., "atari", "dmc", "metaworld")
            env_name: Environment name (e.g., "BoxingNoFrameskip-v4")
            seed: Random seed forwarded to the server when reset() is first called
            socket_wait_timeout: Seconds to wait while polling for the socket
            **kwargs: Additional environment-specific arguments (image_size, etc.)
        """
        print_log("INFO", f"Initializing RLEnv: {Colors.PURPLE}{env_type}/{env_name}{Colors.ENDC}")

        self.env_type = env_type
        self.env_name = env_name

        socket_id = generate_socket_id()
        socket_path = resolve_socket_path(socket_id, create_parent=True)

        self.socket_path = str(socket_path)
        self.socket_id = socket_id
        self.seed = seed
        self.env_kwargs = kwargs
        self.socket_wait_timeout = socket_wait_timeout
        self._initialized = False
        self._server_started = False
        self._launcher = EnvServerLauncher(env_type)
        self._socket_manager = SocketManager(
            self.socket_path, self.socket_wait_timeout, log_fn=print_log, role="client"
        )

        print_log("INFO", f"Using socket path: {Colors.ORANGE}{self.socket_path}{Colors.ENDC}")

        # Initialize logger for client with socket path
        set_log_socket_path(self.socket_path, log_type="client")

        # Always start the server when creating an RLEnv client
        self._start_server()

        # Connect to socket
        self._connect()

        # Get action space information and create ActionSpace object
        self.action_space_info = self._socket_manager.get_action_space(
            self.env_type, self.env_name, self.seed, self.env_kwargs
        )
        self.action_space = ActionSpace(self.action_space_info, expand_dim=False)

    def _start_server(self):
        """Start the environment server through the launcher."""
        try:
            self._launcher.start(self.socket_id)
            self._server_started = True
        except Exception as exc:
            print_log("ERROR", f"Failed to auto-start server: {exc}")
            raise

    def _connect(self):
        """Wait for the socket to be ready, then connect."""
        self._socket_manager.connect()

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

        response = self._socket_manager.send_request(message)

        if response.get("status") != "ok":
            print_log("ERROR", f"Failed to reset environment: {response.get('message')}")
            raise RuntimeError(f"Failed to reset environment: {response.get('message')}")

        obs = self._socket_manager.extract_observation(response)
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
        serialized_action = self._socket_manager.serialize_action(action)

        message = {
            "operation": "step",
            "env_type": self.env_type,
            "env_name": self.env_name,
            "action": serialized_action,
        }
        response = self._socket_manager.send_request(message)

        if response.get("status") != "ok":
            print_log("ERROR", f"Failed to step environment: {response.get('message')}")
            raise RuntimeError(f"Failed to step environment: {response.get('message')}")

        obs = self._socket_manager.extract_observation(response)
        reward = float(response.get("reward", 0.0))
        terminated = bool(response.get("terminated", False))
        truncated = bool(response.get("truncated", False))
        info = response.get("info", {})

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
