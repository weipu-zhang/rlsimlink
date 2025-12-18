"""Shared socket communication utilities for client and server."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from rlsimlink.utils import Colors, print_log

LogFn = Callable[[str, str], None]


class SocketManager:
    """Utility class for establishing and managing Unix socket connections."""

    def __init__(
        self,
        socket_path: str,
        timeout: float = 15.0,
        log_fn: Optional[LogFn] = None,
        role: str = "client",
    ):
        if role not in {"client", "server"}:
            raise ValueError("SocketManager role must be either 'client' or 'server'")

        self.socket_path = socket_path
        self.timeout = timeout
        self.log_fn = log_fn or print_log
        self.role = role
        self.socket: Optional[socket.socket] = None

    def _log(self, level: str, message: str) -> None:
        if self.log_fn:
            self.log_fn(level, message)

    def wait_for_socket(self) -> None:
        """Block until the Unix socket exists and accepts connections."""
        self._log("INFO", f"Waiting for socket to be ready (timeout: {self.timeout}s)...")
        deadline = time.time() + self.timeout
        socket_file = Path(self.socket_path)

        while time.time() < deadline:
            if socket_file.exists():
                test_socket: Optional[socket.socket] = None
                try:
                    test_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    test_socket.settimeout(0.2)
                    test_socket.connect(self.socket_path)
                    test_socket.close()
                    self._log("SUCCESS", "Socket is ready for connections")
                    return
                except (ConnectionRefusedError, FileNotFoundError, OSError):
                    if test_socket:
                        test_socket.close()
                    time.sleep(0.2)
                    continue
            time.sleep(0.2)

        self._log("ERROR", f"Socket not ready after {self.timeout} seconds")
        raise RuntimeError(f"Socket at {self.socket_path} not ready after {self.timeout} seconds")

    def connect(self) -> socket.socket:
        """Wait for the socket to be ready and establish a connection."""
        self.wait_for_socket()

        self._log("INFO", f"Connecting to Unix socket at {self.socket_path}...")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.socket_path)
            self._log("LINK", f"Connected to Unix socket at {Colors.ORANGE}{self.socket_path}{Colors.ENDC}")
            self.socket = sock
            return sock
        except ConnectionRefusedError as exc:
            self._log("ERROR", "Connection refused")
            sock.close()
            raise RuntimeError(
                f"Could not connect to Unix socket at {self.socket_path}. Make sure `rlsimlink serve` is running."
            ) from exc
        except FileNotFoundError as exc:
            self._log("ERROR", "Socket file not found")
            sock.close()
            raise RuntimeError(
                f"Unix socket not found at {self.socket_path}. Make sure `rlsimlink serve` created the socket and is running."
            ) from exc

    def close(self) -> None:
        """Close the managed socket connection."""
        if self.socket:
            self._log("INFO", "Closing socket connection...")
            self.socket.close()
            self.socket = None
            self._log("SUCCESS", "Socket closed successfully")

    def send_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request message and wait for the response."""
        if self.role != "client":
            raise RuntimeError("send_request is only available in client role")
        if self.socket is None:
            raise RuntimeError("Socket not connected")

        self.send_json_message(self.socket, message)
        response = self.receive_json_message(self.socket)

        if response is None:
            raise RuntimeError("Failed to receive response from server")

        return response

    def get_action_space(
        self,
        env_type: str,
        env_name: str,
        seed: Optional[int],
        env_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Request action space information for an environment."""
        if self.role != "client":
            raise RuntimeError("get_action_space is only available in client role")

        self._log("INFO", f"Getting action space for: {Colors.BOLD}{env_type}/{env_name}{Colors.ENDC}")
        message: Dict[str, Any] = {
            "operation": "get_action_space",
            "env_type": env_type,
            "env_name": env_name,
            "seed": seed,
        }

        if env_kwargs:
            message.update(env_kwargs)

        response = self.send_request(message)

        if response.get("status") != "ok":
            self._log("ERROR", f"Failed to get action space: {response.get('message')}")
            raise RuntimeError(f"Failed to get action space: {response.get('message')}")

        action_space_info = response.get("action_space", {})
        self._log("INFO", f"Action space retrieved: {action_space_info}")

        return action_space_info

    def send_json(self, message: Dict[str, Any]) -> None:
        """Send a JSON message over the managed socket."""
        if self.socket is None:
            raise RuntimeError("Socket not connected")
        self.send_json_message(self.socket, message)

    def receive_json(self) -> Optional[Dict[str, Any]]:
        """Receive a JSON message over the managed socket."""
        if self.socket is None:
            raise RuntimeError("Socket not connected")
        return self.receive_json_message(self.socket)

    @staticmethod
    def _recv_exact(sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        data = b""
        while len(data) < num_bytes:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    @staticmethod
    def receive_json_message(sock: socket.socket) -> Optional[Dict[str, Any]]:
        """Receive a JSON message prefixed by its length."""
        length_bytes = SocketManager._recv_exact(sock, 4)
        if not length_bytes:
            return None

        message_length = int.from_bytes(length_bytes, byteorder="big")
        data = SocketManager._recv_exact(sock, message_length)
        if data is None:
            return None

        return json.loads(data.decode("utf-8"))

    @staticmethod
    def send_json_message(sock: socket.socket, message: Dict[str, Any]) -> None:
        """Send a JSON message with a 4-byte length prefix."""
        payload = json.dumps(message).encode("utf-8")
        message_length = len(payload)
        sock.sendall(message_length.to_bytes(4, byteorder="big"))
        sock.sendall(payload)

    @staticmethod
    def serialize_action(action: Any) -> Any:
        """Serialize action to JSON-friendly formats."""
        if isinstance(action, np.ndarray):
            return action.tolist()
        if isinstance(action, (int, np.integer)):
            return [int(action)]
        if isinstance(action, list):
            serialized = []
            for item in action:
                if isinstance(item, (np.integer, int)):
                    serialized.append(int(item))
                elif isinstance(item, (np.floating, float)):
                    serialized.append(float(item))
                else:
                    serialized.append(item)
            return serialized
        raise ValueError(f"Unsupported action type: {type(action)}")

    @staticmethod
    def deserialize_observation(obs_data: Any) -> np.ndarray:
        """Deserialize observation data into a numpy array."""
        if isinstance(obs_data, list):
            return np.array(obs_data, dtype=np.float32)
        return np.array(obs_data, dtype=np.float32)

    @staticmethod
    def load_observation_from_shm(obs_path: Optional[str]) -> np.ndarray:
        """Load observation data stored at a shared-memory path."""
        if not obs_path:
            raise RuntimeError("Observation path missing in response")

        obs_file = Path(obs_path)
        if not obs_file.exists():
            raise RuntimeError(f"Observation file not found at {obs_path}")

        with open(obs_file, "rb") as f:
            return np.load(f, allow_pickle=False)


__all__ = ["SocketManager"]
