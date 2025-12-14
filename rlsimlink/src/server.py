#!/usr/bin/env python3
"""Unix socket server that exposes RL environments to external processes."""

import socket
import json
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Import color utilities from the shared logger utilities module
from rlsimlink.utils import Colors, print_log, set_log_socket_path
from .envs import create_env_manager
from .socket_paths import extract_socket_id, resolve_observation_path


class RLEnvServer:
    """Universal server for managing a single RL environment via Unix socket."""

    def __init__(self, socket_path: str):
        """Initialize the RL environment server.

        Args:
            socket_path: Path to Unix socket (must be provided via SOCKET_PATH env var)
        """
        if not socket_path:
            print_log("ERROR", "SOCKET_PATH environment variable is required")
            raise ValueError("SOCKET_PATH environment variable must be set")

        self.socket_path = socket_path
        self.socket_id = self._extract_socket_id(socket_path)
        self.obs_file_path = self._build_obs_path()
        self.env_manager = None  # Environment manager instance
        self.env_type = None
        self.env_name = None
        self.server_socket = None
        self.running = False

        # Initialize logger with socket path
        set_log_socket_path(socket_path)

        print_log("INFO", f"Initializing server with socket path: {Colors.BOLD}{socket_path}{Colors.ENDC}")

        # Ensure socket directory exists
        socket_dir = Path(socket_path).parent
        socket_dir.mkdir(parents=True, exist_ok=True)
        print_log("INFO", f"Socket directory: {socket_dir}")

        # Remove existing socket file if it exists
        if os.path.exists(socket_path):
            print_log("WARNING", f"Removing existing socket file: {socket_path}")
            os.unlink(socket_path)

    def _extract_socket_id(self, socket_path: str) -> str:
        """Derive socket identifier from socket path."""
        return extract_socket_id(socket_path)

    def _build_obs_path(self) -> Path:
        """Build shared-memory observation file path."""
        if self.socket_id != "manual":
            return resolve_observation_path(self.socket_id, create_parent=True)

        base_dir = Path(self.socket_path).parent
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / "obs"

    def _save_observation(self, obs: Any) -> str:
        """Persist observation to shared memory and return the path."""
        try:
            self.obs_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.obs_file_path, "wb") as f:
                np.save(f, obs, allow_pickle=False)
            return str(self.obs_file_path)
        except Exception as exc:
            print_log("ERROR", f"Failed to save observation: {exc}")
            raise

    def create_environment(self, env_type: str, env_name: str, seed: Optional[int] = None, **kwargs):
        """Create the environment instance.

        Args:
            env_type: Environment type (e.g., "atari")
            env_name: Environment name (e.g., "BoxingNoFrameskip-v4")
            seed: Random seed (optional)
            **kwargs: Additional environment-specific arguments
        """
        print_log("INFO", f"Creating environment: {Colors.BOLD}{env_type}/{env_name}{Colors.ENDC}")
        if seed is not None:
            print_log("INFO", f"Using seed: {seed}")

        try:
            # Create environment manager for the given type
            self.env_manager = create_env_manager(env_type)

            # Handle environment-specific parameters
            if env_type == "atari":
                image_size = kwargs.get("image_size", None)
                if image_size is not None:
                    image_size = tuple(image_size)
                    print_log("INFO", f"Image size: {image_size}")
                self.env_manager.create(env_name, seed, image_size)
            else:
                # For future environment types
                self.env_manager.create(env_name, seed, **kwargs)

            self.env_type = env_type
            self.env_name = env_name
            print_log("SUCCESS", f"Environment created successfully: {env_type}/{env_name}")
        except Exception as e:
            print_log("ERROR", f"Failed to create environment {env_type}:{env_name}: {e}")
            raise RuntimeError(f"Failed to create environment {env_type}:{env_name}: {e}")

    def serialize_observation(self, obs: Any) -> Any:
        """Serialize observation (numpy array) to JSON-serializable format.

        Args:
            obs: Observation (numpy array or other)

        Returns:
            JSON-serializable observation
        """
        if isinstance(obs, np.ndarray):
            return obs.tolist()
        elif hasattr(obs, "tolist"):
            return obs.tolist()
        else:
            return obs

    def deserialize_action(self, action_data: Any) -> Any:
        """Deserialize action from JSON format to appropriate type.

        Args:
            action_data: Action data (typically a list from JSON)

        Returns:
            Deserialized action (list or single value depending on action space)
        """
        if isinstance(action_data, list):
            # For single discrete action, unwrap the list
            if len(action_data) == 1:
                return action_data[0]
            # For multi-dimensional actions, keep as list
            return action_data
        else:
            # Already in correct format
            return action_data

    def serialize_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize info dictionary to JSON-serializable format.

        Args:
            info: Info dictionary

        Returns:
            JSON-serializable info dictionary
        """
        serialized = {}
        for key, value in info.items():
            if isinstance(value, (np.ndarray, np.generic)):
                serialized[key] = value.tolist() if hasattr(value, "tolist") else float(value)
            elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized

    def handle_reset(self, **kwargs) -> Dict[str, Any]:
        """Reset the environment.

        Args:
            **kwargs: Additional reset arguments

        Returns:
            Response dictionary with observation and info
        """
        if self.env_manager is None:
            print_log("ERROR", "Environment not created")
            return {"status": "error", "message": "Environment not created"}

        print_log("INFO", f"Resetting environment: {Colors.BOLD}{self.env_type}/{self.env_name}{Colors.ENDC}")
        observation, info = self.env_manager.reset(**kwargs)

        print_log(
            "SUCCESS",
            f"Environment reset successfully (obs shape: {observation.shape if hasattr(observation, 'shape') else 'N/A'})",
        )

        return {
            "status": "ok",
            "observation_path": self._save_observation(observation),
            "info": self.serialize_info(info),
        }

    def handle_step(self, action: Any) -> Dict[str, Any]:
        """Step in the environment.

        Args:
            action: Action to take (serialized from client)

        Returns:
            Response dictionary with step results
        """
        if self.env_manager is None:
            print_log("ERROR", "Environment not created")
            return {"status": "error", "message": "Environment not created"}

        # Deserialize action from JSON format
        deserialized_action = self.deserialize_action(action)

        observation, reward, terminated, truncated, info = self.env_manager.step(deserialized_action)

        # Color code the reward output
        reward_color = Colors.OKGREEN if reward > 0 else Colors.FAIL if reward < 0 else Colors.ENDC
        status_str = f"reward={reward_color}{reward:.3f}{Colors.ENDC}"

        if terminated:
            status_str += f" {Colors.WARNING}[TERMINATED]{Colors.ENDC}"
        if truncated:
            status_str += f" {Colors.WARNING}[TRUNCATED]{Colors.ENDC}"

        print_log("STEP", f"action={deserialized_action} {status_str}")

        return {
            "status": "ok",
            "observation_path": self._save_observation(observation),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": self.serialize_info(info),
        }

    def handle_get_action_space(self, env_type: str, env_name: str, **kwargs) -> Dict[str, Any]:
        """Get action space information without creating the main environment.

        Args:
            env_type: Environment type (e.g., "atari")
            env_name: Environment name (e.g., "BoxingNoFrameskip-v4")
            **kwargs: Additional environment-specific arguments

        Returns:
            Response dictionary with action space information
        """
        print_log("INFO", f"Getting action space for: {Colors.BOLD}{env_type}/{env_name}{Colors.ENDC}")

        try:
            # Create a temporary environment manager to get action space
            temp_manager = create_env_manager(env_type)

            # Get action space information (this creates a dummy env internally)
            if env_type == "atari":
                image_size = kwargs.get("image_size", None)
                if image_size is not None:
                    image_size = tuple(image_size)
                action_space_info = temp_manager.get_action_space(env_name, kwargs.get("seed"), image_size)
            else:
                # For future environment types
                action_space_info = temp_manager.get_action_space(env_name, kwargs.get("seed"), **kwargs)

            print_log("SUCCESS", f"Action space retrieved: {action_space_info}")

            return {"status": "ok", "action_space": action_space_info}
        except Exception as e:
            print_log("ERROR", f"Failed to get action space: {str(e)}")
            return {"status": "error", "message": f"Failed to get action space: {str(e)}"}

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming message and return response.

        Unified interface: supports reset, step, and get_action_space.
        Environment is automatically created on first reset if not exists.

        Args:
            message: Incoming message dictionary with "operation" field ("reset", "step", or "get_action_space")

        Returns:
            Response dictionary
        """
        operation = message.get("operation", "")

        if operation == "get_action_space":
            # Get action space information
            env_type = message.get("env_type")
            env_name = message.get("env_name")

            if not env_type:
                return {"status": "error", "message": "env_type required for get_action_space"}
            if not env_name:
                return {"status": "error", "message": "env_name required for get_action_space"}

            # Extract environment-specific kwargs
            env_kwargs = {k: v for k, v in message.items() if k not in ["operation", "env_type", "env_name", "seed"]}
            env_kwargs["seed"] = message.get("seed", None)

            return self.handle_get_action_space(env_type, env_name, **env_kwargs)

        elif operation == "reset":
            # Get or create environment
            env_type = message.get("env_type")
            env_name = message.get("env_name")

            if not env_type:
                return {"status": "error", "message": "env_type required for reset"}
            if not env_name:
                return {"status": "error", "message": "env_name required for reset"}

            # Create environment if it doesn't exist
            if self.env_manager is None:
                try:
                    print_log("INFO", "Creating environment instance...")
                    seed = message.get("seed", None)
                    # Extract environment-specific kwargs (excluding common ones)
                    env_kwargs = {
                        k: v
                        for k, v in message.items()
                        if k not in ["operation", "env_type", "env_name", "seed", "kwargs"]
                    }

                    # Handle image_size for atari
                    if "image_size" in env_kwargs and env_kwargs["image_size"] is not None:
                        env_kwargs["image_size"] = tuple(env_kwargs["image_size"])

                    self.create_environment(env_type, env_name, seed, **env_kwargs)
                except Exception as e:
                    print_log("ERROR", f"Failed to create environment: {str(e)}")
                    return {"status": "error", "message": f"Failed to create environment: {str(e)}"}

            # Reset environment
            reset_kwargs = message.get("kwargs", {})
            return self.handle_reset(**reset_kwargs)

        elif operation == "step":
            action = message.get("action")

            if action is None:
                return {"status": "error", "message": "action required for step"}

            if self.env_manager is None:
                return {"status": "error", "message": "Environment not created. Call reset first."}

            return self.handle_step(action)

        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}. Supported operations: get_action_space, reset, step",
            }

    def handle_client(self, client_socket: socket.socket):
        """Handle client connection.

        Args:
            client_socket: Client socket
        """
        print_log("SUCCESS", "Client connected via Unix socket")
        try:
            while True:
                # Receive message length (4 bytes)
                length_bytes = client_socket.recv(4)
                if not length_bytes or len(length_bytes) < 4:
                    break

                message_length = int.from_bytes(length_bytes, byteorder="big")

                # Receive message data
                data = b""
                while len(data) < message_length:
                    chunk = client_socket.recv(message_length - len(data))
                    if not chunk:
                        break
                    data += chunk

                if len(data) < message_length:
                    break

                try:
                    message = json.loads(data.decode("utf-8"))
                    response = self.handle_message(message)
                    response_json = json.dumps(response).encode("utf-8")

                    # Send response length (4 bytes) + response data
                    response_length = len(response_json)
                    client_socket.send(response_length.to_bytes(4, byteorder="big"))
                    client_socket.send(response_json)
                except json.JSONDecodeError as e:
                    print_log("ERROR", f"Invalid JSON received: {str(e)}")
                    error_response = {"status": "error", "message": f"Invalid JSON: {str(e)}"}
                    error_json = json.dumps(error_response).encode("utf-8")
                    error_length = len(error_json)
                    client_socket.send(error_length.to_bytes(4, byteorder="big"))
                    client_socket.send(error_json)
        except Exception as e:
            print_log("ERROR", f"Error handling client: {e}")
        finally:
            client_socket.close()
            print_log("WARNING", "Client disconnected")

    def start(self):
        """Start the Unix socket server."""
        print_log("INFO", "Starting Unix socket server...")

        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        self.running = True

        # Set socket permissions to allow access from host
        os.chmod(self.socket_path, 0o666)
        print_log("INFO", f"Socket permissions set to 0o666")

        print_log("LINK", f"Server listening on: {Colors.ORANGE}{self.socket_path}{Colors.ENDC}")
        print_log("INFO", "Waiting for client connections...")

        while self.running:
            try:
                client_socket, _ = self.server_socket.accept()
                # Handle each client in a separate thread
                thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                thread.daemon = True
                thread.start()
            except Exception as e:
                if self.running:
                    print_log("ERROR", f"Error accepting connection: {e}")

    def stop(self):
        """Stop the Unix socket server."""
        print_log("INFO", "Stopping server...")

        self.running = False
        if self.server_socket:
            self.server_socket.close()
            print_log("INFO", "Server socket closed")

        # Remove socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
            print_log("INFO", f"Socket file removed: {self.socket_path}")

        # Close environment
        if self.env_manager is not None:
            print_log("INFO", "Closing environment...")
            try:
                self.env_manager.close()
                print_log("INFO", f"Closed environment: {self.env_type}/{self.env_name}")
            except Exception as e:
                print_log("WARNING", f"Failed to close environment: {e}")
            self.env_manager = None

        print_log("SUCCESS", "Server stopped successfully")


def main():
    """Main entry point for running server directly via SOCKET_PATH env."""
    # Get socket_path from environment variable (required)
    socket_path = os.environ.get("SOCKET_PATH")

    if not socket_path:
        print_log("ERROR", "SOCKET_PATH environment variable is not set")
        print_log("INFO", "Usage: SOCKET_PATH=/path/to/socket python server.py")
        sys.exit(1)

    # Initialize logger with socket path
    set_log_socket_path(socket_path)

    print_log("INFO", f"Starting RL Environment Server")
    print_log("INFO", f"Socket path from env: {Colors.BOLD}{socket_path}{Colors.ENDC}")

    server = RLEnvServer(socket_path=socket_path)

    try:
        server.start()
    except KeyboardInterrupt:
        print_log("WARNING", "\nReceived keyboard interrupt")
        server.stop()
    except Exception as e:
        print_log("ERROR", f"Server error: {e}")
        server.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
