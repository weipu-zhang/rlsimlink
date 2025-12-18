#!/usr/bin/env python3
"""Unix socket server that exposes RL environments to external processes."""

import socket
from json import JSONDecodeError
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Import color utilities from the shared logger utilities module
from rlsimlink.utils import Colors, print_log, set_log_socket_path
from .common import SocketManager
from .envs import create_env_manager
from .socket_paths import extract_socket_id


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
        self.env_manager = None  # Environment manager instance
        self.env_type: Optional[str] = None
        self.env_name: Optional[str] = None
        self.running = False
        self._socket_manager = SocketManager(socket_path, log_fn=print_log, role="server")

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

    def handle_reset(self, **kwargs) -> Dict[str, Any]:
        """Reset the environment.

        Args:
            **kwargs: Additional reset arguments

        Returns:
            Response dictionary with observation and info
        """
        if self.env_manager is None:
            print_log("ERROR", "Environment not initialized")
            return {"status": "error", "message": "Environment not initialized"}

        print_log("INFO", f"Resetting environment: {Colors.BOLD}{self.env_type}/{self.env_name}{Colors.ENDC}")
        observation, info = self.env_manager.reset(**kwargs)

        print_log(
            "SUCCESS",
            f"Environment reset successfully (obs shape: {observation.shape if hasattr(observation, 'shape') else 'N/A'})",
        )

        payload = {
            "status": "ok",
            "info": self._socket_manager.serialize_info(info),
        }
        self._socket_manager.attach_observation(payload, observation)
        return payload

    def handle_step(self, action: Any) -> Dict[str, Any]:
        """Step in the environment.

        Args:
            action: Action to take (serialized from client)

        Returns:
            Response dictionary with step results
        """
        if self.env_manager is None:
            print_log("ERROR", "Environment not initialized")
            return {"status": "error", "message": "Environment not initialized"}

        # Deserialize action from JSON format
        deserialized_action = self._socket_manager.deserialize_action(action)

        observation, reward, terminated, truncated, info = self.env_manager.step(deserialized_action)

        payload = {
            "status": "ok",
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": self._socket_manager.serialize_info(info),
        }
        self._socket_manager.attach_observation(payload, observation)
        return payload

    def handle_get_action_space(self, env_type: str, env_name: str, seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Get action space information from the initialized environment."""
        print_log("INFO", f"Getting action space for: {Colors.BOLD}{env_type}/{env_name}{Colors.ENDC}")

        if self.env_manager is None:
            message = "Environment not initialized. Call initialize before requesting action space."
            print_log("ERROR", message)
            return {"status": "error", "message": message}

        mismatch_response = self._ensure_environment_matches(env_type, env_name)
        if mismatch_response and mismatch_response.get("status") == "error":
            return mismatch_response

        try:
            action_space_info = self.env_manager.get_action_space(env_name, seed, **kwargs)
            print_log("SUCCESS", f"Action space retrieved: {action_space_info}")
            return {"status": "ok", "action_space": action_space_info}
        except Exception as e:
            print_log("ERROR", f"Failed to get action space: {str(e)}")
            return {"status": "error", "message": f"Failed to get action space: {str(e)}"}

    def _extract_env_config(self, message: Dict[str, Any]) -> Tuple[str, str, Optional[int], Dict[str, Any]]:
        """Parse environment configuration from a message payload."""
        env_type = message.get("env_type")
        env_name = message.get("env_name")

        if not env_type:
            raise ValueError("env_type required")
        if not env_name:
            raise ValueError("env_name required")

        seed = message.get("seed", None)
        env_kwargs = {
            k: v
            for k, v in message.items()
            if k not in ["operation", "env_type", "env_name", "seed", "kwargs"]
        }

        # Normalize Atari image_size tuple if present
        if "image_size" in env_kwargs and env_kwargs["image_size"] is not None:
            env_kwargs["image_size"] = tuple(env_kwargs["image_size"])

        return env_type, env_name, seed, env_kwargs

    def _ensure_environment_matches(self, env_type: str, env_name: str) -> Optional[Dict[str, Any]]:
        """Validate that the requested env matches the already-created one."""
        if self.env_manager is None:
            return None

        if env_type != self.env_type or env_name != self.env_name:
            message = (
                "Environment already initialized as "
                f"{self.env_type}/{self.env_name}, cannot switch to {env_type}/{env_name}"
            )
            print_log("ERROR", message)
            return {"status": "error", "message": message}

        return {"status": "ok", "message": "Environment already initialized"}

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming message and return response.

        Unified interface: supports initialize, get_action_space, reset, and step.
        Environments must be created via an explicit initialize request before use.

        Args:
            message: Incoming message dictionary with "operation" field ("initialize", "get_action_space", "reset", "step")

        Returns:
            Response dictionary
        """
        operation = message.get("operation", "")

        if operation == "initialize":
            try:
                env_type, env_name, seed, env_kwargs = self._extract_env_config(message)
            except ValueError as err:
                return {"status": "error", "message": str(err)}

            if self.env_manager is not None:
                return self._ensure_environment_matches(env_type, env_name) or {
                    "status": "ok",
                    "message": "Environment already initialized",
                }

            try:
                print_log("INFO", f"Initializing environment on client request...")
                self.create_environment(env_type, env_name, seed, **env_kwargs)
                return {"status": "ok", "message": "Environment ready"}
            except Exception as exc:
                return {"status": "error", "message": f"Failed to initialize environment: {exc}"}

        if operation == "get_action_space":
            try:
                env_type, env_name, seed, env_kwargs = self._extract_env_config(message)
            except ValueError as err:
                return {"status": "error", "message": str(err)}

            return self.handle_get_action_space(env_type, env_name, seed, **env_kwargs)

        elif operation == "reset":
            try:
                env_type, env_name, _, _ = self._extract_env_config(message)
            except ValueError as err:
                return {"status": "error", "message": str(err)}

            if self.env_manager is None:
                message = "Environment not initialized. Call initialize before reset."
                print_log("ERROR", message)
                return {"status": "error", "message": message}

            mismatch_response = self._ensure_environment_matches(env_type, env_name)
            if mismatch_response and mismatch_response.get("status") == "error":
                return mismatch_response

            # Reset environment
            reset_kwargs = message.get("kwargs", {})
            return self.handle_reset(**reset_kwargs)

        elif operation == "step":
            action = message.get("action")

            if action is None:
                return {"status": "error", "message": "action required for step"}

            if self.env_manager is None:
                return {"status": "error", "message": "Environment not initialized. Call initialize first."}

            return self.handle_step(action)

        else:
            return {
                "status": "error",
                "message": (
                    "Unknown operation: "
                    f"{operation}. Supported operations: initialize, get_action_space, reset, step"
                ),
            }

    def handle_client(self, client_socket: socket.socket):
        """Handle client connection.

        Args:
            client_socket: Client socket
        """
        print_log("SUCCESS", "Client connected via Unix socket")
        try:
            while True:
                try:
                    message = self._socket_manager.receive_message(client_socket)
                except JSONDecodeError as e:
                    print_log("ERROR", f"Invalid JSON received: {str(e)}")
                    error_response = {"status": "error", "message": f"Invalid JSON: {str(e)}"}
                    self._socket_manager.send_message(client_socket, error_response)
                    continue

                if message is None:
                    break

                try:
                    response = self.handle_message(message)
                    self._socket_manager.send_message(client_socket, response)
                except Exception as handler_error:
                    print_log("ERROR", f"Failed to handle message: {handler_error}")
                    error_response = {"status": "error", "message": str(handler_error)}
                    self._socket_manager.send_message(client_socket, error_response)
        except Exception as e:
            print_log("ERROR", f"Error handling client: {e}")
        finally:
            client_socket.close()
            print_log("WARNING", "Client disconnected")

    def start(self):
        """Start the Unix socket server."""
        print_log("INFO", "Starting Unix socket server...")
        self._socket_manager.start_server()
        self.running = True
        print_log("INFO", "Waiting for client connections...")

        while self.running:
            try:
                client_socket = self._socket_manager.accept_client()
                self.handle_client(client_socket)
            except Exception as e:
                if self.running:
                    print_log("ERROR", f"Error accepting connection: {e}")

    def stop(self):
        """Stop the Unix socket server."""
        print_log("INFO", "Stopping server...")

        self.running = False
        self._socket_manager.close()

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
