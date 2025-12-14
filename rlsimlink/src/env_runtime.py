"""Environment-specific runtime launcher for starting RL servers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from rlsimlink.utils import print_log, Colors


class EnvServerLauncher:
    """Launches environment servers based on env_type."""

    def __init__(self, env_type: str):
        self.env_type = env_type
        self.process: Optional[subprocess.Popen] = None
        self.socket_path: Optional[str] = None

    def _build_command(self, socket_path: str) -> list[str]:
        """Map env_type to the command that starts its server."""
        if self.env_type == "atari":
            # Start rlsimlink server inside the `atari` conda environment
            return ["conda", "run", "-n", "atari", "rlsimlink", "serve", "--socket-path", socket_path]

        raise ValueError(f"Auto-launch not implemented for env_type={self.env_type}")

    def start(self, socket_path: str):
        """Start the server process if not already running."""
        if self.process and self.process.poll() is None:
            return

        command = self._build_command(socket_path)
        self.socket_path = socket_path

        print_log("LINK", f"Launching {Colors.PURPLE}{self.env_type}{Colors.ENDC} server via: {' '.join(command)}")

        # Ensure socket directory exists
        Path(socket_path).parent.mkdir(parents=True, exist_ok=True)

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

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
        self.socket_path = None
