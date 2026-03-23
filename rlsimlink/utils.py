import re
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV is required at runtime but optional for import
    cv2 = None

# Import extract_socket_id to get socket id from path
from .src.socket_paths import extract_socket_id


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    ORANGE = "\033[38;5;208m"
    PURPLE = "\033[38;5;141m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class Logger:
    """Simple logger that writes to both console and file."""

    _instance: Optional["Logger"] = None

    # Log level hierarchy (lower number = higher priority)
    LOG_LEVELS = {
        "ERROR": 0,
        "WARN": 1,
        "LINK": 2,
        "SUCCESS": 3,
        "INFO": 4,
    }

    LEVEL_ALIASES = {
        "WARNING": "WARN",
        "STEP": "INFO",
    }

    def __init__(self, socket_path: Optional[str] = None, log_type: str = "server", log_level: str = "LINK"):
        """Initialize logger with optional socket path for log file location.

        Args:
            socket_path: Socket path to determine log file location
            log_type: Type of log file ("server" or "client")
            log_level: Minimum log level to display/write (ERROR, WARN, LINK, SUCCESS, INFO)
                      Default is LINK (shows ERROR, WARN, LINK)
        """
        self.log_file = None
        self.log_type = log_type
        self.set_log_level(log_level)
        if socket_path:
            self.set_log_file(socket_path, log_type)

    @classmethod
    def get_instance(cls) -> "Logger":
        """Get or create singleton logger instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_log_level(self, log_level: str):
        """Set the minimum log level to display/write.

        Args:
            log_level: Minimum log level (ERROR, WARNING, SUCCESS, INFO)
        """
        log_level_upper = self.LEVEL_ALIASES.get(log_level.upper(), log_level.upper())
        if log_level_upper not in self.LOG_LEVELS:
            print(f"{Colors.WARNING}[WARN]{Colors.ENDC} Invalid log level: {log_level}, using LINK")
            log_level_upper = "LINK"
        self.log_level = log_level_upper
        self.log_level_value = self.LOG_LEVELS[log_level_upper]

    def set_log_file(self, socket_id: str, log_type: str = "server"):
        """Set log file based on socket identifier.

        Args:
            socket_id: Identifier used to group logs
            log_type: Type of log file ("server", "client", etc.)
        """
        resolved_id = socket_id or "manual"

        utils_dir = Path(__file__).parent  # rlsimlink/
        log_dir = utils_dir / "logs" / resolved_id
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file based on type (server.log or client.log)
        self.log_type = log_type
        self.log_file = log_dir / f"{log_type}.log"

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI color codes from text.

        Args:
            text: Text with potential ANSI codes

        Returns:
            Text without ANSI codes
        """
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        return ansi_escape.sub("", text)

    def log(self, level: str, message: str):
        """Log message to both console and file.

        Args:
            level: Log level ("ERROR", "WARN", "LINK", "SUCCESS", "INFO")
            message: Message to log
        """
        level_upper = self.LEVEL_ALIASES.get(level.upper(), level.upper())

        # Check if this log level should be displayed
        if level_upper not in self.LOG_LEVELS:
            level_upper = "INFO"

        current_level_value = self.LOG_LEVELS.get(level_upper, 3)
        if current_level_value > self.log_level_value:
            # Skip this log message (lower priority than minimum level)
            return

        # Determine color for console output
        if level_upper == "ERROR":
            color = Colors.FAIL
        elif level_upper == "WARN":
            color = Colors.WARNING
        elif level_upper == "LINK":
            color = Colors.ORANGE
        elif level_upper == "SUCCESS":
            color = Colors.OKGREEN
        else:
            color = Colors.OKCYAN

        # Print to console with color
        console_message = f"{color}[{level_upper}]{Colors.ENDC} {message}"
        print(console_message)

        # Write to log file without color codes
        if self.log_file:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Strip ANSI codes from message
                clean_message = self._strip_ansi(message)
                log_entry = f"[{timestamp}] [{level_upper}] {clean_message}\n"

                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_entry)
            except Exception as e:
                # If logging to file fails, just print to console
                print(f"{Colors.WARNING}[WARN]{Colors.ENDC} Failed to write to log file: {e}")

    def save_snapshot(self, filename: str, observation: Any):
        """Save an observation snapshot next to the log file."""
        if observation is None:
            self.log("WARN", "No observation provided; skipping snapshot save.")
            return
        if self.log_file is None:
            self.log("WARN", "Log file not configured; cannot determine snapshot directory.")
            return
        if cv2 is None:
            self.log("WARN", "OpenCV not available; cannot save snapshot.")
            return

        image = np.asarray(observation)
        if image.size == 0:
            self.log("WARN", "Observation empty; skipping snapshot save.")
            return

        if image.ndim == 3 and image.shape[-1] == 3:
            image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_to_save = image

        writer = getattr(cv2, "imsave", None)
        if writer is None:
            writer = cv2.imwrite

        output_path = self.log_file.parent / filename
        try:
            success = writer(str(output_path), image_to_save)
            if success is False:
                raise RuntimeError("cv2 writer reported failure")
            self.log("SUCCESS", f"Saved snapshot to {output_path}")
        except Exception as exc:  # pragma: no cover - filesystem errors are non-deterministic
            self.log("WARN", f"Failed to write snapshot {output_path}: {exc}")


# Global logger instance
_logger = Logger.get_instance()


def set_log_socket_id(socket_id: str, log_type: str = "server"):
    """Configure logging based on a socket identifier."""
    global _logger
    _logger.set_log_file(socket_id, log_type)


def set_log_socket_path(socket_path: str, log_type: str = "server"):
    """Backwards-compatible helper that accepts a socket path."""
    socket_id = extract_socket_id(socket_path)
    set_log_socket_id(socket_id, log_type)


def set_log_level(log_level: str):
    """Set the minimum log level to display/write.

    Args:
        log_level: Minimum log level (ERROR, WARN, LINK, SUCCESS, INFO)
                  ERROR: Only errors
                  WARN: Errors and warnings
                  LINK: Errors, warnings, and link messages (default)
                  SUCCESS: Errors, warnings, link, and success messages
                  INFO: All messages
    """
    global _logger
    _logger.set_log_level(log_level)


def print_log(level: str, message: str):
    """Print log message with specified level to both console and file.

    Args:
        level: Log level ("ERROR", "WARN", "LINK", "SUCCESS", "INFO")
        message: Message to print
    """
    global _logger
    _logger.log(level, message)


def save_snapshot(filename: str, observation: Any):
    """Persist an observation using the configured logger directory."""
    global _logger
    _logger.save_snapshot(filename, observation)


def _print_info(message: str):
    """Print info message in cyan."""
    print_log("INFO", message)


def _print_success(message: str):
    """Print success message in green."""
    print_log("SUCCESS", message)


def _print_warning(message: str):
    """Print warning message in yellow."""
    print_log("WARN", message)


def _print_error(message: str):
    """Print error message in red."""
    print_log("ERROR", message)
