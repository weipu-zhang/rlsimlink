import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

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

    def set_log_file(self, socket_path: str, log_type: str = "server"):
        """Set log file based on socket path.

        Args:
            socket_path: Socket path to determine log file location
            log_type: Type of log file ("server" or "client")
        """
        # Extract socket id from socket path
        socket_id = extract_socket_id(socket_path)

        # Sanitize socket id to create a valid filename
        # Keep only alphanumeric, dash, and underscore characters
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", socket_id)
        if not sanitized:
            sanitized = "manual"

        # Create logs directory relative to this file's location
        # From utils.py in rlsimlink/ -> logs/
        utils_dir = Path(__file__).parent  # rlsimlink/
        log_dir = utils_dir / "logs" / sanitized
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


# Global logger instance
_logger = Logger.get_instance()


def set_log_socket_path(socket_path: str, log_type: str = "server"):
    """Set the socket path for logging.

    Args:
        socket_path: Socket path to determine log file location
        log_type: Type of log file ("server" or "client")
    """
    global _logger
    _logger.set_log_file(socket_path, log_type)


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
