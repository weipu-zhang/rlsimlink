"""rlsimlink - lightweight Unix-socket bridge for RL environments."""

from .src.client import RLEnv
from .src.server import RLEnvServer
from .src.common import ActionSpace
from .src.socket_paths import extract_socket_id, format_socket_path, generate_socket_id, resolve_socket_path

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "ActionSpace",
    "RLEnv",
    "RLEnvServer",
    "extract_socket_id",
    "format_socket_path",
    "generate_socket_id",
    "resolve_socket_path",
]
