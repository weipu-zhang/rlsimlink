"""Utilities for building and parsing Unix socket paths used by rlsimlink."""

from __future__ import annotations

import re
import secrets
from pathlib import Path

SHM_NAMESPACE = "rlsimlink"
SOCKET_FILENAME = "socket"
OBS_FILENAME = "obs"


def _namespace_root(create: bool = False) -> Path:
    """Return the base shared-memory directory (/dev/shm/rlsimlink)."""
    base = Path("/dev/shm") / SHM_NAMESPACE
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def sanitize_socket_id(socket_id: str) -> str:
    """Sanitize the socket identifier, keeping only [-_a-zA-Z0-9]."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", (socket_id or "").strip())
    if not sanitized:
        raise ValueError("socket_id must contain at least one alphanumeric character")
    return sanitized


def generate_socket_id(num_bytes: int = 4) -> str:
    """Generate a random hex socket identifier (default: 8 chars)."""
    if num_bytes <= 0:
        raise ValueError("num_bytes must be positive")
    return secrets.token_hex(num_bytes)


def socket_dir(socket_id: str, *, create: bool = False) -> Path:
    """Return the directory inside /dev/shm assigned to a socket id."""
    sanitized = sanitize_socket_id(socket_id)
    base = _namespace_root(create=create)
    path = base / sanitized
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_socket_path(socket_id: str, *, create_parent: bool = False) -> Path:
    """Return the Unix socket path for a socket id."""
    return socket_dir(socket_id, create=create_parent) / SOCKET_FILENAME


def resolve_observation_path(socket_id: str, *, create_parent: bool = False) -> Path:
    """Return the observation file path for a socket id."""
    return socket_dir(socket_id, create=create_parent) / OBS_FILENAME


def extract_socket_id(socket_path: str) -> str:
    """Extract the socket id from a full socket path, fallback to 'manual'."""
    path = Path(socket_path)
    parts = path.parts
    if SHM_NAMESPACE in parts:
        idx = parts.index(SHM_NAMESPACE)
        if idx + 1 < len(parts):
            return parts[idx + 1]

    match = re.search(r"socket_([a-zA-Z0-9]+)", path.name)
    if match:
        return match.group(1)

    return "manual"


def format_socket_path(socket_id: str) -> str:
    """Convenience helper that returns the socket path as string without creating directories."""
    return str(resolve_socket_path(socket_id, create_parent=False))


__all__ = [
    "OBS_FILENAME",
    "SHM_NAMESPACE",
    "SOCKET_FILENAME",
    "extract_socket_id",
    "format_socket_path",
    "generate_socket_id",
    "resolve_observation_path",
    "resolve_socket_path",
    "sanitize_socket_id",
    "socket_dir",
]
