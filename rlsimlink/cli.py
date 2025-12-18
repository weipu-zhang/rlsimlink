#!/usr/bin/env python3
"""Command-line utilities for rlsimlink."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .src.server import RLEnvServer
from .src.socket_paths import generate_socket_id, resolve_socket_path
from .docker.docker import add_docker_subparser as _add_docker_subparser
from .docker.docker import main as _docker_main
from .utils import Colors, print_log, set_log_level, set_log_socket_path


def _run_server(socket_path: str):
    """Instantiate and run the RL environment server."""
    set_log_socket_path(socket_path)

    print_log("INFO", f"Starting RL Environment Server")
    print_log("INFO", f"Socket path: {Colors.BOLD}{socket_path}{Colors.ENDC}")

    server = RLEnvServer(socket_path=socket_path)

    try:
        server.start()
    except KeyboardInterrupt:
        print_log("WARNING", "\nReceived keyboard interrupt")
        server.stop()
    except Exception as exc:
        print_log("ERROR", f"Server error: {exc}")
        server.stop()
        sys.exit(1)


def main():
    """Main CLI entry point for rlsimlink."""
    parser = argparse.ArgumentParser(prog="rlsimlink", description="Lightweight RL environment bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Start the RL environment server in the current Python env")
    serve_parser.add_argument(
        "--socket-id",
        type=str,
        default=None,
        help="Socket identifier. The Unix socket will be created under /dev/shm/rlsimlink/<id>/socket.",
    )
    serve_parser.add_argument(
        "--log-level",
        type=str,
        default="LINK",
        help="Minimum log level (ERROR, WARN, LINK, SUCCESS, INFO). Default: LINK.",
    )

    _add_docker_subparser(subparsers)

    args = parser.parse_args()

    if args.command == "serve":
        set_log_level(args.log_level)

        socket_id = args.socket_id or generate_socket_id()
        socket_path = str(resolve_socket_path(socket_id, create_parent=True))

        Path(socket_path).parent.mkdir(parents=True, exist_ok=True)

        print_log("INFO", "Share this id with the training process to connect via rlsimlink.RLEnv.")

        _run_server(socket_path)
    elif args.command == "docker":
        _docker_main(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
