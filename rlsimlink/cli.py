#!/usr/bin/env python3
"""Command-line utilities for rlsimlink."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .docker.docker import add_docker_subparser as _add_docker_subparser
from .docker.docker import main as _docker_main
from .src.envs import create_env_manager
from .src.server import RLEnvServer
from .src.socket_paths import generate_socket_id, resolve_socket_path
from .utils import Colors, print_log, save_snapshot, set_log_level, set_log_socket_id, set_log_socket_path


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

    test_parser = subparsers.add_parser("test", help="Smoke test an environment inside this interpreter.")
    test_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Minimum log level for this test run (ERROR, WARN, LINK, SUCCESS, INFO). Default: INFO.",
    )

    test_subparsers = test_parser.add_subparsers(dest="env_type", required=True)

    atari_test = test_subparsers.add_parser("atari", help="Test an Atari environment.")
    atari_test.add_argument(
        "--env-name",
        default="BoxingNoFrameskip-v4",
        help="Gymnasium environment name. Defaults to BoxingNoFrameskip-v4.",
    )

    dmlab_test = test_subparsers.add_parser("dmlab", help="Test a DMLab environment.")
    dmlab_test.add_argument(
        "--env-name",
        default="rooms_keys_doors_puzzle",
        help='DMLab level name. Defaults to "rooms_keys_doors_puzzle".',
    )

    vizdoom_test = test_subparsers.add_parser("vizdoom", help="Test a VizDoom environment.")
    vizdoom_test.add_argument(
        "--env-name",
        default="VizdoomBasic-v0",
        help="VizDoom environment name. Defaults to VizdoomBasic-v0.",
    )

    minerl_test = test_subparsers.add_parser("minerl", help="Test a MineRL environment.")
    minerl_test.add_argument(
        "--env-name",
        default="MineRLBasaltFindCave-v0",
        help="MineRL environment name. Defaults to MineRLBasaltFindCave-v0.",
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
    elif args.command == "test":
        set_log_level(args.log_level)
        set_log_socket_id("test", log_type="test")
        print_log("LINK", "please ensure running inside of the simulation environments/docker/etc.")

        env_manager = create_env_manager(args.env_type)

        try:
            if args.env_type == "atari":
                env_manager.create(args.env_name)
            elif args.env_type == "dmlab":
                env_manager.create(env_name=args.env_name)
            elif args.env_type == "vizdoom":
                env_manager.create(args.env_name)
            elif args.env_type == "minerl":
                env_manager.create(args.env_name)
            else:
                raise ValueError(f"Unsupported env_type: {args.env_type}")

            test_result = env_manager.test_env()
            save_snapshot(f"{args.env_type}.png", test_result.get("reset_pixel_observation"))
            print_log("SUCCESS", "Environment smoke test complete.")
        except Exception as exc:
            print_log("ERROR", f"Smoke test failed: {exc}")
            sys.exit(1)
        finally:
            env_manager.close()
    elif args.command == "docker":
        _docker_main(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
