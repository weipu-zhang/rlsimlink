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
    atari_test.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    atari_test.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Optional resize target as two integers, e.g. --image-size 84 84.",
    )

    dmlab_test = test_subparsers.add_parser("dmlab", help="Test a DMLab environment.")
    dmlab_test.add_argument(
        "--env-name",
        default="rooms_keys_doors_puzzle",
        help='DMLab level name. Defaults to "rooms_keys_doors_puzzle".',
    )
    dmlab_test.add_argument("--seed", type=int, default=None, help="Optional seed.")
    dmlab_test.add_argument("--repeat", type=int, default=4, help="Number of simulator steps per action.")
    dmlab_test.add_argument(
        "--size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Frame size as HEIGHT WIDTH. Defaults to 64 64.",
    )
    dmlab_test.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Environment mode controlling level split.",
    )
    dmlab_test.add_argument(
        "--actions",
        choices=["impala", "popart"],
        default="popart",
        help="Action set to use.",
    )
    dmlab_test.add_argument("--episodic", dest="episodic", action="store_true", help="Force episodic termination.")
    dmlab_test.add_argument(
        "--no-episodic",
        dest="episodic",
        action="store_false",
        help="Disable episodic termination.",
    )
    dmlab_test.add_argument("--text", dest="text", action="store_true", help="Enable instruction text observations.")
    dmlab_test.add_argument(
        "--no-text", dest="text", action="store_false", help="Disable instruction text observations."
    )
    dmlab_test.set_defaults(episodic=True, text=None)

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
                image_size = tuple(args.image_size) if args.image_size else None
                env_manager.create(args.env_name, seed=args.seed, image_size=image_size)
            elif args.env_type == "dmlab":
                size = tuple(args.size) if args.size else (64, 64)
                env_manager.create(
                    env_name=args.env_name,
                    seed=args.seed,
                    repeat=args.repeat,
                    size=size,
                    mode=args.mode,
                    actions=args.actions,
                    episodic=args.episodic,
                    text=args.text,
                )
            else:
                raise ValueError(f"Unsupported env_type: {args.env_type}")

            test_result = env_manager.test_env()
            save_snapshot(f"{args.env_type}.png", test_result.get("reset_raw_observation"))
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
