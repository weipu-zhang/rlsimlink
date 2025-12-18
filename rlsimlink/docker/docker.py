# Copyright (c) 2024
# A helper class for managing Reinforcement Learning simulation Docker containers.

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Union
import argparse
import grp
import getpass
from textwrap import dedent

from .utils import (
    get_hostname,
    is_user_in_docker_group,
    check_port_occupied,
    load_env_version,
)
from ..utils import print_log


class RLContainerInterface:
    """A helper class for managing RL simulation Docker containers."""

    def __init__(
        self,
        work_dir: Path,
        workspace_target: str = "/root/rlsimlink",
        tcp_port: int = 8888,
        use_gpu: bool = True,
        hash_code: str = "manual",
    ):
        self.work_dir = work_dir.resolve().expanduser()
        self.workspace_target = workspace_target
        self.tcp_port = tcp_port
        self.use_gpu = use_gpu
        self.hash_code = hash_code

        # set the context directory
        repo_root = Path(__file__).resolve().parent
        self.context_dir = repo_root.joinpath("resources")
        self.composefile_path = repo_root.joinpath("docker-compose.yml")
        self.tag = load_env_version(default="dmlab")
        self.repo_name = "wp-bit/rlsimlink"
        self.image_id = None

        if self.does_image_exist():
            self.image_id = self.get_image_id()

        # Generate container name: hash_code-rlsimlink-env
        self.container_name = f"{self.hash_code}-rlsimlink-{self.tag}"
        self.host_name = get_hostname()

        assert is_user_in_docker_group(), dedent(
            f"""
            The current user is not in the 'docker' group. Please add the user to the 'docker' group and restart the terminal:
            `sudo usermod -a -G docker {getpass.getuser()}`
        """
        )

        # Setup volume mounts
        self.mounted_volumes: List[Dict[str, Union[str, bool]]] = []
        # mount core implementation to docker
        repo_dir = Path(__file__).resolve().parent.parent.parent
        self.mount_volume(source=repo_dir, target=Path(workspace_target))

        # Setup /dev/shm/rlsimlink directory (shared host<->container socket location)
        host_shm_dir = Path("/dev/shm/rlsimlink")
        container_shm_dir = Path("/dev/shm/rlsimlink")
        if not host_shm_dir.exists():
            os.makedirs(host_shm_dir, exist_ok=True)
        self.mount_volume(source=host_shm_dir, target=container_shm_dir)

        # Keep the environment variables from the current environment
        self.environ = os.environ.copy()

    @property
    def image_name(self) -> str:
        return f"{self.repo_name}:{self.tag}"

    def get_image_id(self) -> str:
        """Get the image ID of the Docker image.

        Returns:
            The image ID of the Docker image.
        """
        result = subprocess.run(
            ["docker", "images", "--format", "{{.ID}},{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        for line in result.splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                img_id, name = parts
                if name == self.image_name:
                    return img_id
        raise RuntimeError(f"Image not found. Please pull or build the image first.")

    def mount_volume(
        self,
        source: Path,
        target: Path,
        type: str = "bind",
        read_only: bool = False,
    ):
        """Mount a volume to the container.

        Args:
            source: The source path on the host machine.
            target: The target path in the container.
            type: The type of mount. Defaults to "bind".
            read_only: Whether the mount is read-only. Defaults to False.
        """
        if not os.path.exists(source) and type == "bind":
            os.makedirs(source, exist_ok=True)
        self.mounted_volumes.append(
            {
                "source": str(source),
                "target": str(target),
                "type": type,
                "read_only": read_only,
            }
        )

    def mount_args(self):
        """Generate Docker mount arguments from mounted volumes."""
        mount_args = []
        for mount in self.mounted_volumes:
            mount_args.append("--mount")
            mount_args.append(f"type={mount['type']},source={mount['source']},target={mount['target']}")
            if mount["read_only"]:
                mount_args[-1] += ",readonly"
        return mount_args

    def is_container_running(self) -> bool:
        """Check if the container is running.

        Returns:
            True if the container is running, otherwise False.
        """
        status = subprocess.run(
            ["docker", "container", "inspect", "-f", "{{.State.Status}}", self.container_name],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return status == "running"

    def does_image_exist(self) -> bool:
        """Check if the Docker image exists.

        Returns:
            True if the image exists, otherwise False.
        """
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        names = [line.strip() for line in result.splitlines()]
        return self.image_name in names

    def build(self):
        """Build the Docker image using docker-compose."""
        command = [
            "docker",
            "compose",
            "--file",
            str(self.composefile_path),
            "build",
        ]
        http_proxy = self.environ.get("http_proxy", "")
        https_proxy = self.environ.get("https_proxy", "")
        env = {**self.environ, "HOSTNAME": self.host_name}

        if len(http_proxy) > 0:
            print_log("INFO", f"Using HTTP proxy {http_proxy} for building the image.")
        if len(https_proxy) > 0:
            print_log("INFO", f"Using HTTPS proxy {https_proxy} for building the image.")
        if len(http_proxy) == 0 and len(https_proxy) == 0:
            print_log("WARNING", "No proxy environment variables found. Building without proxy.")

        subprocess.run(command, check=False, cwd=Path(__file__).resolve().parent, env=env)

    def pull(self):
        """Pull the Docker image from the registry."""
        if self.does_image_exist():
            print_log("INFO", f"The image '{self.image_name}' already exists. No need to pull it again.")
            return
        command = ["docker", "pull", self.image_name]
        subprocess.run(command, check=True, capture_output=False, text=True)

    def start(self):
        """Start the Docker container."""
        if not self.is_container_running():
            if not self.does_image_exist():
                raise RuntimeError(
                    f"The image '{self.image_name}' does not exist. "
                    f"Please pull or build it first by `rlsimlink docker pull` or `rlsimlink docker build`."
                )
            else:
                # Ensure image_id is set
                if self.image_id is None:
                    self.image_id = self.get_image_id()
                print_log("INFO", f"Starting container '{self.container_name}' with image '{self.image_name}'")

            # Check and find available TCP port
            port = self.tcp_port
            while check_port_occupied(port):
                print_log("WARNING", f"Port {port} is already occupied. Trying the next port...")
                port += 1
            if port != self.tcp_port:
                print_log("INFO", f"Using port {port} instead of {self.tcp_port} for TCP communication.")
            self.tcp_port = port

            # Build docker run command
            command = [
                "docker",
                "run",
                "--rm",
                "-dit",
                "--name",
                self.container_name,
                "--hostname",
                self.host_name,
                *self.mount_args(),
                f"--env=TCP_PORT={self.tcp_port}",
            ]

            # Add GPU support if requested
            if self.use_gpu:
                command.append("--gpus=all")
                print_log("INFO", "GPU support enabled.")

            # Add shared memory size (8GB)
            command.append("--shm-size=8g")
            print_log("INFO", "Shared memory size set to 8GB.")

            # Add network mode
            command.extend(
                [
                    "--network=host",
                    self.image_id,
                ]
            )

            # Run docker command and capture output to avoid printing container ID
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                print_log("ERROR", f"Failed to start container: {result.stderr.strip()}")
            else:
                print_log("SUCCESS", f"Container '{self.container_name}' started successfully.")
                print_log("INFO", f"TCP communication port: {self.tcp_port}")
        else:
            print_log("INFO", f"The container '{self.container_name}' is already running.")

    def enter(self):
        """Enter the running container by executing a bash shell.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self.is_container_running():
            print_log("INFO", f"Entering the existing '{self.container_name}' container in a bash session...")
            subprocess.run(
                [
                    "docker",
                    "exec",
                    "--interactive",
                    "--tty",
                    self.container_name,
                    "bash",
                ]
            )
        else:
            raise RuntimeError(f"The container '{self.container_name}' is not running.")

    def stop(self):
        """Stop the running container.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self.is_container_running():
            print_log("INFO", f"Stopping the docker container '{self.container_name}'...")
            subprocess.run(
                ["docker", "kill", self.container_name],
                check=False,
                env=self.environ,
            )
        else:
            raise RuntimeError(f"Can't stop container '{self.container_name}' as it is not running.")

    def attach(self, container_name: str = None):
        """Attach to a running container by executing a bash shell.

        Args:
            container_name: Optional container name. If not provided, will list
                all running rlsimdock containers and prompt for selection.

        Raises:
            RuntimeError: If the container is not running or not found.
        """
        if container_name is None:
            # Interactive mode: list containers and let user choose
            containers = list_running_rlsimdock_containers()
        if not containers:
            raise RuntimeError("No running rlsimlink containers found.")

            print_log("INFO", "The following rlsimlink containers are currently running:")
            for idx, (container_id, name, status) in enumerate(containers, start=1):
                print(f"  {idx}. {name} ({status})")

            while True:
                try:
                    choice = input(f"\nSelect container to attach [1-{len(containers)}]: ").strip()
                    if not choice:
                        print_log("INFO", "attach aborted.")
                        return
                    idx = int(choice) - 1
                    if 0 <= idx < len(containers):
                        container_name = containers[idx][1]
                        break
                    else:
                        print_log("ERROR", f"Invalid selection. Please enter a number between 1 and {len(containers)}.")
                except ValueError:
                    print_log("ERROR", "Invalid input. Please enter a number.")
        else:
            # Direct mode: use provided container name
            # Verify container is running
            status = subprocess.run(
                ["docker", "container", "inspect", "-f", "{{.State.Status}}", container_name],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            if status != "running":
                raise RuntimeError(f"Container '{container_name}' is not running (status: {status}).")

        print_log("INFO", f"Attaching to container '{container_name}' in a bash session...")
        subprocess.run(
            [
                "docker",
                "exec",
                "--interactive",
                "--tty",
                container_name,
                "bash",
            ]
        )

    def server_start(self, socket_path: str, env_type: str, install_package: bool = True):
        """Start the rlsimdock server in the container within a conda environment.

        Args:
            socket_path: Path to the Unix socket
            env_type: The conda environment name (e.g., "atari", "dmc", "metaworld")
            install_package: Whether to install the package in editable mode first

        Raises:
            RuntimeError: If the container is not running or server fails to start.
        """
        if not self.is_container_running():
            raise RuntimeError(f"Container '{self.container_name}' is not running. Start it first.")

        # Install package if requested
        if install_package:
            print_log("INFO", f"Installing rlsimlink package in conda environment '{env_type}'...")
            install_cmd = [
                "docker",
                "exec",
                self.container_name,
                "bash",
                "-c",
                f"cd {self.workspace_target} && conda run -n {env_type} pip install -e .",
            ]
            result = subprocess.run(install_cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to install package in conda environment '{env_type}': {result.stderr.strip()}"
                )
            print_log("SUCCESS", f"Package installed successfully in conda environment '{env_type}'")

        # Start the server using rlsimlink CLI command
        print_log("INFO", f"Starting server in conda environment '{env_type}' with socket: {socket_path}")
        server_cmd = [
            "docker",
            "exec",
            "-d",
            self.container_name,
            "bash",
            "-c",
            f"cd {self.workspace_target} && conda run -n {env_type} rlsimlink serve --socket-path {socket_path}",
        ]
        result = subprocess.run(server_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start server: {result.stderr.strip()}")

        print_log("SUCCESS", f"Server started successfully in conda environment '{env_type}'")


def list_running_rlsimdock_containers():
    """List all running Docker containers whose name contains 'rlsimlink'.

    Returns:
        List of tuples (container_id, container_name, status) for running rlsimlink containers.
    """
    ps_command = [
        "docker",
        "ps",
        "--filter",
        "name=rlsimlink",
        "--format",
        "{{.ID}} {{.Names}} {{.Status}}",
    ]
    result = subprocess.run(ps_command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list rlsimlink containers: {result.stderr.strip()}")

    containers_raw = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not containers_raw:
        return []

    containers = []
    for line in containers_raw:
        parts = line.split(maxsplit=2)
        container_id = parts[0]
        name = parts[1] if len(parts) > 1 else container_id
        status = parts[2] if len(parts) > 2 else "unknown"
        containers.append((container_id, name, status))

    return containers


def stop_all_rlsimdock_containers():
    """Stop every running Docker container whose name contains 'rlsimlink'."""
    containers = list_running_rlsimdock_containers()
    if not containers:
        print_log("INFO", "No running rlsimlink containers found.")
        return

    print_log("INFO", "The following rlsimlink containers are currently running:")
    for _, name, _ in containers:
        print(f"  - {name}")

    confirmation = input("Stop all listed containers? [y/N]: ").strip().lower()
    if confirmation != "y":
        print_log("INFO", "stop-all aborted.")
        return

    stopped = []
    failures = []
    for container_id, name, _ in containers:
        stop_result = subprocess.run(
            ["docker", "kill", container_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if stop_result.returncode == 0:
            stopped.append(name)
        else:
            failures.append((name, stop_result.stderr.strip()))

    if stopped:
        print_log("SUCCESS", f"Stopped containers: {', '.join(stopped)}")

    if failures:
        failure_messages = "; ".join([f"{name}: {err}" for name, err in failures])
        raise RuntimeError(f"Failed to stop some containers ({failure_messages})")


def parse_cli_args(argv: List[str] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Utility for managing RL simulation Docker containers.")

    # Common options
    parent_parser = argparse.ArgumentParser(add_help=False)

    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("start", help="Start the docker container.", parents=[parent_parser])
    subparsers.add_parser("enter", help="Enter an existing container.", parents=[parent_parser])
    attach_parser = subparsers.add_parser(
        "attach", help="Attach to a running container (interactive or by name).", parents=[parent_parser]
    )
    attach_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Container name to attach to. If not provided, will list containers for selection.",
    )

    # Add server-start command
    server_start_parser = subparsers.add_parser(
        "server-start", help="Start the rlsimdock server in a conda environment.", parents=[parent_parser]
    )
    server_start_parser.add_argument(
        "socket_path",
        type=str,
        help="Path to the Unix socket for server communication.",
    )
    server_start_parser.add_argument(
        "--env-type",
        type=str,
        required=True,
        help="Conda environment name (e.g., 'atari', 'dmc', 'metaworld').",
    )
    server_start_parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip pip install -e . step.",
    )

    subparsers.add_parser("stop", help="Stop the docker container.", parents=[parent_parser])
    subparsers.add_parser("pull", help="Pull the docker image from the registry.", parents=[parent_parser])
    subparsers.add_parser("build", help="Build the docker image from the Dockerfile.", parents=[parent_parser])
    subparsers.add_parser("stop-all", help="Stop every running rlsimlink container.")

    return parser.parse_args(argv)


def add_docker_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register `rlsimlink docker ...` CLI into an existing argparse tree."""
    docker_parser = subparsers.add_parser("docker", help="Manage RL simulation docker environments")
    docker_subparsers = docker_parser.add_subparsers(dest="docker_command", required=True)

    docker_subparsers.add_parser("start", help="Start the docker container.")
    docker_subparsers.add_parser("enter", help="Enter an existing container.")

    attach_parser = docker_subparsers.add_parser(
        "attach", help="Attach to a running container (interactive or by name)."
    )
    attach_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Container name to attach to. If not provided, will list containers for selection.",
    )

    server_start_parser = docker_subparsers.add_parser(
        "server-start", help="Start the rlsimlink server in a conda environment."
    )
    server_start_parser.add_argument("socket_path", type=str, help="Path to the Unix socket for server communication.")
    server_start_parser.add_argument(
        "--env-type",
        type=str,
        required=True,
        help="Conda environment name (e.g., 'atari', 'dmc', 'metaworld').",
    )
    server_start_parser.add_argument("--no-install", action="store_true", help="Skip pip install -e . step.")

    docker_subparsers.add_parser("stop", help="Stop the docker container.")
    docker_subparsers.add_parser("pull", help="Pull the docker image from the registry.")
    docker_subparsers.add_parser("build", help="Build the docker image from the Dockerfile.")
    docker_subparsers.add_parser("stop-all", help="Stop every running rlsimlink container.")

    return docker_parser


def main(args: argparse.Namespace):
    """Main function for the Docker utility."""
    cmd = getattr(args, "docker_command", None) or getattr(args, "command", None)

    # Check if docker is installed
    if not shutil.which("docker"):
        raise RuntimeError(
            "Docker is not installed! Please install Docker following "
            "https://docs.docker.com/engine/install/ubuntu/ and try again."
        )

    # Handle stop-all command without creating container interface
    if cmd == "stop-all":
        stop_all_rlsimdock_containers()
        return

    # Create container interface
    container_interface = RLContainerInterface(
        work_dir=Path(os.getcwd()).expanduser(),
        # tcp_port=args.port,
        # use_gpu=not args.no_gpu,
    )

    if cmd == "start":
        container_interface.start()
    elif cmd == "enter":
        container_interface.enter()
    elif cmd == "attach":
        container_interface.attach(container_name=getattr(args, "name", None))
    elif cmd == "server-start":
        container_interface.server_start(
            socket_path=args.socket_path,
            env_type=args.env_type,
            install_package=not args.no_install,
        )
    elif cmd == "stop":
        container_interface.stop()
    elif cmd == "pull":
        container_interface.pull()
    elif cmd == "build":
        container_interface.build()
    else:
        raise RuntimeError(f"Invalid command provided: {cmd}. Please check the help message.")


def cli_main():
    """Entry point for the rlsimdock command-line interface."""
    args_cli = parse_cli_args()
    main(args_cli)


if __name__ == "__main__":
    cli_main()
