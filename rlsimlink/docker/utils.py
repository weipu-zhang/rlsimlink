import re
import os
import subprocess
import grp
import getpass
from pathlib import Path


def get_hostname() -> str:
    """Get the hostname of the machine.

    Returns:
        The hostname of the machine.
    """
    return subprocess.run(
        ["hostname"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def is_user_in_docker_group() -> bool:
    """
    Check if the current user is a member of the 'docker' group.

    Returns:
        True if the user is in the 'docker' group, False otherwise.
    """
    try:
        docker_grp = grp.getgrnam("docker")
    except KeyError:
        # 'docker' group does not exist on the system
        return False

    # Get list of group IDs the process belongs to
    user_gids = os.getgroups()
    # Also check membership by username in the group's member list
    username = getpass.getuser()

    return (docker_grp.gr_gid in user_gids) or (username in docker_grp.gr_mem)


def get_architecture() -> str:
    """Get the architecture of the machine.

    Returns:
        The architecture of the machine.
    """
    return subprocess.run(
        ["uname", "-m"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def download_file(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"[INFO] Resource {dest} already exists. Skipping download.")
        return
    command = ["wget", url, "-O", dest]
    print(f"[INFO] Downloading resources with command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, cwd=str(Path(dest).resolve().parent))
    except subprocess.CalledProcessError as e:
        if url.startswith("https://github.com"):
            command[1] = "http://gh-proxy.com/" + command[1]  # Fallback to HTTP proxy if download fails
            print(f"[WARNING] Download failed with error: {e}. Retrying with proxy...")
            subprocess.run(command, check=True, capture_output=True, text=True, cwd=str(Path(dest).resolve().parent))
        else:
            raise e


def check_port_occupied(port: int) -> bool:
    """Check if a port is occupied.

    Args:
        port: The port to be checked.

    Returns:
        True if the port is occupied, False otherwise.
    """
    result = subprocess.run(
        ["netstat", "-tunlp"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    pattern = rf":{re.escape(str(port))}\b"
    return re.search(pattern, result) is not None


def load_env_version(default: str = "dmlab") -> str:
    """Load the environment type from docker-compose.yml by parsing the image tag.

    Returns:
        The environment type (e.g., 'dmlab', 'atari') from the image name.
    """
    compose_file = Path(__file__).resolve().parent / "docker-compose.yml"
    if compose_file.exists():
        with compose_file.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                # Look for image: wp-bit/rlsimlink:dmlab or wp-bit/rlsimlink:atari pattern
                if stripped.startswith("image:") and "wp-bit/rlsimlink:" in stripped:
                    # Extract environment type after the colon
                    parts = stripped.split("wp-bit/rlsimlink:", 1)
                    if len(parts) == 2:
                        env_type = parts[1].strip()
                        if env_type:
                            return env_type
    return default
