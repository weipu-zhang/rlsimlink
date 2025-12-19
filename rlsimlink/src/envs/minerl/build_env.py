import gym
import minerl


def build_single_minerl_env(env_name):
    """Build a single MineRL environment.

    Args:
        env_name: Name of the MineRL environment (e.g., "MineRLBasaltFindCave-v0")

    Returns:
        MineRL environment instance
    """
    env = gym.make(env_name)
    return env
