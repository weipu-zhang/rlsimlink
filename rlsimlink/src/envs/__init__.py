# Environment implementations

from typing import Any


def create_env_manager(env_type: str) -> Any:
    """Factory function to create environment manager based on type.

    Args:
        env_type: Type of environment (e.g., "atari", "dmc", "metaworld")

    Returns:
        Environment manager instance

    Raises:
        ValueError: If env_type is not supported
    """
    if env_type == "atari":
        from .atari import AtariEnvManager

        return AtariEnvManager()
    if env_type == "dmlab":
        from .dmlab import DMLabEnvManager

        return DMLabEnvManager()
    if env_type == "vizdoom":
        from .vizdoom import VizDoomEnvManager

        return VizDoomEnvManager()
    if env_type == "minerl":
        from .minerl import MineRLEnvManager

        return MineRLEnvManager()
    else:
        raise ValueError(
            f"Unsupported environment type: {env_type}. Currently supported types: 'atari', 'dmlab', 'vizdoom', 'minerl'."
        )


__all__ = ["create_env_manager"]
