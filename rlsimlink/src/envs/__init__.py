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
    else:
        raise ValueError(f"Unsupported environment type: {env_type}. Currently only 'atari' is supported.")


__all__ = ["create_env_manager"]
