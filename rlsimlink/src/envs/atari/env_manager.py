"""
Atari environment manager for handling environment creation and operations.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from .build_env import build_single_atari_env


class AtariEnvManager:
    """Manager for a single Atari environment instance."""

    def __init__(self):
        """Initialize the Atari environment manager."""
        self.env = None
        self.env_name = None
        self.seed = None
        self.image_size = None

    def create(self, env_name: str, seed: Optional[int] = None, image_size: Optional[Tuple[int, int]] = None):
        """Create an Atari environment.

        Args:
            env_name: Name of the Atari environment (e.g., "BoxingNoFrameskip-v4")
            seed: Random seed (optional)
            image_size: Image size as (height, width) tuple (optional)
        """
        self.env_name = env_name
        self.seed = seed
        self.image_size = image_size

        # Build the environment
        self.env = build_single_atari_env(env_name, seed, image_size)

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment.

        Args:
            **kwargs: Additional reset arguments

        Returns:
            Tuple of (observation, info)
        """
        if self.env is None:
            raise RuntimeError("Environment not created. Call create() first.")

        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.env is None:
            raise RuntimeError("Environment not created. Call create() first.")

        return self.env.step(action)

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def get_info(self) -> Dict[str, Any]:
        """Get environment information.

        Returns:
            Dictionary with environment metadata
        """
        return {
            "env_name": self.env_name,
            "seed": self.seed,
            "image_size": self.image_size,
            "created": self.env is not None,
        }

    def get_action_space(self, env_name: str, seed: Optional[int] = None, image_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """Get action space information by creating a dummy environment.

        Args:
            env_name: Name of the Atari environment (e.g., "BoxingNoFrameskip-v4")
            seed: Random seed (optional)
            image_size: Image size as (height, width) tuple (optional)

        Returns:
            Dictionary with action space information:
            {
                "dimensions": 1,
                "spaces": [
                    {
                        "type": "discrete",
                        "n": <number of actions>
                    }
                ]
            }
        """
        # Create a temporary dummy environment
        dummy_env = build_single_atari_env(env_name, seed, image_size)
        
        try:
            # Get action space from the environment
            action_space = dummy_env.action_space
            
            # For Atari, action space is Discrete
            from gymnasium.spaces import Discrete
            
            if isinstance(action_space, Discrete):
                action_space_info = {
                    "dimensions": 1,
                    "spaces": [
                        {
                            "type": "discrete",
                            "n": int(action_space.n)
                        }
                    ]
                }
            else:
                # Fallback for unexpected action space types
                action_space_info = {
                    "dimensions": 0,
                    "spaces": []
                }
            
            return action_space_info
        finally:
            # Clean up the dummy environment
            dummy_env.close()

