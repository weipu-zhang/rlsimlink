"""Wrappers for VizDoom environments."""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple


class ExtractScreenWrapper(gym.ObservationWrapper):
    """Wrapper to extract 'screen' from dictionary observation.

    VizDoom environments return dictionary observations with 'screen' key.
    This wrapper extracts just the screen array to make it compatible with
    standard RL pipelines that expect array observations.
    """

    def __init__(self, env: gym.Env):
        """Initialize the wrapper.

        Args:
            env: The VizDoom environment to wrap
        """
        super().__init__(env)

        # Update observation space to reflect the screen array
        if isinstance(env.observation_space, gym.spaces.Dict):
            if "screen" in env.observation_space.spaces:
                self.observation_space = env.observation_space.spaces["screen"]
            else:
                raise ValueError(
                    "Environment observation space does not contain 'screen' key. "
                    f"Available keys: {list(env.observation_space.spaces.keys())}"
                )
        else:
            raise ValueError(f"Expected Dict observation space, got {type(env.observation_space)}")

    def observation(self, observation: Dict[str, Any]) -> np.ndarray:
        """Extract the screen from the observation dictionary.

        Args:
            observation: Dictionary observation containing 'screen' key

        Returns:
            Screen array (typically RGB image)
        """
        if isinstance(observation, dict) and "screen" in observation:
            return observation["screen"]
        else:
            raise ValueError(f"Expected dict observation with 'screen' key, got {type(observation)}")
