"""
MineRL environment manager for handling environment creation and operations.
"""

from typing import Any, Dict, Optional, Tuple

from rlsimlink.src.common import ActionSpace
from .build_env import build_single_minerl_env
from rlsimlink.utils import Colors, print_log


class MineRLEnvManager:
    """Manager for a single MineRL environment instance."""

    def __init__(self):
        """Initialize the MineRL environment manager."""
        self.env = None
        self.env_name = None
        self.seed = None

    def create(self, env_name: str, seed: Optional[int] = None):
        """Create a MineRL environment.

        Args:
            env_name: Name of the MineRL environment (e.g., "MineRLBasaltFindCave-v0")
            seed: Random seed (optional)
        """
        self.env_name = env_name
        self.seed = seed

        # Build the environment
        self.env = build_single_minerl_env(env_name)
        print_log("LINK", f"Environment created: {Colors.PURPLE}MineRL: {env_name}{Colors.ENDC}")

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment.

        Args:
            **kwargs: Additional reset arguments

        Returns:
            Tuple of (observation, info)
        """
        if self.env is None:
            raise RuntimeError("Environment not created. Call create() first.")

        # MineRL uses old gym API (returns obs only on reset)
        obs = self.env.reset(**kwargs)
        # Return empty info dict for compatibility with new gym API
        return obs, {}

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.env is None:
            raise RuntimeError("Environment not created. Call create() first.")

        # MineRL uses old gym API (returns 4 values: obs, reward, done, info)
        obs, reward, done, info = self.env.step(action)
        # Convert to new gym API (5 values: obs, reward, terminated, truncated, info)
        # In old gym, done = terminated, truncated is always False
        return obs, reward, done, False, info

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def _describe_observation(self, observation: Any) -> str:
        """Return a friendly description of the observation payload."""
        if hasattr(observation, "shape"):
            return f"ndarray shape={tuple(observation.shape)}"
        if isinstance(observation, dict):
            desc_parts = []
            for key, value in observation.items():
                if hasattr(value, "shape"):
                    desc_parts.append(f"{key}: shape={tuple(value.shape)}")
                else:
                    desc_parts.append(f"{key}: {type(value).__name__}")
            return f"dict {{{', '.join(desc_parts)}}}"
        if isinstance(observation, (list, tuple)):
            return f"{type(observation).__name__} len={len(observation)}"
        return type(observation).__name__

    def test_env(self) -> Dict[str, Any]:
        """Run a lightweight smoke test on the managed environment."""
        if self.env is None:
            raise RuntimeError("Environment not created. Call create() first.")

        print_log("INFO", f"Resetting {Colors.BOLD}{self.env_name}{Colors.ENDC} for smoke test...")
        observation, info = self.reset()
        observation_desc = self._describe_observation(observation)
        print_log("SUCCESS", f"Reset observation -> {observation_desc}")
        print_log("INFO", f"Reset info: {info}")

        action_space_info = self.get_action_space(self.env_name, self.seed)
        print_log("INFO", f"Action space metadata: {action_space_info}")

        # TODO: use unified action space
        # action_space = ActionSpace(action_space_info, expand_dim=False)
        # random_action = action_space.sample()
        random_action = self.env.action_space.sample()

        # For MineRL BASALT environments, ESC action ends the episode, so set it to 0
        if isinstance(random_action, dict) and "ESC" in random_action:
            random_action["ESC"] = 0

        print_log("INFO", f"Sampled random action: {type(random_action).__name__}")
        observation, reward, terminated, truncated, step_info = self.step(random_action)
        post_observation_desc = self._describe_observation(observation)
        print_log(
            "SUCCESS",
            f"Step result -> reward={reward:.4f}, terminated={terminated}, truncated={truncated}, obs={post_observation_desc}",
        )
        print_log("INFO", f"Step info: {step_info}")

        # Extract pixel observation if available
        pixel_obs = observation.get("pov") if isinstance(observation, dict) else observation

        return {
            "reset_observation": observation_desc,
            "reset_info": info,
            "action_space": action_space_info,
            "reset_pixel_observation": pixel_obs,
            "step": {
                "action": random_action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": step_info,
            },
        }

    def get_info(self) -> Dict[str, Any]:
        """Get environment information.

        Returns:
            Dictionary with environment metadata
        """
        return {
            "env_name": self.env_name,
            "seed": self.seed,
            "created": self.env is not None,
        }

    def get_action_space(self, env_name: str, seed: Optional[int] = None) -> Dict[str, Any]:
        """Get action space information using the already created environment."""
        if self.env is None:
            raise RuntimeError("Environment not created. Call create() before querying action space.")

        action_space = self.env.action_space

        # MineRL uses Dict action space
        from gym.spaces import Dict as DictSpace, Discrete, Box, MultiDiscrete

        if isinstance(action_space, DictSpace):
            spaces = []
            for key, space in action_space.spaces.items():
                if isinstance(space, Discrete):
                    spaces.append({"type": "discrete", "n": int(space.n), "name": key})
                elif isinstance(space, Box):
                    spaces.append(
                        {
                            "type": "box",
                            "low": space.low.tolist() if hasattr(space.low, "tolist") else float(space.low),
                            "high": space.high.tolist() if hasattr(space.high, "tolist") else float(space.high),
                            "shape": list(space.shape),
                            "name": key,
                        }
                    )
                elif isinstance(space, MultiDiscrete):
                    spaces.append(
                        {
                            "type": "multi_discrete",
                            "nvec": space.nvec.tolist(),
                            "name": key,
                        }
                    )
            return {"dimensions": len(spaces), "spaces": spaces}

        # Fallback for unexpected action space types
        return {"dimensions": 0, "spaces": []}
