"""
VizDoom environment manager for handling environment creation and operations.
"""

from typing import Any, Dict, Optional, Tuple

from rlsimlink.src.common import ActionSpace
from .build_env import build_single_vizdoom_env
from rlsimlink.utils import Colors, print_log


class VizDoomEnvManager:
    """Manager for a single VizDoom environment instance."""

    def __init__(self):
        """Initialize the VizDoom environment manager."""
        self.env = None
        self.env_name = None
        self.seed = None

    def create(self, env_name: str, seed: Optional[int] = None):
        """Create a VizDoom environment.

        Args:
            env_name: Name of the VizDoom environment (e.g., "VizdoomBasic-v0")
            seed: Random seed (optional)
        """
        self.env_name = env_name
        self.seed = seed

        # Build the environment
        self.env = build_single_vizdoom_env(env_name)
        print_log("LINK", f"Environment created: {Colors.PURPLE}VizDoom: {env_name}{Colors.ENDC}")

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

    def _describe_observation(self, observation: Any) -> str:
        """Return a friendly description of the observation payload."""
        if hasattr(observation, "shape"):
            return f"ndarray shape={tuple(observation.shape)}"
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

        action_space = ActionSpace(action_space_info, expand_dim=False)
        random_action = action_space.sample()
        print_log("INFO", f"Sampled random action: {random_action}")
        observation, reward, terminated, truncated, step_info = self.step(random_action)
        post_observation_desc = self._describe_observation(observation)
        print_log(
            "SUCCESS",
            f"Step result -> reward={reward:.4f}, terminated={terminated}, truncated={truncated}, obs={post_observation_desc}",
        )
        print_log("INFO", f"Step info: {step_info}")

        return {
            "reset_observation": observation_desc,
            "reset_info": info,
            "action_space": action_space_info,
            "reset_pixel_observation": observation,
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

        # For VizDoom, action space is typically Discrete
        from gymnasium.spaces import Discrete

        if isinstance(action_space, Discrete):
            return {"dimensions": 1, "spaces": [{"type": "discrete", "n": int(action_space.n)}]}
        # Fallback for unexpected action space types
        return {"dimensions": 0, "spaces": []}
