"""DMLab environment manager mirroring the Atari manager API."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from rlsimlink.src.common import ActionSpace
from rlsimlink.utils import Colors, print_log
from .build_env import build_single_dmlab_env


class DMLabEnvManager:
    """Manager for a single DMLab environment instance."""

    def __init__(self) -> None:
        self.env_name: Optional[str] = None
        self.seed: Optional[int] = None
        self.env = None
        self.env_kwargs: Dict[str, Any] = {}

    def create(
        self,
        env_name: str,
        seed: Optional[int] = None,
        repeat: int = 4,
        size: Tuple[int, int] = (64, 64),
        mode: str = "train",
        actions: str = "popart",
        episodic: bool = True,
        text: Optional[bool] = None,
        **extra_kwargs: Any,
    ) -> None:
        """Create a DMLab environment instance."""

        self.env_name = env_name
        self.seed = seed
        self.env_kwargs = {
            "repeat": repeat,
            "size": size,
            "mode": mode,
            "actions": actions,
            "episodic": episodic,
            "text": text,
        }
        self.env_kwargs.update(extra_kwargs)

        print_log("INFO", f"Creating environment: {Colors.BOLD}dmlab/{env_name}{Colors.ENDC}")
        self.env = build_single_dmlab_env(env_name, seed=seed, **self.env_kwargs)
        print_log("LINK", f"Environment created: {Colors.PURPLE}DMLab: {env_name}{Colors.ENDC}")

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return the initial observation."""

        if self.env is None:
            raise RuntimeError("Environment not created. Call create() first.")

        observation, _, _, _, info = self._step_internal(action_index=0, reset=True)
        return observation, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""

        if self.env is None:
            raise RuntimeError("Environment not created. Call create() first.")

        action_index = self._normalize_action(action)
        return self._step_internal(action_index=action_index, reset=False)

    def close(self) -> None:
        """Close the underlying environment."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def _describe_observation(self, observation: Any) -> str:
        """Return a short description for logging."""
        if hasattr(observation, "shape"):
            return f"ndarray shape={tuple(observation.shape)}"
        return type(observation).__name__

    def test_env(self) -> Dict[str, Any]:
        """Run a quick smoke test for the active DMLab environment."""
        if self.env is None:
            raise RuntimeError("Environment not created. Call create() first.")

        print_log("INFO", f"Resetting {Colors.BOLD}{self.env_name}{Colors.ENDC} for smoke test...")
        observation, info = self.reset()
        reset_desc = self._describe_observation(observation)
        print_log("SUCCESS", f"Reset observation -> {reset_desc}")
        print_log("INFO", f"Reset info: {info}")

        action_space_info = self.get_action_space()
        print_log("INFO", f"Action space metadata: {action_space_info}")

        action_space = ActionSpace(action_space_info, expand_dim=False)
        random_action = action_space.sample()
        print_log("INFO", f"Sampled random action: {random_action}")
        observation, reward, terminated, truncated, step_info = self.step(random_action)
        step_desc = self._describe_observation(observation)
        print_log(
            "SUCCESS",
            f"Step result -> reward={reward:.4f}, terminated={terminated}, truncated={truncated}, obs={step_desc}",
        )
        print_log("INFO", f"Step info: {step_info}")

        return {
            "reset_observation": reset_desc,
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

    def get_action_space(
        self,
        env_name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return action space metadata for the environment."""
        if self.env is None:
            raise RuntimeError("Environment not created. Call create() before querying action space.")

        action_count = self._extract_action_count(self.env)
        return {"dimensions": 1, "spaces": [{"type": "discrete", "n": int(action_count)}]}

    def _step_internal(self, action_index: int, reset: bool) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        payload = {"action": int(action_index), "reset": bool(reset)}
        timestep = self.env.step(payload)

        observation = np.asarray(timestep.get("image"), dtype=np.uint8)
        reward = float(timestep.get("reward", 0.0))
        is_last = bool(timestep.get("is_last", False))
        is_terminal = bool(timestep.get("is_terminal", is_last))
        terminated = is_terminal
        truncated = is_last and not is_terminal

        info: Dict[str, Any] = {
            "is_first": bool(timestep.get("is_first", False)),
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

        if "instr" in timestep:
            info["instr"] = np.asarray(timestep["instr"], dtype=np.float32)

        return observation, reward, terminated, truncated, info

    def _normalize_action(self, action: Any) -> int:
        if isinstance(action, (list, tuple)):
            if not action:
                raise ValueError("Action list is empty")
            return int(action[0])
        return int(action)

    def _extract_action_count(self, env: Any) -> int:
        action_set = getattr(env, "_actions", None)
        if action_set is None:
            raise RuntimeError("DMLab environment missing action set definition")
        return len(action_set)


__all__ = ["DMLabEnvManager"]
