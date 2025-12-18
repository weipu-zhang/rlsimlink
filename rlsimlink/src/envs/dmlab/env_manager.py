"""DMLab environment manager mirroring the Atari manager API."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

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

    def get_action_space(
        self,
        env_name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return action space metadata for the environment."""

        target_env = self.env
        cleanup_env = None

        if target_env is None:
            env_name = env_name or self.env_name
            if env_name is None:
                raise RuntimeError("Environment name required to query action space")
            build_kwargs = dict(self.env_kwargs)
            build_kwargs.update(kwargs)
            cleanup_env = build_single_dmlab_env(env_name, seed=seed or self.seed, **build_kwargs)
            target_env = cleanup_env

        try:
            action_count = self._extract_action_count(target_env)
            return {"dimensions": 1, "spaces": [{"type": "discrete", "n": int(action_count)}]}
        finally:
            if cleanup_env is not None:
                cleanup_env.close()

    def _step_internal(
        self, action_index: int, reset: bool
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
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
