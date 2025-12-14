# Gymnasium version need to be lower than 1.0 for this version
# gymnasium[atari,accept-rom-license]==0.29.1 is preferred
import gymnasium
from gymnasium.core import Env
from . import env_wrapper
import numpy as np
from typing import Tuple


def build_single_atari_env(env_name, seed, image_size=None) -> Tuple[Env]:
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.StochasticSeedEnvWrapper(env)
    env = env_wrapper.NoopResetWrapper(env, noop_max=30)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    if image_size is not None:
        env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfoWrapper(env)
    env = env_wrapper.AutoSqueezeActionDimWrapper(env)
    env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=5000)  # truncate episode

    return env
