import gymnasium

from vizdoom import gymnasium_wrapper  # noqa
from . import env_wrapper


def build_single_vizdoom_env(env_name):
    # TODO seeds
    env = gymnasium.make(env_name, render_mode="human", frame_skip=4)
    env = env_wrapper.ExtractScreenWrapper(env)
    return env
