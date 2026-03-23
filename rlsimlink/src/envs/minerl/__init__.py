# MineRL environment implementation
# TODO: test can now only be started with screens, need to be able to run everywhere
# TODO: only conda passed the test, docker has GLFW issue
# docker: exec `xhost +` at host, set `DISPLAY` env var at container equals to host's `DISPLAY`

from .env_manager import MineRLEnvManager

__all__ = ["MineRLEnvManager"]
