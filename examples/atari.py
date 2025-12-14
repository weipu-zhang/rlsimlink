#!/usr/bin/env python3
"""Example: connect to an Atari server started via `rlsimlink serve --socket-id <id>`."""

import cv2
from tqdm import tqdm
import argparse

from rlsimlink import RLEnv, ActionSpace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # visualize option
    parser.add_argument("--visualize", action="store_true", help="Visualize the environment")
    # env name
    parser.add_argument("--env-name", type=str, default="BoxingNoFrameskip-v4", help="Environment name")
    parser.add_argument(
        "--socket-id",
        type=str,
        default=None,
        help="Optional socket identifier. If omitted, RLEnv auto-generates one and launches the server.",
    )
    args = parser.parse_args()

    env_kwargs = {}
    if args.socket_id:
        env_kwargs["socket_id"] = args.socket_id

    env = RLEnv(env_type="atari", env_name=args.env_name, **env_kwargs)
    action_space = ActionSpace(env.action_space_info, expand_dim=False)
    obs, info = env.reset()
    print(obs.shape)
    print(info)
    for i in tqdm(range(1000)):
        action = env.action_space.sample()
        if args.visualize:
            cv2.imshow("Atari Environment", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        obs, reward, terminated, truncated, info = env.step(action)
        # print(f"obs shape: {obs.shape}")
        # print(f"reward: {reward}")
        # print(f"terminated: {terminated}")
        # print(f"truncated: {truncated}")
        # print(f"info: {info}")
    env.close()
