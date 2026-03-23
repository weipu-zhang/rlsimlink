#!/usr/bin/env python3
"""Example: run a VizDoom environment managed automatically by rlsimlink."""

import argparse

import cv2
from tqdm import tqdm

from rlsimlink import RLEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Visualize the environment with OpenCV.")
    parser.add_argument(
        "--env-name",
        type=str,
        default="VizdoomBasic-v0",
        help='VizDoom environment to load. Default: "VizdoomBasic-v0".',
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of interaction steps to run.")
    return parser.parse_args()


def main():
    args = parse_args()

    env = RLEnv(
        env_type="vizdoom",
        env_name=args.env_name,
    )

    try:
        obs, info = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Reset info: {info}")

        for _ in tqdm(range(args.steps)):
            if args.visualize:
                cv2.imshow("VizDoom Environment", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print(f"Episode finished. Reward: {reward}, terminated={terminated}, truncated={truncated}")
                obs, info = env.reset()
    finally:
        env.close()
        if args.visualize:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
