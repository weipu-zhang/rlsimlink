#!/usr/bin/env python3
"""Example: run a DMLab environment managed automatically by rlsimlink."""

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
        default="rooms_keys_doors_puzzle",
        help='DMLab level to load. Default: "rooms_keys_doors_puzzle".',
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of interaction steps to run.")
    parser.add_argument("--repeat", type=int, default=4, help="Simulator steps per action.")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(64, 64),
        help="Frame size as HEIGHT WIDTH. Default: 64 64.",
    )
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Environment split to use.")
    parser.add_argument("--actions", choices=["impala", "popart"], default="popart", help="Action set to use.")
    parser.add_argument("--episodic", dest="episodic", action="store_true", help="Force episodic termination.")
    parser.add_argument("--no-episodic", dest="episodic", action="store_false", help="Disable episodic termination.")
    parser.add_argument("--text", dest="text", action="store_true", help="Enable instruction text observations.")
    parser.add_argument("--no-text", dest="text", action="store_false", help="Disable instruction text observations.")
    parser.set_defaults(episodic=True, text=None)
    return parser.parse_args()


def main():
    args = parse_args()
    frame_size = tuple(args.size)

    env = RLEnv(
        env_type="dmlab",
        env_name=args.env_name,
        seed=args.seed,
        repeat=args.repeat,
        size=frame_size,
        mode=args.mode,
        actions=args.actions,
        episodic=args.episodic,
        text=args.text,
    )

    try:
        obs, info = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Reset info: {info}")

        for _ in tqdm(range(args.steps)):
            if args.visualize:
                cv2.imshow("DMLab Environment", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
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
