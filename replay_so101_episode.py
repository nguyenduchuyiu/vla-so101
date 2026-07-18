"""Replay recorded SO-101 commands through the evaluator's 25 Hz adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mediapy as media
from so101_nexus.lerobot_dataset import dataset_row_to_sim_qpos

from vla_data.counterfactual_collector import _gripper_limits, _make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source_index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--target_index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.episode) as episode:
        actions = episode["action"].copy()

    env = _make_env(("red", "orange"), ("green", "white"),
                    args.source_index, args.target_index, 256, 256)
    try:
        obs, info = env.reset(seed=args.seed)
        frames = [np.concatenate([obs["overhead_camera"], obs["wrist_camera"]], axis=1)]
        limits = _gripper_limits(env)
        for action_row in actions:
            command = dataset_row_to_sim_qpos(action_row, gripper_limits_rad=limits)
            current = env.unwrapped.data.ctrl[env.unwrapped._actuator_ids].copy()
            for control_step in range(2):
                alpha = (control_step + 1) / 2
                obs, _, terminated, truncated, info = env.step(
                    current + alpha * (command - current)
                )
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
            frames.append(np.concatenate([obs["overhead_camera"], obs["wrist_camera"]], axis=1))
    finally:
        env.close()

    print(f"success: {bool(info.get('success', False))}")
    print(f"is_obj_placed: {bool(info.get('is_obj_placed', False))}")
    print(f"is_grasped: {bool(info.get('is_grasped', False))}")
    print(f"obj_to_target_dist: {float(info.get('obj_to_target_dist', np.inf)):.6f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(args.output, frames, fps=25)
        print(f"video: {args.output}")


if __name__ == "__main__":
    main()
