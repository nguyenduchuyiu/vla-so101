"""Replay recorded SO-101 commands through the evaluator's 25 Hz adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mediapy as media
from so101_nexus.lerobot_dataset import dataset_row_to_sim_qpos

from old_vla_data.counterfactual_collector import _gripper_limits, _make_env


def _episode_meta(episode_path: Path) -> dict:
    """Look up this episode's scene_seed/source_index/target_index from the sibling meta.

    The NPZ stores no scene metadata; it lives in ``meta/episodes.jsonl`` next to
    ``episodes/``. Episode index is parsed from the filename (``episode_000007``
    -> 7). The collector reset with ``seed=scene_seed``, so replay must too.
    """
    episode_index = int(episode_path.stem.split("_")[-1])
    meta_path = episode_path.parent.parent / "meta" / "episodes.jsonl"
    with meta_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row["episode_index"] == episode_index:
                return row
    raise FileNotFoundError(f"episode {episode_index} not found in {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    # Defaults are inferred from the episode's meta row; pass explicitly to override.
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--source_index", type=int, default=None)
    parser.add_argument("--target_index", type=int, default=None)
    parser.add_argument("--output", type=Path, default="outputs/out.mp4")
    args = parser.parse_args()

    meta = _episode_meta(args.episode)
    seed = args.seed if args.seed is not None else meta["scene_seed"]
    source_index = args.source_index if args.source_index is not None else meta["source_index"]
    target_index = args.target_index if args.target_index is not None else meta["target_index"]

    with np.load(args.episode) as episode:
        actions = episode["action"].copy()

    env = _make_env(("red", "orange"), ("green", "white"),
                    source_index, target_index, 256, 256)
    try:
        obs, info = env.reset(seed=seed)
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
