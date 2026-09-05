"""Replay one nominal episode using the controller targets recoverable from data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapy as media
import numpy as np

from cf_data.collect import make_env
from cf_data.core import Snapshot, restore_snapshot


def replay(data_dir: Path, episode_index: int, output: Path | None, settle_steps: int) -> dict:
    records = [json.loads(line) for line in (data_dir / "meta/nominal_episodes.jsonl").read_text().splitlines() if line]
    if not 0 <= episode_index < len(records):
        raise IndexError(f"episode must be in [0, {len(records)})")
    meta = records[episode_index]
    info_meta = json.loads((data_dir / "meta/info.json").read_text())
    with np.load(data_dir / meta["file"]) as episode:
        qpos = episode["snapshot.qpos"].copy()
        qvel = episode["snapshot.qvel"].copy()
        ctrl = episode["snapshot.ctrl"].copy()

    height, width, _ = info_meta["image_shape"]
    env = make_env(width, height, source_index=meta["source_id"], robot_init_qpos_noise=0.0)
    frames = []
    try:
        # Recreate static target body positions, then restore all dynamic state.
        scene_seed = int(meta["scene_id"].removeprefix("scene_"))
        env.reset(seed=scene_seed)
        env.set_objective(meta["source_id"], meta["target_id"])
        restore_snapshot(env, Snapshot(qpos[0], qvel[0], ctrl[0]))
        if env.control_mode != "pd_joint_pos":
            raise ValueError("Nominal data requires pd_joint_pos control")

        result = env._get_info()
        # Because frame t is captured before action[t], ctrl[t+1] is action[t].
        # The final action is not present; holding the last recovered target lets
        # the PD controller settle and reproduces the terminal placement check.
        commands = ctrl[1:]
        for command in commands:
            obs, _, terminated, truncated, result = env.step(command)
            if output is not None:
                frames.append(np.concatenate([obs["overhead_camera"], obs["wrist_camera"]], axis=1))
            if terminated or truncated:
                break
        if not result.get("success", False):
            for _ in range(settle_steps):
                obs, _, terminated, truncated, result = env.step(commands[-1])
                if output is not None:
                    frames.append(np.concatenate([obs["overhead_camera"], obs["wrist_camera"]], axis=1))
                if terminated or truncated:
                    break
        summary = {
            "episode_index": episode_index,
            "episode_id": meta["episode_id"],
            "source_id": meta["source_id"],
            "target_id": meta["target_id"],
            "recorded_success": bool(meta["success"]),
            "replay_success": bool(result.get("success", False)),
            "is_grasped": float(result.get("is_grasped", 0.0)),
            "obj_to_target_dist": float(result.get("obj_to_target_dist", float("nan"))),
            "commands_replayed": len(commands),
        }
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            media.write_video(output, frames, fps=info_meta["fps"])
            summary["video"] = str(output)
        return summary
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cf_nominal"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--settle-steps", type=int, default=50)
    args = parser.parse_args()
    if args.settle_steps < 0:
        raise ValueError("settle-steps must be nonnegative")
    print(json.dumps(replay(args.data, args.episode, args.output, args.settle_steps), indent=2))


if __name__ == "__main__":
    main()
