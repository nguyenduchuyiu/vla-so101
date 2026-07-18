"""CLI for collecting successful, language-conditioned oracle episodes."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import so101_nexus.mujoco  # noqa: F401 - registers Gymnasium environments
from so101_nexus import CubeObject, PickAndPlaceConfig, PickConfig, TouchConfig
from so101_nexus.config import MoveConfig
from so101_nexus.lerobot_dataset import sim_qpos_to_dataset_row
from so101_nexus.observations import (
    EndEffectorPose,
    GraspState,
    JointPositions,
    ObjectOffset,
    ObjectPose,
    OverheadCamera,
    TargetOffset,
    TargetPosition,
    WristCamera,
)

from vla_data.language import (
    canonical_instruction,
    make_instruction,
    task_spec_from_env,
)
from vla_data.oracle import Oracle

TASK_TO_ENV = {
    "pick_lift": "MuJoCoPickLift-v1",
    "pick_and_place": "MuJoCoPickAndPlace-v1",
    "touch": "MuJoCoTouch-v1",
    "move": "MuJoCoMove-v1",
}
COLORS = ("red", "green", "blue")
TARGET_COLORS = ("yellow", "white")
MOVE_DIRECTIONS = ("up", "down", "left", "right", "forward", "backward")


@dataclass
class EpisodeBuffer:
    state: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    environment_state: list[np.ndarray] = field(default_factory=list)
    overhead: list[np.ndarray] = field(default_factory=list)
    wrist: list[np.ndarray] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    success: list[float] = field(default_factory=list)
    done: list[float] = field(default_factory=list)
    timestamp: list[float] = field(default_factory=list)
    oracle_stage: list[str] = field(default_factory=list)

    def append(
        self,
        obs: dict[str, np.ndarray],
        privileged_state: np.ndarray,
        state_row: np.ndarray,
        action_row: np.ndarray,
        reward: float,
        success: bool,
        done: bool,
        timestamp: float,
        stage: str,
    ) -> None:
        self.state.append(state_row.astype(np.float32))
        self.action.append(action_row.astype(np.float32))
        self.environment_state.append(np.asarray(privileged_state, dtype=np.float32).copy())
        self.overhead.append(obs["overhead_camera"].copy())
        self.wrist.append(obs["wrist_camera"].copy())
        self.reward.append(float(reward))
        self.success.append(float(success))
        self.done.append(float(done))
        self.timestamp.append(float(timestamp))
        self.oracle_stage.append(stage)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            **{
                "observation.state": np.stack(self.state),
                "action": np.stack(self.action),
                "observation.environment_state": np.stack(self.environment_state),
                "observation.images.overhead": np.stack(self.overhead),
                "observation.images.wrist": np.stack(self.wrist),
                "reward": np.asarray(self.reward, dtype=np.float32),
                "success": np.asarray(self.success, dtype=np.float32),
                "done": np.asarray(self.done, dtype=np.float32),
                "timestamp": np.asarray(self.timestamp, dtype=np.float32),
                "oracle_stage": np.asarray(self.oracle_stage),
            },
        )


def _observations(task: str, width: int, height: int) -> list[Any]:
    common: list[Any] = [
        JointPositions(),
        EndEffectorPose(),
        WristCamera(width=width, height=height),
        OverheadCamera(width=width, height=height),
    ]
    if task in {"pick_lift", "touch"}:
        return common[:2] + [GraspState(), ObjectPose(), ObjectOffset()] + common[2:]
    if task == "pick_and_place":
        return (
            common[:2]
            + [GraspState(), TargetPosition(), ObjectPose(), ObjectOffset(), TargetOffset()]
            + common[2:]
        )
    if task == "move":
        return common[:2] + [TargetOffset()] + common[2:]
    raise ValueError(task)


def _make_env(task: str, seed: int, width: int, height: int, position_test: bool):
    observations = _observations(task, width, height)
    # The scripted side grasp is validated in this reachable annulus. Position
    # test episodes widen it slightly without entering known-unreachable poses.
    spawn = dict(
        spawn_center=(0.25, 0.0),
        spawn_min_radius=0.08,
        spawn_max_radius=0.19 if position_test else 0.18,
        spawn_angle_half_range_deg=55.0 if position_test else 50.0,
    )
    visual = dict(observations=observations, obs_mode="visual", **spawn)
    objects = [CubeObject(color=color) for color in COLORS]
    if task == "pick_lift":
        config = PickConfig(objects=objects, n_distractors=2, **visual)
    elif task == "pick_and_place":
        config = PickAndPlaceConfig(
            cube_colors=list(COLORS),
            target_colors=list(TARGET_COLORS),
            min_object_target_separation=0.06,
            **visual,
        )
    elif task == "touch":
        config = TouchConfig(objects=objects, n_distractors=2, **visual)
    elif task == "move":
        direction = MOVE_DIRECTIONS[seed % len(MOVE_DIRECTIONS)]
        # Move targets are relative to TCP and do not use scene spawn bounds.
        config = MoveConfig(
            direction=direction,
            target_distance=0.07,
            observations=observations,
            obs_mode="visual",
        )
    else:
        raise ValueError(task)
    return gym.make(
        TASK_TO_ENV[task],
        config=config,
        control_mode="pd_joint_pos",
        max_episode_steps=1000,
    )


def _split_for_episode(index: int) -> str:
    mod = index % 10
    if mod == 8:
        return "test_paraphrase"
    if mod == 9:
        return "test_position"
    return "train"


def _gripper_limits(env: Any) -> tuple[float, float]:
    u = env.unwrapped
    return float(u._target_low[5]), float(u._target_high[5])


def _collect_attempt(
    task: str,
    seed: int,
    split: str,
    width: int,
    height: int,
    record_stride: int,
) -> tuple[EpisodeBuffer | None, dict[str, Any]]:
    env_id = TASK_TO_ENV[task]
    env = _make_env(task, seed, width, height, split == "test_position")
    try:
        obs, info = env.reset(seed=seed)
        spec = task_spec_from_env(env_id, env)
        language_rng = np.random.default_rng(seed ^ 0x5EED)
        instruction = make_instruction(
            spec, language_rng, held_out=split == "test_paraphrase"
        )
        canonical = canonical_instruction(env)
        oracle = Oracle(env_id, env)
        limits = _gripper_limits(env)
        buffer = EpisodeBuffer()
        privileged = np.asarray(info["privileged_state"], dtype=np.float32)
        final_info = info
        max_control_steps = sum(stage.steps for stage in oracle.stages) + 20

        for control_step in range(max_control_steps):
            action_rad, stage = oracle.select_action()
            state_rad = np.asarray(obs["state"], dtype=np.float64)
            state_row = sim_qpos_to_dataset_row(state_rad.copy(), gripper_limits_rad=limits)
            action_row = sim_qpos_to_dataset_row(
                action_rad.astype(np.float64).copy(), gripper_limits_rad=limits
            )
            next_obs, reward, terminated, truncated, final_info = env.step(action_rad)
            done = bool(terminated or truncated)
            should_record = control_step % record_stride == 0 or done
            if should_record:
                buffer.append(
                    obs,
                    privileged,
                    state_row,
                    action_row,
                    reward,
                    bool(final_info.get("success", False)),
                    done,
                    control_step * 0.02,
                    stage,
                )
            obs = next_obs
            privileged = np.asarray(final_info["privileged_state"], dtype=np.float32)
            if done:
                break

        success = bool(final_info.get("success", False))
        metadata = {
            "seed": seed,
            "task": task,
            "env_id": env_id,
            "split": split,
            "task_spec": spec.to_dict(),
            "canonical_instruction": canonical,
            "language_instruction": instruction,
            "success": success,
            "num_frames": len(buffer.state),
            "final_info": {
                key: bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
                for key, value in final_info.items()
                if key in {
                    "success",
                    "is_grasped",
                    "is_obj_placed",
                    "lift_height",
                    "obj_to_target_dist",
                    "tcp_to_obj_dist",
                    "tcp_to_target_dist",
                    "move_displacement",
                }
            },
        }
        return (buffer if success else None), metadata
    except RuntimeError as exc:
        return None, {"seed": seed, "task": task, "split": split, "error": str(exc)}
    finally:
        env.close()


def collect_dataset(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    (output / "episodes").mkdir(parents=True)
    (output / "meta").mkdir()

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    unknown = set(tasks) - set(TASK_TO_ENV)
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}; choose from {sorted(TASK_TO_ENV)}")

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    next_seed = args.seed
    episode_index = 0
    max_attempts = args.episodes * args.max_attempts_per_episode
    attempts = 0
    while episode_index < args.episodes and attempts < max_attempts:
        task = tasks[episode_index % len(tasks)]
        split = _split_for_episode(episode_index)
        buffer, metadata = _collect_attempt(
            task,
            next_seed,
            split,
            args.width,
            args.height,
            args.record_stride,
        )
        attempts += 1
        next_seed += 1
        if buffer is None:
            failures.append(metadata)
            continue
        filename = f"episode_{episode_index:06d}.npz"
        buffer.save(output / "episodes" / filename)
        metadata.update({"episode_index": episode_index, "file": f"episodes/{filename}"})
        records.append(metadata)
        print(
            f"saved {episode_index + 1}/{args.episodes}: {task}, "
            f"seed={metadata['seed']}, frames={metadata['num_frames']}, split={split}"
        )
        episode_index += 1

    if episode_index < args.episodes:
        raise RuntimeError(
            f"Only collected {episode_index}/{args.episodes} successful episodes "
            f"after {attempts} attempts"
        )

    with (output / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / "meta" / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    info = {
        "format": "so101_language_npz_v1",
        "so101_nexus_version": "0.4.8",
        "robot_type": "so101_mujoco",
        "fps": 50 // args.record_stride,
        "total_episodes": len(records),
        "total_frames": sum(record["num_frames"] for record in records),
        "image_shape": [args.height, args.width, 3],
        "action_semantics": "absolute_joint_position",
        "joint_order": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        "units": {
            "observation.state[0:5]": "degrees",
            "action[0:5]": "degrees",
            "observation.state[5]": "gripper_percent_0_100",
            "action[5]": "gripper_percent_0_100",
        },
        "language_scope": "one instruction per episode; constant for all frames",
    }
    (output / "meta" / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/so101_language"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--tasks", default="pick_lift,pick_and_place,touch,move")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--record-stride", type=int, default=2)
    parser.add_argument("--max-attempts-per-episode", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.record_stride <= 0 or 50 % args.record_stride != 0:
        raise ValueError("--record-stride must be a positive divisor of 50")
    output = collect_dataset(args)
    print(f"dataset ready: {output}")


if __name__ == "__main__":
    main()
