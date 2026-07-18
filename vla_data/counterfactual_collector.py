"""Generate grouped counterfactual pick-and-place demonstrations."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from so101_nexus import CubeObject, PickAndPlaceConfig
from so101_nexus.lerobot_dataset import sim_qpos_to_dataset_row

from vla_data.collector import EpisodeBuffer, _gripper_limits, _observations
from vla_data.counterfactual_env import ENV_ID
from vla_data.language import canonical_instruction, make_instruction, task_spec_from_env
from vla_data.oracle import Oracle


def _group_design(group_index: int) -> dict[str, Any]:
    """Assign a whole counterfactual group to one semantic split."""
    slot = group_index % 12
    if slot < 8:
        if slot % 2 == 0:
            return {
                "split": "train",
                "source_colors": ("red", "orange"),
                "target_colors": ("green", "white"),
            }
        return {
            "split": "train",
            "source_colors": ("blue", "orange"),
            "target_colors": ("yellow", "white"),
        }
    if slot == 8:
        return {
            "split": "test_seen_semantics_new_scene",
            "source_colors": ("red", "orange"),
            "target_colors": ("green", "white"),
        }
    if slot == 9:
        return {
            "split": "test_paraphrase",
            "source_colors": ("blue", "orange"),
            "target_colors": ("yellow", "white"),
        }
    if slot == 10:
        return {
            "split": "test_heldout_pair",
            "source_colors": ("red", "blue"),
            "target_colors": ("green", "yellow"),
        }
    return {
        "split": "test_heldout_color",
        "source_colors": ("purple", "orange"),
        "target_colors": ("black", "white"),
    }


def _make_env(
    source_colors: tuple[str, str],
    target_colors: tuple[str, str],
    source_index: int,
    target_index: int,
    width: int,
    height: int,
):
    config = PickAndPlaceConfig(
        objects=[CubeObject(color=color) for color in source_colors],
        target_colors=list(target_colors),
        observations=_observations("pick_and_place", width, height),
        obs_mode="visual",
        goal_thresh=0.03,
    )
    return gym.make(
        ENV_ID,
        config=config,
        source_index=source_index,
        target_index=target_index,
        target_colors=target_colors,
        control_mode="pd_joint_pos",
        max_episode_steps=1200,
    )


def _scene_spec(env: Any, source_colors: tuple[str, str], target_colors: tuple[str, str]) -> dict:
    u = env.unwrapped
    return {
        "sources": [
            {
                "object_type": "cube",
                "color": color,
                "size": "small",
                "initial_pose": u.data.qpos[slot.qpos_addr : slot.qpos_addr + 7].tolist(),
            }
            for color, slot in zip(source_colors, u._slots, strict=True)
        ],
        "targets": [
            {
                "object_type": "tray",
                "color": color,
                "position": u.data.xpos[body_id].tolist(),
            }
            for color, body_id in zip(target_colors, u._target_body_ids, strict=True)
        ],
    }


def _is_heldout_pair(spec: dict[str, Any]) -> bool:
    source = spec["source"]["color"]
    target = spec["target"]["color"]
    return (source, target) in {("red", "yellow"), ("blue", "green")}


def _collect_episode(
    *,
    scene_seed: int,
    group_index: int,
    source_colors: tuple[str, str],
    target_colors: tuple[str, str],
    source_index: int,
    target_index: int,
    split: str,
    width: int,
    height: int,
    record_stride: int,
) -> tuple[EpisodeBuffer | None, dict[str, Any]]:
    env = _make_env(
        source_colors, target_colors, source_index, target_index, width, height
    )
    try:
        obs, info = env.reset(seed=scene_seed)
        initial_scene_spec = _scene_spec(env, source_colors, target_colors)
        spec = task_spec_from_env(ENV_ID, env)
        instruction_rng = np.random.default_rng(
            scene_seed * 101 + source_index * 11 + target_index
        )
        instruction = make_instruction(
            spec,
            instruction_rng,
            held_out=split == "test_paraphrase",
        )
        oracle = Oracle(ENV_ID, env)
        limits = _gripper_limits(env)
        privileged = np.asarray(info["privileged_state"], dtype=np.float32)
        buffer = EpisodeBuffer()
        final_info = info
        max_steps = sum(stage.steps for stage in oracle.stages) + 20
        for control_step in range(max_steps):
            action_rad, stage = oracle.select_action()
            state_rad = np.asarray(obs["state"], dtype=np.float64)
            state_row = sim_qpos_to_dataset_row(
                state_rad.copy(), gripper_limits_rad=limits
            )
            action_row = sim_qpos_to_dataset_row(
                action_rad.astype(np.float64).copy(), gripper_limits_rad=limits
            )
            next_obs, reward, terminated, truncated, final_info = env.step(action_rad)
            done = bool(terminated or truncated)
            if control_step % record_stride == 0 or done:
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
        spec_dict = spec.to_dict()
        success = bool(final_info.get("success", False)) and not bool(
            final_info.get("is_grasped", False)
        )
        metadata = {
            "scene_seed": scene_seed,
            "counterfactual_group_id": group_index,
            "split": split,
            "source_index": source_index,
            "target_index": target_index,
            "task": "counterfactual_pick_and_place",
            "env_id": ENV_ID,
            "scene_spec": initial_scene_spec,
            "task_spec": spec_dict,
            "canonical_instruction": canonical_instruction(env),
            "language_instruction": instruction,
            "heldout_pair": split == "test_heldout_pair" and _is_heldout_pair(spec_dict),
            "success": success,
            "num_frames": len(buffer.state),
            "final_info": {
                "success": bool(final_info.get("success", False)),
                "is_grasped": float(final_info.get("is_grasped", 0.0)),
                "is_obj_placed": bool(final_info.get("is_obj_placed", False)),
                "obj_to_target_dist": float(final_info.get("obj_to_target_dist", np.inf)),
            },
        }
        return (buffer if success else None), metadata
    except RuntimeError as exc:
        return None, {
            "scene_seed": scene_seed,
            "counterfactual_group_id": group_index,
            "split": split,
            "source_index": source_index,
            "target_index": target_index,
            "error": str(exc),
        }
    finally:
        env.close()


def collect(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    (output / "episodes").mkdir(parents=True)
    (output / "meta").mkdir()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    scene_seed = args.seed
    group_index = 0
    episode_index = 0
    attempts = 0
    while group_index < args.groups:
        design = _group_design(group_index)
        group_results: list[tuple[EpisodeBuffer, dict[str, Any]]] = []
        group_failed = False
        for source_index in range(2):
            for target_index in range(2):
                buffer, metadata = _collect_episode(
                    scene_seed=scene_seed,
                    group_index=group_index,
                    source_colors=design["source_colors"],
                    target_colors=design["target_colors"],
                    source_index=source_index,
                    target_index=target_index,
                    split=design["split"],
                    width=args.width,
                    height=args.height,
                    record_stride=args.record_stride,
                )
                if buffer is None:
                    group_failed = True
                    failures.append(metadata)
                    break
                group_results.append((buffer, metadata))
            if group_failed:
                break
        attempts += 1
        scene_seed += 1
        if group_failed:
            if attempts >= args.groups * args.max_attempts_per_group:
                raise RuntimeError(
                    f"collected only {group_index}/{args.groups} complete groups"
                )
            continue
        for buffer, metadata in group_results:
            filename = f"episode_{episode_index:06d}.npz"
            buffer.save(output / "episodes" / filename)
            metadata.update(
                {"episode_index": episode_index, "file": f"episodes/{filename}"}
            )
            records.append(metadata)
            episode_index += 1
        print(
            f"saved group {group_index + 1}/{args.groups}: seed={scene_seed - 1}, "
            f"split={design['split']}, episodes=4"
        )
        group_index += 1

    with (output / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / "meta" / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    info = {
        "format": "so101_language_npz_v1",
        "dataset_kind": "grouped_counterfactual",
        "so101_nexus_version": "0.4.8",
        "robot_type": "so101_mujoco",
        "fps": 50 // args.record_stride,
        "total_groups": args.groups,
        "episodes_per_group": 4,
        "total_episodes": len(records),
        "total_frames": sum(record["num_frames"] for record in records),
        "image_shape": [args.height, args.width, 3],
        "action_semantics": "absolute_joint_position",
        "oracle_stage_transition": "observable_state_with_step_timeout",
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
        "group_split_rule": "all episodes from a scene group belong to one split",
    }
    (output / "meta" / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/so101_counterfactual"))
    parser.add_argument("--groups", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--record-stride", type=int, default=2)
    parser.add_argument("--max-attempts-per-group", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.groups <= 0:
        raise ValueError("--groups must be positive")
    if args.record_stride <= 0 or 50 % args.record_stride != 0:
        raise ValueError("--record-stride must divide 50")
    print(f"dataset ready: {collect(args)}")


if __name__ == "__main__":
    main()
