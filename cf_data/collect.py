"""Stage A: collect nominal pick-and-place trajectories with per-frame snapshots.

Each scene (one reset seed) yields 5 nominal episodes, one per objective, all run
from the same initial state S0. Per control step we store rendered images, the
proprio row (deg + gripper %), the phase derived from the oracle's current stage,
and a full physics snapshot (qpos/qvel/ctrl) so build.py can restore the exact state
for counterfactual rollout. Only successful episodes are kept; a scene is dropped
entirely if any of its 5 objectives fails, so the 5 objectives stay balanced per scene.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
from pathlib import Path

import numpy as np
from so101_nexus import CubeObject
from so101_nexus.config import PickAndPlaceConfig

from cf_data.core import (
    OBJECTIVE_COLORS,
    TARGET_COLOR,
    objective_instruction,
    qpos_to_row,
    restore_snapshot,
    save_snapshot,
    stage_to_phase,
)
from cf_data.env import CFMultiObjectEnv, ENV_ID
from old_vla_data.collector import _observations
from old_vla_data.oracle import Oracle

CONTROL_DT = 0.02


def make_env(width: int, height: int, source_index: int, robot_init_qpos_noise: float) -> CFMultiObjectEnv:
    config = PickAndPlaceConfig(
        objects=[CubeObject(color=c) for c in OBJECTIVE_COLORS],
        target_colors=[TARGET_COLOR],
        observations=_observations("pick_and_place", width, height),
        obs_mode="visual",
        goal_thresh=0.03,
    )
    return CFMultiObjectEnv(
        config,
        source_index=source_index,
        target_colors=(TARGET_COLOR,),
        render_mode=None,
        control_mode="pd_joint_pos",
        robot_init_qpos_noise=robot_init_qpos_noise,
    )


def _collect_objective(env, oracle, max_steps: int) -> tuple[list[dict], dict, bool]:
    """Run one objective from the current (already-restored) state; record every control step."""
    frames: list[dict] = []
    obs = env._get_obs()  # render at the restored initial state (t=0)
    info: dict = {}
    steps = 0
    while not oracle.finished and steps < max_steps:
        action, stage = oracle.select_action()
        frames.append(
            {
                "overhead": obs["overhead_camera"].copy(),
                "wrist": obs["wrist_camera"].copy(),
                "proprio": qpos_to_row(env._get_current_qpos()),
                "phase": stage_to_phase(stage),
                "stage": stage,
                "qpos": env.data.qpos.copy(),
                "qvel": env.data.qvel.copy(),
                "ctrl": env.data.ctrl[env._actuator_ids].copy(),
            }
        )
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    success = bool(info.get("success", False)) and not bool(info.get("is_grasped", False))
    return frames, info, success


def _save_episode(out: Path, episode_index: int, frames: list[dict]) -> str:
    filename = f"ep_{episode_index:06d}.npz"
    np.savez_compressed(
        out / "episodes" / filename,
        **{
            "observation.state": np.stack([f["proprio"] for f in frames]).astype(np.float32),
            "observation.images.overhead": np.stack([f["overhead"] for f in frames]),
            "observation.images.wrist": np.stack([f["wrist"] for f in frames]),
            "phase": np.asarray([f["phase"] for f in frames], dtype=np.int8),
            "oracle_stage": np.asarray([f["stage"] for f in frames]),
            "snapshot.qpos": np.stack([f["qpos"] for f in frames]),
            "snapshot.qvel": np.stack([f["qvel"] for f in frames]),
            "snapshot.ctrl": np.stack([f["ctrl"] for f in frames]).astype(np.float32),
            "timestamp": (np.arange(len(frames)) * CONTROL_DT).astype(np.float32),
        },
    )
    return f"episodes/{filename}"


def collect(args: argparse.Namespace) -> Path:
    out = args.out.resolve()
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    (out / "episodes").mkdir(parents=True)
    (out / "meta").mkdir()

    records: list[dict] = []
    failures: list[dict] = []
    episode_index = 0
    width, height = args.width, args.height
    env = make_env(width, height, source_index=0, robot_init_qpos_noise=args.robot_noise)

    for scene_index in range(args.scenes):
        seed = args.seed + scene_index
        scene_id = f"scene_{seed:06d}"
        initial_state_id = f"init_{seed:06d}"
        env.reset(seed=seed)
        s0 = save_snapshot(env)
        # All 5 objectives share the same stage plan length; probe objective 0 to size max_steps.
        env.set_objective(0)
        max_steps = sum(s.steps for s in Oracle(ENV_ID, env).stages) + 20

        scene_episodes: list[tuple[list[dict], dict]] = []
        scene_failed = False
        for objective_id in range(len(OBJECTIVE_COLORS)):
            restore_snapshot(env, s0)
            env.set_objective(objective_id)
            oracle = Oracle(ENV_ID, env)
            try:
                frames, info, success = _collect_objective(env, oracle, max_steps)
            except RuntimeError as exc:
                failures.append({"scene_id": scene_id, "objective_id": objective_id, "error": str(exc)})
                scene_failed = True
                break
            if not success or len(frames) < 8:
                failures.append(
                    {
                        "scene_id": scene_id,
                        "objective_id": objective_id,
                        "success": success,
                        "num_frames": len(frames),
                    }
                )
                scene_failed = True
                break
            scene_episodes.append(
                (
                    frames,
                    {
                        "scene_id": scene_id,
                        "initial_state_id": initial_state_id,
                        "scene_index": scene_index,
                        "episode_id": f"{scene_id}_obj{objective_id}",
                        "objective_id": objective_id,
                        "objective_color": OBJECTIVE_COLORS[objective_id],
                        "instruction": objective_instruction(objective_id),
                        "is_counterfactual": False,
                        "success": success,
                        "num_frames": len(frames),
                        "final_obj_to_target_dist": float(info.get("obj_to_target_dist", float("nan"))),
                    },
                )
            )

        if scene_failed:
            continue
        for frames, meta in scene_episodes:
            rel = _save_episode(out, episode_index, frames)
            meta.update({"episode_index": episode_index, "file": rel})
            records.append(meta)
            episode_index += 1
        print(f"scene {scene_index + 1}/{args.scenes} ({scene_id}): saved 5 episodes")

    env.close()

    with (out / "meta" / "nominal_episodes.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (out / "meta" / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    info = {
        "format": "cf_nominal_v1",
        "dataset_kind": "counterfactual_nominal",
        "num_objectives": len(OBJECTIVE_COLORS),
        "objective_colors": list(OBJECTIVE_COLORS),
        "target_color": TARGET_COLOR,
        "num_scenes_requested": args.scenes,
        "num_scenes_saved": len({r["scene_id"] for r in records}),
        "total_episodes": len(records),
        "total_frames": sum(r["num_frames"] for r in records),
        "image_shape": [height, width, 3],
        "control_dt": CONTROL_DT,
        "fps": int(round(1.0 / CONTROL_DT)),
        "action_semantics": "absolute_joint_position",
        "joint_order": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        "units": {
            "observation.state[0:5]": "degrees",
            "observation.state[5]": "gripper_percent_0_100",
        },
        "phase_names": ["REACH_PICK", "GRASP", "REACH_PLACE", "PLACE"],
        "env_id": ENV_ID,
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"saved {len(records)} episodes across {info['num_scenes_saved']} scenes "
        f"({len(failures)} failed objective attempts)"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/cf_nominal"))
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--robot-noise", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.scenes <= 0:
        raise ValueError("--scenes must be positive")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("collect timed out")))
    signal.alarm(max(60, 150 * args.scenes))
    print(f"dataset ready: {collect(args)}")


if __name__ == "__main__":
    main()