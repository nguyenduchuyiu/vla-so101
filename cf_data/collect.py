"""Stage A: collect nominal pick-and-place trajectories with per-frame snapshots.

Each scene (one reset seed) yields NUM_OBJECTIVES x NUM_TARGETS nominal episodes,
one per (source cube, place target) pair, all run from the same initial state S0.
Per control step we store rendered images, the proprio row (deg + gripper %), the
phase derived from the oracle's current stage, and a full physics snapshot
(qpos/qvel/ctrl) so build.py can restore the exact state for counterfactual
rollout. Only the individual unsuccessful (source, target) episode is dropped;
other successful episodes from the same scene are retained.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import signal
from pathlib import Path

import numpy as np
from tqdm import tqdm

from so101_nexus import CubeObject
from so101_nexus.config import PickAndPlaceConfig

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

from cf_data.core import (
    NUM_OBJECTIVES,
    NUM_TARGETS,
    OBJECTIVE_COLORS,
    TARGET_COLORS,
    instruction,
    variant_for,
    qpos_to_row,
    restore_snapshot,
    save_snapshot,
    stage_to_phase,
)
from cf_data.env import CFMultiObjectEnv, ENV_ID
from cf_data.oracle import Oracle

CONTROL_DT = 0.02


def _observations(width: int, height: int):
    return [
        JointPositions(),
        EndEffectorPose(),
        GraspState(),
        TargetPosition(),
        ObjectPose(),
        ObjectOffset(),
        TargetOffset(),
        WristCamera(width=width, height=height),
        OverheadCamera(width=width, height=height),
    ]


def make_env(width: int, height: int, source_index: int, robot_init_qpos_noise: float) -> CFMultiObjectEnv:
    config = PickAndPlaceConfig(
        objects=[CubeObject(color=c) for c in OBJECTIVE_COLORS],
        target_colors=list(TARGET_COLORS),
        observations=_observations(width, height),
        obs_mode="visual",
        goal_thresh=0.03,
    )
    return CFMultiObjectEnv(
        config,
        source_index=source_index,
        target_colors=TARGET_COLORS,
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
    settle_steps = 0
    # Keep applying the final held target after the scripted stages finish so
    # the PD controller can settle and the environment can emit terminal.
    while steps < max_steps:
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
        if stage == "finished":
            settle_steps += 1
            if settle_steps >= 20:
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


def _collect_scene_worker(
    staging_root: str,
    scene_index: int,
    seed: int,
    width: int,
    height: int,
    robot_noise: float,
) -> dict:
    """Collect one scene in an isolated process and write its images locally."""
    scene_out = Path(staging_root) / f"scene-{scene_index:06d}"
    (scene_out / "episodes").mkdir(parents=True)
    scene_id = f"scene_{seed:06d}"
    initial_state_id = f"init_{seed:06d}"
    env = make_env(width, height, source_index=0, robot_init_qpos_noise=robot_noise)
    episodes = []
    failures = []
    try:
        env.reset(seed=seed)
        s0 = save_snapshot(env)
        for source_id in range(NUM_OBJECTIVES):
            for target_id in range(NUM_TARGETS):
                restore_snapshot(env, s0)
                env.set_objective(source_id, target_id)
                ep_id = f"{scene_id}_s{source_id}_t{target_id}"
                try:
                    oracle = Oracle(ENV_ID, env)
                    max_steps = sum(stage.steps for stage in oracle.stages) + 20
                    frames, rollout_info, success = _collect_objective(env, oracle, max_steps)
                except (RuntimeError, ValueError) as exc:
                    failures.append(
                        {
                            "scene_id": scene_id,
                            "source_id": source_id,
                            "target_id": target_id,
                            "error": str(exc),
                        }
                    )
                    continue
                if not success or len(frames) < 8:
                    failures.append(
                        {
                            "scene_id": scene_id,
                            "source_id": source_id,
                            "target_id": target_id,
                            "success": success,
                            "num_frames": len(frames),
                        }
                    )
                    continue
                local_index = source_id * NUM_TARGETS + target_id
                rel = _save_episode(scene_out, local_index, frames)
                episodes.append(
                    {
                        "temporary_path": str(scene_out / rel),
                        "meta": {
                            "scene_id": scene_id,
                            "scene_seed": seed,
                            "initial_state_id": initial_state_id,
                            "scene_index": scene_index,
                            "episode_id": ep_id,
                            "objective_id": source_id,
                            "source_id": source_id,
                            "objective_color": OBJECTIVE_COLORS[source_id],
                            "target_id": target_id,
                            "target_color": TARGET_COLORS[target_id],
                            "instruction": instruction(source_id, target_id, variant_for(ep_id)),
                            "is_counterfactual": False,
                            "success": success,
                            "num_frames": len(frames),
                            "final_obj_to_target_dist": float(
                                rollout_info.get("obj_to_target_dist", float("nan"))
                            ),
                        },
                    }
                )
    finally:
        env.close()
    return {
        "scene_index": scene_index,
        "scene_id": scene_id,
        "episodes": episodes,
        "failures": failures,
    }


def collect(args: argparse.Namespace) -> Path:
    out = args.out.resolve()
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    (out / "episodes").mkdir(parents=True)
    (out / "meta").mkdir()
    staging_root = out / ".collect_shards"
    staging_root.mkdir()

    records: list[dict] = []
    failures: list[dict] = []
    episode_index = 0
    width, height = args.width, args.height
    requested_workers = int(getattr(args, "workers", 1))
    if requested_workers < 1:
        raise ValueError("workers must be positive")
    workers = min(requested_workers, args.scenes)
    jobs = [
        (str(staging_root), scene_index, args.seed + scene_index, width, height, args.robot_noise)
        for scene_index in range(args.scenes)
    ]
    if workers == 1:
        scene_results = [_collect_scene_worker(*job) for job in tqdm(jobs, desc="collect scenes")]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            scene_results = list(
                tqdm(
                    executor.map(_collect_scene_worker, *zip(*jobs, strict=True)),
                    total=len(jobs),
                    desc=f"collect scenes ({workers} workers)",
                )
            )

    for result in sorted(scene_results, key=lambda item: item["scene_index"]):
        failures.extend(result["failures"])
        for failure in result["failures"]:
            detail = failure.get(
                "error",
                f"success={failure.get('success')} frames={failure.get('num_frames')}",
            )
            tqdm.write(
                f"WARNING {failure['scene_id']} source={failure['source_id']} "
                f"target={failure['target_id']}: {detail}; dropping this objective only"
            )
        for item in result["episodes"]:
            rel = f"episodes/ep_{episode_index:06d}.npz"
            shutil.move(item["temporary_path"], out / rel)
            meta = item["meta"]
            meta.update({"episode_index": episode_index, "file": rel})
            records.append(meta)
            episode_index += 1
        print(
            f"scene {result['scene_index'] + 1}/{args.scenes} ({result['scene_id']}): "
            f"saved {len(result['episodes'])} episodes, dropped {len(result['failures'])}"
        )
    shutil.rmtree(staging_root)

    with (out / "meta" / "nominal_episodes.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (out / "meta" / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    info = {
        "format": "cf_nominal_v1",
        "dataset_kind": "counterfactual_nominal",
        "num_objectives": NUM_OBJECTIVES,
        "objective_colors": list(OBJECTIVE_COLORS),
        "num_targets": NUM_TARGETS,
        "target_colors": list(TARGET_COLORS),
        "num_scenes_requested": args.scenes,
        "num_scenes_saved": len({r["scene_id"] for r in records}),
        "collection_workers": workers,
        "total_episodes": len(records),
        "total_frames": sum(r["num_frames"] for r in records),
        "image_shape": [height, width, 3],
        "control_dt": CONTROL_DT,
        "fps": int(round(1.0 / CONTROL_DT)),
        "action_semantics": "pd_joint_pos_command_recoverable_from_snapshot_ctrl",
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
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.scenes <= 0:
        raise ValueError("--scenes must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("collect timed out")))
    # 15 episodes/scene (NUM_OBJECTIVES x NUM_TARGETS) at 256x256 render; the old
    # 150*scenes constant predates the 3x episode bump and timed out. ~30s/episode.
    signal.alarm(max(60, 30 * args.scenes * NUM_OBJECTIVES * NUM_TARGETS))
    print(f"dataset ready: {collect(args)}")


if __name__ == "__main__":
    main()
