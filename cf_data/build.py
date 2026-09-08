"""Stage B-E: build the balanced counterfactual dataset from nominal episodes.

Reads the Stage-A nominal dataset (episodes/*.npz + meta/nominal_episodes.jsonl) and
emits, into the same directory:

  * cf_anchors/an_<anchor_id>.npz  -- per-anchor expert command prefixes.
  * cf_trajectories/tr_<branch_id>.npz -- every real CF observation/action from
    the shared anchor through terminal; nominal branches reference their recorded
    episode suffix without duplicating it.
  * meta/anchors.jsonl            -- one line per anchor (output C, the balanced
    training set; each line lists its branches).
  * meta/eval_pairs.jsonl          -- REACH_PICK + REACH_PLACE counterfactual groups
    in the test split (output D, branches sharing images/proprio, differing instruction).
  * meta/stats.json               -- dataset statistics report (output E).

Two counterfactual axes, each at its own decision frame:

  * REACH_PICK source-CF: keep state/image/proprio, swap the SOURCE cube over all
    NUM_OBJECTIVES (target fixed = episode target). Forces reading the source word.
  * REACH_PLACE target-CF: keep state/image/proprio (cube already grasped), swap the
    TARGET over all NUM_TARGETS (source fixed = episode source). Forces reading the
    target word. A fresh oracle from a grasped-state anchor replays near-no-op
    approach/descend/close/lift stages before moving to the new target, so the future
    chunk diverges later than source-CF -- the contrast is weaker but valid (the
    place location still differs).

Each CF group picks ONE instruction-template variant (deterministic from anchor_id)
shared by all branches, so branches differ ONLY in the swapped color word. An anchor
is kept only if every non-anchor (source|target) oracle plans successfully. The
nominal branch of a group reuses the nominal episode's own continuation
(frames t+1..t+H), so it is exact and free. Nominal-only anchors (GRASP/PLACE) carry
a single nominal branch.

Balance: we pick N_rp anchors from each phase, where N_rp is the size of the smallest
pool among {valid REACH_PICK groups, valid REACH_PLACE groups, GRASP, PLACE}. Nominal
branches = 4*N_rp (one per phase group); CF branches = (NUM_OBJECTIVES-1)*N_rp +
(NUM_TARGETS-1)*N_rp. So nominal/total = 4/(NUM_OBJECTIVES+NUM_TARGETS+2) (0.4 for
5 sources, 3 targets).
"""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import json
import os
import shutil
import signal
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cf_data.collect import CONTROL_DT, make_env
from cf_data.core import (
    GRASP,
    NUM_OBJECTIVES,
    NUM_TARGETS,
    OBJECTIVE_COLORS,
    PLACE,
    REACH_PICK,
    REACH_PLACE,
    Snapshot,
    TARGET_COLORS,
    instruction,
    variant_for,
    qpos_to_row,
    restore_snapshot,
    split_for_scene,
    stage_to_phase,
    step_physics,
)
from cf_data.env import ENV_ID
from cf_data.oracle import Oracle

PHASE_POOLS = (REACH_PICK, GRASP, REACH_PLACE, PLACE)


def _prepare_scene(env, meta: dict) -> None:
    """Restore scene-level model state (notably fixed target body positions)."""
    seed = int(meta.get("scene_seed", str(meta["scene_id"]).rsplit("_", 1)[-1]))
    if getattr(env, "_cf_prepared_scene_seed", None) != seed:
        env.reset(seed=seed)
        env._cf_prepared_scene_seed = seed


def _load_nominal(in_dir: Path) -> list[dict]:
    """Load metadata + the non-image arrays for every nominal episode."""
    metas = [json.loads(line) for line in (in_dir / "meta" / "nominal_episodes.jsonl").read_text().splitlines() if line]
    episodes: list[dict] = []
    for meta in metas:
        path = in_dir / meta["file"]
        with np.load(path) as npz:
            episodes.append(
                {
                    "meta": meta,
                    "path": str(path),
                    "state": npz["observation.state"].astype(np.float32),
                    "phase": npz["phase"],
                    "snap_qpos": npz["snapshot.qpos"].astype(np.float64),
                    "snap_qvel": npz["snapshot.qvel"].astype(np.float64),
                    "snap_ctrl": npz["snapshot.ctrl"].astype(np.float64),
                }
            )
    return episodes


def _future_from_nominal(ep: dict, t: int, horizon: int) -> np.ndarray:
    """Recover expert position commands from the next frame's held controller target.

    Frame t is recorded immediately before action[t] is applied. Consequently
    snapshot.ctrl[t+1] equals action[t]. The final unobserved command cannot be
    recovered and tail chunks repeat the last known command.
    """
    ctrl = ep["snap_ctrl"][t + 1 : t + 1 + horizon]
    if len(ctrl) == 0:
        ctrl = ep["snap_ctrl"][t : t + 1]
    if len(ctrl) < horizon:
        ctrl = np.concatenate([ctrl, np.repeat(ctrl[-1:], horizon - len(ctrl), axis=0)])
    return np.stack([qpos_to_row(row) for row in ctrl]).astype(np.float32)


def _cf_valid(env, snap: Snapshot, source_id: int, target_id: int,
              anchor_src: int, anchor_tgt: int) -> bool:
    """A counterfactual is valid iff the oracle can plan (IK) for (source,target) from the anchor state."""
    if source_id == anchor_src and target_id == anchor_tgt:
        return True  # nominal branch uses the stored continuation, no oracle needed
    restore_snapshot(env, snap)
    env.set_objective(source_id, target_id)
    try:
        Oracle(ENV_ID, env)
    except (RuntimeError, ValueError):
        # least_squares raises ValueError when q0 is outside the IK bounds
        # (e.g. off-manifold anchor poses); treat as unplannable.
        return False
    return True


def _start_oracle_at_phase(oracle: Oracle, phase: int) -> None:
    if phase == REACH_PICK:
        return
    for index, stage in enumerate(oracle.stages):
        if stage_to_phase(stage.name) == phase:
            oracle.stage_index = index
            oracle.stage_step = 0
            return
    raise ValueError(f"oracle has no stage for phase {phase}")


def _roll_cf(
    env, snap: Snapshot, source_id: int, target_id: int, horizon: int, start_phase: int = REACH_PICK
) -> np.ndarray:
    """Roll a fresh oracle and retain its actual absolute joint-position commands."""
    restore_snapshot(env, snap)
    env.set_objective(source_id, target_id)
    oracle = Oracle(ENV_ID, env)
    _start_oracle_at_phase(oracle, start_phase)
    chunk: list[np.ndarray] = []
    while len(chunk) < horizon:
        if oracle.finished:
            break
        action, _ = oracle.select_action()
        chunk.append(qpos_to_row(action))
        step_physics(env, action)
    if not chunk:  # oracle already finished at the anchor (degenerate); hold the anchor
        chunk.append(qpos_to_row(env._get_current_qpos()))
    while len(chunk) < horizon:
        chunk.append(chunk[-1].copy())
    return np.stack(chunk).astype(np.float32)


def _roll_cf_trajectory(
    env, snap: Snapshot, source_id: int, target_id: int, start_phase: int = REACH_PICK
) -> tuple[dict[str, np.ndarray], dict]:
    """Roll one CF branch to completion and retain every observation and command.

    Frames follow the collector convention: observation/state are recorded just
    before ``action[t]`` is applied.  Unlike ``_roll_cf`` this deliberately uses
    ``env.step`` so cameras and task termination are available.
    """
    restore_snapshot(env, snap)
    env.set_objective(source_id, target_id)
    oracle = Oracle(ENV_ID, env)
    _start_oracle_at_phase(oracle, start_phase)
    max_steps = sum(stage.steps for stage in oracle.stages[oracle.stage_index :]) + 20
    obs = env._get_obs()
    frames: list[dict] = []
    info: dict = {}
    terminated = truncated = False
    settle_steps = 0
    # Continue holding the final target after the plan finishes. Success also
    # requires the robot to settle, which can take several control steps.
    while len(frames) < max_steps:
        action, stage = oracle.select_action()
        frame = {
            "overhead": obs["overhead_camera"].copy(),
            "wrist": obs["wrist_camera"].copy(),
            "state": qpos_to_row(env._get_current_qpos()),
            "action": qpos_to_row(action),
            "phase": stage_to_phase(stage),
            "terminal": False,
        }
        obs, _, terminated, truncated, info = env.step(action)
        frame["terminal"] = bool(terminated or truncated)
        frames.append(frame)
        if terminated or truncated:
            break
        if stage == "finished":
            settle_steps += 1
            if settle_steps >= 20:
                break
    if not frames:
        raise RuntimeError("counterfactual rollout produced no frames")
    if oracle.finished:
        frames[-1]["terminal"] = True
    success = bool(info.get("success", False)) and not bool(info.get("is_grasped", False))
    if not success:
        diagnostics = {
            key: info.get(key)
            for key in ("is_obj_placed", "is_grasped", "is_robot_static", "obj_to_target_dist")
        }
        raise RuntimeError(
            f"counterfactual rollout did not succeed after {len(frames)} steps "
            f"(terminated={terminated}, truncated={truncated}, diagnostics={diagnostics})"
        )
    frames[-1]["terminal"] = True
    arrays = {
        "observation.state": np.stack([f["state"] for f in frames]).astype(np.float32),
        "observation.images.overhead": np.stack([f["overhead"] for f in frames]),
        "observation.images.wrist": np.stack([f["wrist"] for f in frames]),
        "action": np.stack([f["action"] for f in frames]).astype(np.float32),
        "phase": np.asarray([f["phase"] for f in frames], dtype=np.int8),
        "terminal": np.asarray([f["terminal"] for f in frames], dtype=bool),
        "timestamp": (np.arange(len(frames)) * CONTROL_DT).astype(np.float32),
    }
    return arrays, info


def _scan_anchors(episodes: list[dict], anchor_stride: int) -> dict[int, list[tuple[int, int, int]]]:
    """Per-phase anchor candidate pools. REACH_PICK and REACH_PLACE candidates are strided
    (both run per-branch IK filtering, which is costly). GRASP/PLACE are nominal-only (1
    branch), so all their frames are cheap candidates.

    Returns {phase: [(episode_idx, frame, objective_id), ...]}.
    """
    pools: dict[int, list[tuple[int, int, int]]] = {p: [] for p in PHASE_POOLS}
    for ep_idx, ep in enumerate(episodes):
        phases = ep["phase"]
        obj_id = ep["meta"]["objective_id"]
        n = len(phases)
        last_usable = n - 2  # need at least one real future frame after the anchor
        for p in (REACH_PICK, REACH_PLACE):
            indices = np.where(phases == p)[0]
            indices = indices[indices <= last_usable]
            if len(indices) == 0:
                continue
            # Stride candidates; keep the first frame of the phase unconditionally.
            pick = indices[::anchor_stride]
            if pick[0] != indices[0]:
                pick = np.r_[indices[0], pick]
            for t in pick:
                pools[p].append((ep_idx, int(t), obj_id))
        for p in (GRASP, PLACE):
            for t in np.where(phases == p)[0]:
                if t <= last_usable:
                    pools[p].append((ep_idx, int(t), obj_id))
    return pools


def _filter_rp(
    env, episodes: list[dict], rp_candidates: list[tuple[int, int, int]], num_objectives: int
) -> list[tuple[int, int, int]]:
    """Keep REACH_PICK candidates whose non-anchor sources all plan successfully
    (target fixed = the episode's target_id)."""
    valid: list[tuple[int, int, int]] = []
    for ep_idx, t, obj_id in tqdm(rp_candidates, desc="filter REACH_PICK", leave=False):
        ep = episodes[ep_idx]
        _prepare_scene(env, ep["meta"])
        tgt = ep["meta"]["target_id"]
        snap = Snapshot(ep["snap_qpos"][t], ep["snap_qvel"][t], ep["snap_ctrl"][t])
        if all(_cf_valid(env, snap, j, tgt, obj_id, tgt) for j in range(num_objectives)):
            valid.append((ep_idx, t, obj_id))
    return valid


def _filter_place(
    env, episodes: list[dict], place_candidates: list[tuple[int, int, int]], num_targets: int
) -> list[tuple[int, int, int]]:
    """Keep REACH_PLACE candidates whose non-anchor targets all plan successfully
    (source fixed = the episode's source_id)."""
    valid: list[tuple[int, int, int]] = []
    for ep_idx, t, obj_id in tqdm(place_candidates, desc="filter REACH_PLACE", leave=False):
        ep = episodes[ep_idx]
        _prepare_scene(env, ep["meta"])
        src = ep["meta"]["objective_id"]
        tgt = ep["meta"]["target_id"]
        snap = Snapshot(ep["snap_qpos"][t], ep["snap_qvel"][t], ep["snap_ctrl"][t])
        if all(_cf_valid(env, snap, src, k, src, tgt) for k in range(num_targets)):
            valid.append((ep_idx, t, obj_id))
    return valid


def _anchor_id(episode_id: str, frame: int) -> str:
    return f"{episode_id}_f{frame:05d}"


def _build_rp_group(
    env,
    episodes: list[dict],
    ep_idx: int,
    t: int,
    horizon: int,
    num_objectives: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """REACH_PICK source-CF group: 1 nominal branch + (num_objectives-1) CF branches.

    Source varies over all objectives; target is fixed = the episode's target_id.
    All branches share one instruction-template variant (constant within the group).
    Returns (future_chunks, source_ids, target_ids, cf_flags, branches).
    """
    ep = episodes[ep_idx]
    obj_id = ep["meta"]["objective_id"]
    tgt = ep["meta"]["target_id"]
    snap = Snapshot(ep["snap_qpos"][t], ep["snap_qvel"][t], ep["snap_ctrl"][t])
    anchor_id = _anchor_id(ep["meta"]["episode_id"], t)
    variant = variant_for(anchor_id)
    futures: list[np.ndarray] = []
    src_ids: list[int] = []
    tgt_ids: list[int] = []
    cf_flags: list[bool] = []
    branches: list[dict] = []
    for j in range(num_objectives):
        is_cf = j != obj_id
        try:
            future = _roll_cf(env, snap, j, tgt, horizon) if is_cf else _future_from_nominal(ep, t, horizon)
        except (RuntimeError, ValueError) as exc:
            tqdm.write(
                f"WARNING {anchor_id} source={j} target={tgt}: {exc}; dropping this branch only"
            )
            continue
        futures.append(future)
        src_ids.append(j)
        tgt_ids.append(tgt)
        cf_flags.append(is_cf)
        branches.append(
            {
                "branch_id": f"{anchor_id}_s{j}_t{tgt}",
                "objective_id": j,
                "source_id": j,
                "target_id": tgt,
                "instruction": instruction(j, tgt, variant),
                "is_counterfactual": is_cf,
            }
        )
    return (
        np.stack(futures).astype(np.float32),
        np.asarray(src_ids, dtype=np.int8),
        np.asarray(tgt_ids, dtype=np.int8),
        np.asarray(cf_flags, dtype=bool),
        branches,
    )


def _build_place_group(
    env,
    episodes: list[dict],
    ep_idx: int,
    t: int,
    horizon: int,
    num_targets: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """REACH_PLACE target-CF group: 1 nominal branch + (num_targets-1) CF branches.

    Target varies over all targets; source is fixed = the episode's source_id (cube
    already grasped at this anchor). All branches share one instruction-template
    variant (constant within the group).
    Returns (future_chunks, source_ids, target_ids, cf_flags, branches).
    """
    ep = episodes[ep_idx]
    src = ep["meta"]["objective_id"]
    tgt = ep["meta"]["target_id"]
    snap = Snapshot(ep["snap_qpos"][t], ep["snap_qvel"][t], ep["snap_ctrl"][t])
    anchor_id = _anchor_id(ep["meta"]["episode_id"], t)
    variant = variant_for(anchor_id)
    futures: list[np.ndarray] = []
    src_ids: list[int] = []
    tgt_ids: list[int] = []
    cf_flags: list[bool] = []
    branches: list[dict] = []
    for k in range(num_targets):
        is_cf = k != tgt
        try:
            future = (
                _roll_cf(env, snap, src, k, horizon, start_phase=REACH_PLACE)
                if is_cf
                else _future_from_nominal(ep, t, horizon)
            )
        except (RuntimeError, ValueError) as exc:
            tqdm.write(
                f"WARNING {anchor_id} source={src} target={k}: {exc}; dropping this branch only"
            )
            continue
        futures.append(future)
        src_ids.append(src)
        tgt_ids.append(k)
        cf_flags.append(is_cf)
        branches.append(
            {
                "branch_id": f"{anchor_id}_s{src}_t{k}",
                "objective_id": src,
                "source_id": src,
                "target_id": k,
                "instruction": instruction(src, k, variant),
                "is_counterfactual": is_cf,
            }
        )
    return (
        np.stack(futures).astype(np.float32),
        np.asarray(src_ids, dtype=np.int8),
        np.asarray(tgt_ids, dtype=np.int8),
        np.asarray(cf_flags, dtype=bool),
        branches,
    )


def _build_nominal_branch(
    episodes: list[dict],
    ep_idx: int,
    t: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Nominal-only anchor (GRASP/PLACE): a single nominal branch."""
    ep = episodes[ep_idx]
    obj_id = ep["meta"]["objective_id"]
    tgt = ep["meta"]["target_id"]
    anchor_id = _anchor_id(ep["meta"]["episode_id"], t)
    variant = variant_for(anchor_id)
    future = _future_from_nominal(ep, t, horizon)
    branches = [
        {
            "branch_id": f"{anchor_id}_s{obj_id}_t{tgt}",
            "objective_id": int(obj_id),
            "source_id": int(obj_id),
            "target_id": int(tgt),
            "instruction": instruction(int(obj_id), int(tgt), variant),
            "is_counterfactual": False,
        }
    ]
    return (
        future[None].astype(np.float32),
        np.asarray([obj_id], dtype=np.int8),
        np.asarray([tgt], dtype=np.int8),
        np.asarray([False], dtype=bool),
        branches,
    )


_BUILD_WORKER_EPISODES: list[dict] | None = None
_BUILD_WORKER_ENV = None
_BUILD_WORKER_IN_DIR: Path | None = None


def _init_build_worker(in_dir: str, width: int, height: int) -> None:
    """Give each process its own MuJoCo model/data and EGL rendering context."""
    global _BUILD_WORKER_EPISODES, _BUILD_WORKER_ENV, _BUILD_WORKER_IN_DIR
    _BUILD_WORKER_IN_DIR = Path(in_dir)
    _BUILD_WORKER_EPISODES = _load_nominal(_BUILD_WORKER_IN_DIR)
    _BUILD_WORKER_ENV = make_env(width, height, source_index=0, robot_init_qpos_noise=0.0)
    atexit.register(_BUILD_WORKER_ENV.close)


def _filter_candidate_worker(payload: tuple[int, tuple[int, int, int], int]):
    phase, candidate, branch_count = payload
    ep_idx, t, obj_id = candidate
    episodes = _BUILD_WORKER_EPISODES
    env = _BUILD_WORKER_ENV
    if episodes is None or env is None:
        raise RuntimeError("build worker was not initialized")
    ep = episodes[ep_idx]
    _prepare_scene(env, ep["meta"])
    src = int(ep["meta"]["objective_id"])
    tgt = int(ep["meta"]["target_id"])
    snap = Snapshot(ep["snap_qpos"][t], ep["snap_qvel"][t], ep["snap_ctrl"][t])
    if phase == REACH_PICK:
        valid = all(_cf_valid(env, snap, j, tgt, obj_id, tgt) for j in range(branch_count))
    elif phase == REACH_PLACE:
        valid = all(_cf_valid(env, snap, src, k, src, tgt) for k in range(branch_count))
    else:
        raise ValueError(f"cannot CF-filter phase {phase}")
    return candidate if valid else None


def _emit_anchor_core(
    env,
    episodes: list[dict],
    in_dir: Path,
    phase: int,
    ep_idx: int,
    t: int,
    horizon: int,
    num_objectives: int,
    num_targets: int,
    scene_rank: dict[int, int],
    n_scenes: int,
) -> dict:
    ep = episodes[ep_idx]
    meta = ep["meta"]
    _prepare_scene(env, meta)
    anchor_id = _anchor_id(meta["episode_id"], t)
    if phase == REACH_PICK:
        future_chunks, obj_ids, tgt_ids, cf_flags, branches = _build_rp_group(
            env, episodes, ep_idx, t, horizon, num_objectives
        )
    elif phase == REACH_PLACE:
        future_chunks, obj_ids, tgt_ids, cf_flags, branches = _build_place_group(
            env, episodes, ep_idx, t, horizon, num_targets
        )
    else:
        future_chunks, obj_ids, tgt_ids, cf_flags, branches = _build_nominal_branch(
            episodes, ep_idx, t, horizon
        )

    anchor_proprio = ep["state"][t].astype(np.float32)
    kept = []
    failures = []
    snap = Snapshot(ep["snap_qpos"][t], ep["snap_qvel"][t], ep["snap_ctrl"][t])
    for branch_index, branch in enumerate(branches):
        if not branch["is_counterfactual"]:
            branch["trajectory_kind"] = "nominal_suffix"
            branch["trajectory_path"] = meta["file"]
            branch["trajectory_start_frame"] = t
            kept.append(branch_index)
            continue
        try:
            trajectory, rollout_info = _roll_cf_trajectory(
                env,
                snap,
                int(branch["source_id"]),
                int(branch["target_id"]),
                start_phase=phase,
            )
            prefix = trajectory["action"][:horizon]
            expected = future_chunks[branch_index, : len(prefix)]
            if not np.allclose(prefix, expected, atol=1e-4, rtol=1e-5):
                raise RuntimeError("full CF rollout differs from validated action prefix")
            rel = f"cf_trajectories/tr_{branch['branch_id']}.npz"
            np.savez_compressed(in_dir / rel, **trajectory)
            branch["trajectory_kind"] = "counterfactual"
            branch["trajectory_path"] = rel
            branch["trajectory_start_frame"] = 0
            branch["num_frames"] = int(len(trajectory["action"]))
            branch["success"] = True
            branch["final_obj_to_target_dist"] = float(
                rollout_info.get("obj_to_target_dist", float("nan"))
            )
            kept.append(branch_index)
        except (RuntimeError, ValueError) as exc:
            failures.append(
                {
                    "anchor_id": anchor_id,
                    "branch_id": branch["branch_id"],
                    "source_id": int(branch["source_id"]),
                    "target_id": int(branch["target_id"]),
                    "error": str(exc),
                }
            )

    future_chunks = future_chunks[kept]
    obj_ids = obj_ids[kept]
    tgt_ids = tgt_ids[kept]
    cf_flags = cf_flags[kept]
    branches = [branches[index] for index in kept]
    np.savez_compressed(
        in_dir / "cf_anchors" / f"an_{anchor_id}.npz",
        future_chunks=future_chunks,
        objective_ids=obj_ids,
        target_ids=tgt_ids,
        is_counterfactual=cf_flags,
        anchor_proprio=anchor_proprio,
    )
    split = split_for_scene(scene_rank[int(meta["scene_index"])], n_scenes)
    record = {
        "anchor_id": anchor_id,
        "scene_id": meta["scene_id"],
        "scene_index": meta["scene_index"],
        "episode_id": meta["episode_id"],
        "nominal_episode_path": meta["file"],
        "anchor_frame": t,
        "phase": int(phase),
        "phase_name": ("REACH_PICK", "GRASP", "REACH_PLACE", "PLACE")[phase],
        "split": split,
        "horizon": horizon,
        "n_branches": int(len(branches)),
        "cf_path": f"cf_anchors/an_{anchor_id}.npz",
        "branches": branches,
    }
    return {
        "record": record,
        "objective_ids": obj_ids.tolist(),
        "target_ids": tgt_ids.tolist(),
        "cf_flags": cf_flags.tolist(),
        "failures": failures,
    }


def _emit_anchor_worker(payload: tuple) -> dict:
    if _BUILD_WORKER_EPISODES is None or _BUILD_WORKER_ENV is None or _BUILD_WORKER_IN_DIR is None:
        raise RuntimeError("build worker was not initialized")
    return _emit_anchor_core(
        _BUILD_WORKER_ENV,
        _BUILD_WORKER_EPISODES,
        _BUILD_WORKER_IN_DIR,
        *payload,
    )


def build(args: argparse.Namespace) -> Path:
    in_dir: Path = args.in_dir.resolve()
    if not (in_dir / "meta" / "nominal_episodes.jsonl").exists():
        raise FileNotFoundError(f"{in_dir / 'meta' / 'nominal_episodes.jsonl'} missing")
    cf_anchors = in_dir / "cf_anchors"
    cf_trajectories = in_dir / "cf_trajectories"
    existing_outputs = [path for path in (cf_anchors, cf_trajectories) if path.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(f"{existing_outputs[0]} exists; pass --overwrite")
    for path in existing_outputs:
        shutil.rmtree(path)
    cf_anchors.mkdir()
    cf_trajectories.mkdir()

    info = json.loads((in_dir / "meta" / "info.json").read_text())
    info["action_semantics"] = "pd_joint_pos_command_recoverable_from_snapshot_ctrl"
    (in_dir / "meta" / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    height, width, _ = info["image_shape"]
    num_objectives = info["num_objectives"]
    num_targets = info["num_targets"]
    episodes = _load_nominal(in_dir)
    if not episodes:
        raise ValueError("No successful nominal episodes to build")
    scene_indices = sorted({int(ep["meta"]["scene_index"]) for ep in episodes})
    scene_rank = {scene_index: rank for rank, scene_index in enumerate(scene_indices)}
    n_scenes = len(scene_indices)

    pools = _scan_anchors(episodes, args.anchor_stride)
    requested_workers = int(getattr(args, "workers", 1))
    if requested_workers < 1:
        raise ValueError("workers must be positive")
    workers = min(requested_workers, max(1, len(pools[REACH_PICK])))
    executor = None
    env = None
    if workers == 1:
        env = make_env(width, height, source_index=0, robot_init_qpos_noise=0.0)
        valid_rp = _filter_rp(env, episodes, pools[REACH_PICK], num_objectives)
        valid_place = _filter_place(env, episodes, pools[REACH_PLACE], num_targets)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_build_worker,
            initargs=(str(in_dir), width, height),
        )
        rp_payloads = [(REACH_PICK, candidate, num_objectives) for candidate in pools[REACH_PICK]]
        place_payloads = [
            (REACH_PLACE, candidate, num_targets) for candidate in pools[REACH_PLACE]
        ]
        valid_rp = [
            candidate
            for candidate in tqdm(
                executor.map(_filter_candidate_worker, rp_payloads, chunksize=8),
                total=len(rp_payloads),
                desc=f"filter REACH_PICK ({workers} workers)",
                leave=False,
            )
            if candidate is not None
        ]
        valid_place = [
            candidate
            for candidate in tqdm(
                executor.map(_filter_candidate_worker, place_payloads, chunksize=8),
                total=len(place_payloads),
                desc=f"filter REACH_PLACE ({workers} workers)",
                leave=False,
            )
            if candidate is not None
        ]
    phase_counts = {
        "REACH_PICK": len(valid_rp),
        "GRASP": len(pools[GRASP]),
        "REACH_PLACE": len(valid_place),
        "PLACE": len(pools[PLACE]),
    }
    n_rp = min(len(valid_rp), len(valid_place), len(pools[GRASP]), len(pools[PLACE]))
    if args.max_anchors is not None:
        n_rp = min(n_rp, args.max_anchors)

    rng = np.random.default_rng(args.seed)
    sampled = {
        REACH_PICK: [valid_rp[i] for i in rng.permutation(len(valid_rp))[:n_rp]],
        GRASP: [pools[GRASP][i] for i in rng.permutation(len(pools[GRASP]))[:n_rp]],
        REACH_PLACE: [valid_place[i] for i in rng.permutation(len(valid_place))[:n_rp]],
        PLACE: [pools[PLACE][i] for i in rng.permutation(len(pools[PLACE]))[:n_rp]],
    }

    anchor_records: list[dict] = []
    eval_records: list[dict] = []
    counts = {"anchors": 0, "branches": 0, "cf_branches": 0, "nominal_branches": 0}
    samples_per_phase = {p: 0 for p in PHASE_POOLS}
    samples_per_objective = {j: 0 for j in range(num_objectives)}
    samples_per_target = {k: 0 for k in range(num_targets)}
    rejected_rp = len(pools[REACH_PICK]) - len(valid_rp)
    rejected_place = len(pools[REACH_PLACE]) - len(valid_place)
    traj_lengths = [int(ep["meta"]["num_frames"]) for ep in episodes]
    rollout_failures: list[dict] = []

    emit_payloads = [
        (
            phase,
            ep_idx,
            t,
            args.horizon,
            num_objectives,
            num_targets,
            scene_rank,
            n_scenes,
        )
        for phase in PHASE_POOLS
        for ep_idx, t, _obj in sampled[phase]
    ]
    if executor is None:
        emit_results = [
            _emit_anchor_core(env, episodes, in_dir, *payload)
            for payload in tqdm(emit_payloads, desc="build anchors")
        ]
    else:
        try:
            emit_results = list(
                tqdm(
                    executor.map(_emit_anchor_worker, emit_payloads, chunksize=1),
                    total=len(emit_payloads),
                    desc=f"build anchors ({workers} workers)",
                )
            )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    for result in emit_results:
        record = result["record"]
        obj_ids = result["objective_ids"]
        tgt_ids = result["target_ids"]
        cf_flags = result["cf_flags"]
        for failure in result["failures"]:
            rollout_failures.append(failure)
            tqdm.write(
                f"WARNING {failure['branch_id']}: {failure['error']}; dropping this branch only"
            )
        anchor_records.append(record)
        counts["anchors"] += 1
        counts["branches"] += len(record["branches"])
        for j, k, is_cf in zip(obj_ids, tgt_ids, cf_flags):
            counts["cf_branches" if is_cf else "nominal_branches"] += 1
            samples_per_objective[int(j)] += 1
            samples_per_target[int(k)] += 1
        samples_per_phase[record["phase"]] += len(record["branches"])
        # Both REACH_PICK (source-CF) and REACH_PLACE (target-CF) are shared-image
        # groups with differing instructions -- both belong in the eval set.
        if record["split"] == "test" and record["phase"] in (REACH_PICK, REACH_PLACE):
            eval_records.append(record)
    if env is not None:
        env.close()

    (in_dir / "meta" / "anchors.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in anchor_records) + "\n", encoding="utf-8"
    )
    (in_dir / "meta" / "eval_pairs.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_records) + "\n", encoding="utf-8"
    )
    (in_dir / "meta" / "cf_rollout_failures.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rollout_failures)
        + ("\n" if rollout_failures else ""),
        encoding="utf-8",
    )

    stats = {
        "num_nominal_episodes": len(episodes),
        "num_scenes": n_scenes,
        "num_unique_anchors": counts["anchors"],
        "num_branches": counts["branches"],
        "num_cf_branches": counts["cf_branches"],
        "num_nominal_branches": counts["nominal_branches"],
        "samples_per_phase": {("REACH_PICK", "GRASP", "REACH_PLACE", "PLACE")[p]: samples_per_phase[p] for p in PHASE_POOLS},
        "samples_per_objective": {OBJECTIVE_COLORS[j]: samples_per_objective[j] for j in range(num_objectives)},
        "samples_per_target": {TARGET_COLORS[k]: samples_per_target[k] for k in range(num_targets)},
        "nominal_counterfactual_ratio": (
            round(counts["nominal_branches"] / counts["branches"], 4) if counts["branches"] else 0.0
        ),
        "num_valid_cf_groups": len(valid_rp),
        "num_valid_place_groups": len(valid_place),
        "num_rejected_anchors": int(rejected_rp + rejected_place),
        "num_failed_cf_rollouts": len(rollout_failures),
        "n_rp_balanced": int(n_rp),
        "candidate_phase_counts": phase_counts,
        "trajectory_length_distribution": {
            "min": min(traj_lengths) if traj_lengths else 0,
            "max": max(traj_lengths) if traj_lengths else 0,
            "mean": round(float(np.mean(traj_lengths)), 2) if traj_lengths else 0.0,
        },
        "horizon_stored": args.horizon,
        "anchor_stride": args.anchor_stride,
        "build_workers": workers,
    }
    (in_dir / "meta" / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Dataset summary. Each branch now references a full observed trajectory.
    training_meta = {
        "dataset_name": "cf_balanced",
        "data_dir": str(args.in_dir),
        "datalist": [{"sampler": "cf_balanced"}],
        "dataset_root": str(args.in_dir),
        "anchors_file": "meta/anchors.jsonl",
        "horizon_stored": args.horizon,
        "disable_image_augmentation": True,
        "preserve_order": False,
        "num_anchors": counts["anchors"],
        "num_branches": counts["branches"],
        "nominal_counterfactual_ratio": stats["nominal_counterfactual_ratio"],
    }
    (in_dir / "meta" / "cf_balanced.json").write_text(
        json.dumps(training_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"anchors={counts['anchors']} branches={counts['branches']} "
        f"(nominal={counts['nominal_branches']} cf={counts['cf_branches']}) "
        f"ratio_nominal={stats['nominal_counterfactual_ratio']} "
        f"valid_cf_groups={len(valid_rp)} valid_place_groups={len(valid_place)} "
        f"rejected={rejected_rp + rejected_place} n_rp={n_rp}"
    )
    print(f"samples_per_phase={stats['samples_per_phase']}")
    print(f"samples_per_objective={stats['samples_per_objective']}")
    print(f"samples_per_target={stats['samples_per_target']}")
    return in_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_dir", type=Path, default=Path("data/cf_nominal"))
    parser.add_argument("--horizon", type=int, default=32, help="stored future-chunk length (>= training num_actions)")
    parser.add_argument("--anchor-stride", type=int, default=8, help="stride for REACH_PICK anchor candidates")
    parser.add_argument("--max-anchors", type=int, default=None, help="cap N_rp per phase")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=0, help="0 disables the build alarm")
    args = parser.parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    if args.anchor_stride < 1:
        raise ValueError("--anchor-stride must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.timeout_seconds < 0:
        raise ValueError("--timeout-seconds must be nonnegative")
    if args.timeout_seconds:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("build timed out")))
        signal.alarm(args.timeout_seconds)
    print(f"dataset ready: {build(args)}")


if __name__ == "__main__":
    main()
