"""Stage B-E: build the balanced counterfactual dataset from nominal episodes.

Reads the Stage-A nominal dataset (episodes/*.npz + meta/nominal_episodes.jsonl) and
emits, into the same directory:

  * cf_anchors/an_<anchor_id>.npz  -- per anchor, future proprio chunks for every
    branch (5 for a REACH_PICK counterfactual group, 1 for a nominal-only anchor).
  * meta/anchors.jsonl            -- one line per anchor (output C, the balanced
    training set; each line lists its branches).
  * meta/eval_pairs.jsonl          -- REACH_PICK counterfactual groups in the test
    split (output D, 5 branches sharing images/proprio, differing instruction).
  * meta/stats.json               -- dataset statistics report (output E).

A REACH_PICK counterfactual group keeps the scene/robot/object state, the anchor
image and the anchor proprio from the nominal episode, swaps the objective to each
of the other 4 objectives, and rolls a fresh oracle from the restored anchor
snapshot (render-free) to obtain that objective's future proprio chunk. An anchor is
kept only if all 4 counterfactual oracles plan successfully. The nominal branch of a
group reuses the nominal episode's own continuation (frames t+1..t+H), so it is exact
and free. Nominal-only anchors (GRASP/REACH_PLACE/PLACE) carry a single nominal branch.

Balance: we pick N_rp anchors from each phase, where N_rp is the size of the smallest
phase pool among {valid REACH_PICK groups, GRASP, REACH_PLACE, PLACE}. Each REACH_PICK
group contributes 1 nominal + 4 counterfactual branches, so nominal = 4*N_rp and
counterfactual = 4*N_rp -> 50/50, with the 4 nominal phases each at N_rp (12.5%).
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
from pathlib import Path

import numpy as np

from cf_data.collect import make_env
from cf_data.core import (
    GRASP,
    OBJECTIVE_COLORS,
    PLACE,
    REACH_PICK,
    REACH_PLACE,
    Snapshot,
    objective_instruction,
    qpos_to_row,
    restore_snapshot,
    split_for_scene,
    stage_to_phase,
    step_physics,
)
from cf_data.env import ENV_ID
from old_vla_data.oracle import Oracle

PHASE_POOLS = (REACH_PICK, GRASP, REACH_PLACE, PLACE)


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
    """Future proprio chunk [horizon, D] = nominal state[t+1 : t+1+H], repeat-last padded.

    If the anchor is at the episode tail with no remaining future, hold the anchor
    proprio so the chunk is well-formed (callers filter such anchors out, this is a
    defensive backstop).
    """
    future = ep["state"][t + 1 : t + 1 + horizon]
    if len(future) == 0:
        future = np.repeat(ep["state"][t : t + 1], horizon, axis=0)
    elif len(future) < horizon:
        future = np.concatenate([future, np.repeat(future[-1:], horizon - len(future), axis=0)])
    return future.astype(np.float32)


def _cf_valid(env, snap: Snapshot, objective_id: int, anchor_objective: int) -> bool:
    """A counterfactual is valid iff the oracle can plan (IK) for objective_id from the anchor state."""
    if objective_id == anchor_objective:
        return True  # nominal branch uses the stored continuation, no oracle needed
    restore_snapshot(env, snap)
    env.set_objective(objective_id)
    try:
        Oracle(ENV_ID, env)
    except RuntimeError:
        return False
    return True


def _roll_cf(env, snap: Snapshot, objective_id: int, horizon: int) -> np.ndarray:
    """Roll a fresh oracle for objective_id from the anchor state; render-free, repeat-last padded."""
    restore_snapshot(env, snap)
    env.set_objective(objective_id)
    oracle = Oracle(ENV_ID, env)
    chunk: list[np.ndarray] = []
    while len(chunk) < horizon:
        if oracle.finished:
            break
        action, _ = oracle.select_action()
        step_physics(env, action)
        chunk.append(qpos_to_row(env._get_current_qpos()))
    if not chunk:  # oracle already finished at the anchor (degenerate); hold the anchor
        chunk.append(qpos_to_row(env._get_current_qpos()))
    while len(chunk) < horizon:
        chunk.append(chunk[-1].copy())
    return np.stack(chunk).astype(np.float32)


def _scan_anchors(episodes: list[dict], anchor_stride: int) -> dict[int, list[tuple[int, int, int]]]:
    """Per-phase anchor candidate pools. REACH_PICK candidates are strided (CF filtering is costly).

    Returns {phase: [(episode_idx, frame, objective_id), ...]}.
    """
    pools: dict[int, list[tuple[int, int, int]]] = {p: [] for p in PHASE_POOLS}
    for ep_idx, ep in enumerate(episodes):
        phases = ep["phase"]
        obj_id = ep["meta"]["objective_id"]
        n = len(phases)
        last_usable = n - 2  # need at least one real future frame after the anchor
        rp_indices = np.where(phases == REACH_PICK)[0]
        rp_indices = rp_indices[rp_indices <= last_usable]
        # Stride REACH_PICK candidates; keep the first RP frame of the episode unconditionally.
        rp_pick = rp_indices[::anchor_stride]
        if len(rp_indices) and rp_pick[0] != rp_indices[0]:
            rp_pick = np.r_[rp_indices[0], rp_pick]
        for t in rp_pick:
            pools[REACH_PICK].append((ep_idx, int(t), obj_id))
        for p in (GRASP, REACH_PLACE, PLACE):
            for t in np.where(phases == p)[0]:
                if t <= last_usable:
                    pools[p].append((ep_idx, int(t), obj_id))
    return pools


def _filter_rp(
    env, episodes: list[dict], rp_candidates: list[tuple[int, int, int]], num_objectives: int
) -> list[tuple[int, int, int]]:
    """Keep REACH_PICK candidates whose 4 non-anchor objectives all plan successfully."""
    valid: list[tuple[int, int, int]] = []
    for ep_idx, t, obj_id in rp_candidates:
        ep = episodes[ep_idx]
        snap = Snapshot(ep["snap_qpos"][t], ep["snap_qvel"][t], ep["snap_ctrl"][t])
        if all(_cf_valid(env, snap, j, obj_id) for j in range(num_objectives)):
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """REACH_PICK counterfactual group: 1 nominal branch (stored continuation) + 4 CF branches."""
    ep = episodes[ep_idx]
    obj_id = ep["meta"]["objective_id"]
    snap = Snapshot(ep["snap_qpos"][t], ep["snap_qvel"][t], ep["snap_ctrl"][t])
    futures: list[np.ndarray] = []
    obj_ids: list[int] = []
    cf_flags: list[bool] = []
    branches: list[dict] = []
    anchor_id = _anchor_id(ep["meta"]["episode_id"], t)
    for j in range(num_objectives):
        is_cf = j != obj_id
        future = _roll_cf(env, snap, j, horizon) if is_cf else _future_from_nominal(ep, t, horizon)
        futures.append(future)
        obj_ids.append(j)
        cf_flags.append(is_cf)
        branches.append(
            {
                "branch_id": f"{anchor_id}_obj{j}",
                "objective_id": j,
                "instruction": objective_instruction(j),
                "is_counterfactual": is_cf,
            }
        )
    return (
        np.stack(futures).astype(np.float32),
        np.asarray(obj_ids, dtype=np.int8),
        np.asarray(cf_flags, dtype=bool),
        branches,
    )


def _build_nominal_branch(
    episodes: list[dict],
    ep_idx: int,
    t: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Nominal-only anchor (GRASP/REACH_PLACE/PLACE): a single nominal branch."""
    ep = episodes[ep_idx]
    obj_id = ep["meta"]["objective_id"]
    anchor_id = _anchor_id(ep["meta"]["episode_id"], t)
    future = _future_from_nominal(ep, t, horizon)
    branches = [
        {
            "branch_id": f"{anchor_id}_obj{obj_id}",
            "objective_id": int(obj_id),
            "instruction": objective_instruction(int(obj_id)),
            "is_counterfactual": False,
        }
    ]
    return (
        future[None].astype(np.float32),
        np.asarray([obj_id], dtype=np.int8),
        np.asarray([False], dtype=bool),
        branches,
    )


def build(args: argparse.Namespace) -> Path:
    in_dir: Path = args.in_dir.resolve()
    if not (in_dir / "meta" / "nominal_episodes.jsonl").exists():
        raise FileNotFoundError(f"{in_dir / 'meta' / 'nominal_episodes.jsonl'} missing")
    cf_anchors = in_dir / "cf_anchors"
    if cf_anchors.exists():
        if not args.overwrite:
            raise FileExistsError(f"{cf_anchors} exists; pass --overwrite")
        shutil.rmtree(cf_anchors)
    cf_anchors.mkdir()

    info = json.loads((in_dir / "meta" / "info.json").read_text())
    height, width, _ = info["image_shape"]
    num_objectives = info["num_objectives"]
    episodes = _load_nominal(in_dir)
    n_scenes = info["num_scenes_saved"]

    env = make_env(width, height, source_index=0, robot_init_qpos_noise=0.0)

    pools = _scan_anchors(episodes, args.anchor_stride)
    valid_rp = _filter_rp(env, episodes, pools[REACH_PICK], num_objectives)
    phase_counts = {
        "REACH_PICK": len(valid_rp),
        "GRASP": len(pools[GRASP]),
        "REACH_PLACE": len(pools[REACH_PLACE]),
        "PLACE": len(pools[PLACE]),
    }
    n_rp = min(len(valid_rp), len(pools[GRASP]), len(pools[REACH_PLACE]), len(pools[PLACE]))
    if args.max_anchors is not None:
        n_rp = min(n_rp, args.max_anchors)

    rng = np.random.default_rng(args.seed)
    sampled = {
        REACH_PICK: [valid_rp[i] for i in rng.permutation(len(valid_rp))[:n_rp]],
        GRASP: [pools[GRASP][i] for i in rng.permutation(len(pools[GRASP]))[:n_rp]],
        REACH_PLACE: [pools[REACH_PLACE][i] for i in rng.permutation(len(pools[REACH_PLACE]))[:n_rp]],
        PLACE: [pools[PLACE][i] for i in rng.permutation(len(pools[PLACE]))[:n_rp]],
    }

    anchor_records: list[dict] = []
    eval_records: list[dict] = []
    counts = {"anchors": 0, "branches": 0, "cf_branches": 0, "nominal_branches": 0}
    samples_per_phase = {p: 0 for p in PHASE_POOLS}
    samples_per_objective = {j: 0 for j in range(num_objectives)}
    rejected = len(pools[REACH_PICK]) - len(valid_rp)
    traj_lengths = [int(ep["meta"]["num_frames"]) for ep in episodes]

    def _emit(phase: int, ep_idx: int, t: int) -> None:
        ep = episodes[ep_idx]
        meta = ep["meta"]
        anchor_id = _anchor_id(meta["episode_id"], t)
        if phase == REACH_PICK:
            future_chunks, obj_ids, cf_flags, branches = _build_rp_group(
                env, episodes, ep_idx, t, args.horizon, num_objectives
            )
        else:
            future_chunks, obj_ids, cf_flags, branches = _build_nominal_branch(episodes, ep_idx, t, args.horizon)
        anchor_proprio = ep["state"][t].astype(np.float32)
        np.savez_compressed(
            cf_anchors / f"an_{anchor_id}.npz",
            future_chunks=future_chunks,
            objective_ids=obj_ids,
            is_counterfactual=cf_flags,
            anchor_proprio=anchor_proprio,
        )
        split = split_for_scene(meta["scene_index"], n_scenes)
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
            "horizon": args.horizon,
            "n_branches": int(len(branches)),
            "cf_path": f"cf_anchors/an_{anchor_id}.npz",
            "branches": branches,
        }
        anchor_records.append(record)
        counts["anchors"] += 1
        counts["branches"] += len(branches)
        for j, is_cf in zip(obj_ids, cf_flags):
            counts["cf_branches" if is_cf else "nominal_branches"] += 1
            samples_per_objective[int(j)] += 1
        samples_per_phase[phase] += len(branches)
        if split == "test" and phase == REACH_PICK:
            eval_records.append(record)

    for phase in PHASE_POOLS:
        for ep_idx, t, _obj in sampled[phase]:
            _emit(phase, ep_idx, t)

    env.close()

    (in_dir / "meta" / "anchors.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in anchor_records) + "\n", encoding="utf-8"
    )
    (in_dir / "meta" / "eval_pairs.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_records) + "\n", encoding="utf-8"
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
        "nominal_counterfactual_ratio": (
            round(counts["nominal_branches"] / counts["branches"], 4) if counts["branches"] else 0.0
        ),
        "num_valid_cf_groups": len(valid_rp),
        "num_rejected_anchors": int(rejected),
        "n_rp_balanced": int(n_rp),
        "candidate_phase_counts": phase_counts,
        "trajectory_length_distribution": {
            "min": min(traj_lengths) if traj_lengths else 0,
            "max": max(traj_lengths) if traj_lengths else 0,
            "mean": round(float(np.mean(traj_lengths)), 2) if traj_lengths else 0.0,
        },
        "horizon_stored": args.horizon,
        "anchor_stride": args.anchor_stride,
    }
    (in_dir / "meta" / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Training meta consumed by simvla_datasets.dataset_smolvlm + the cf_balanced
    # handler. Point the training --dataset at this file. Image augmentation is
    # disabled so the shared anchor image stays identical across the branches of a
    # group (criterion 6); the handler only needs num_workers=0 (enforced in the
    # dataloader), so batch_size is unconstrained.
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
        f"valid_cf_groups={len(valid_rp)} rejected={rejected} n_rp={n_rp}"
    )
    print(f"samples_per_phase={stats['samples_per_phase']}")
    print(f"samples_per_objective={stats['samples_per_objective']}")
    return in_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_dir", type=Path, default=Path("data/cf_nominal"))
    parser.add_argument("--horizon", type=int, default=32, help="stored future-chunk length (>= training num_actions)")
    parser.add_argument("--anchor-stride", type=int, default=8, help="stride for REACH_PICK anchor candidates")
    parser.add_argument("--max-anchors", type=int, default=None, help="cap N_rp per phase")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    if args.anchor_stride < 1:
        raise ValueError("--anchor-stride must be positive")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("build timed out")))
    signal.alarm(2400)
    print(f"dataset ready: {build(args)}")


if __name__ == "__main__":
    main()