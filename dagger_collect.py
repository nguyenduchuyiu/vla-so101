"""DAgger rollout + off-manifold threshold detection.

For each scene/objective we run THREE passes from the same S0:

  1. GT oracle trajectory (on-manifold expert): step the oracle from S0, record
     proprio + snapshot per control step -> gt_states[t], gt_snaps[t].
  2. Teacher-forcing (on-manifold model error): at every tf_stride-th GT frame
     restore the GT snapshot, run the model, compare the predicted 10-step chunk
     to the GT future -> on_manifold_chunkMAE[t]. Stays low if the model is correct
     on the expert manifold.
  3. Closed-loop (model's own rollout): run the model from S0 with the 2-substep
     interpolation evaluate_so101.py uses, record proprio + snapshot + images per
     control step -> cl_states[t]. Divergence from GT:
        trackMAE[t] = mean|cl_states[t][:5] - gt_states[t][:5]|  (arm, degrees)
     grows when the policy drifts off the expert manifold.

Off-manifold threshold t* = first control step where trackMAE[t] exceeds
--track-mae-threshold for --persist-steps consecutive steps (or the first step
after the GT trajectory ends, since the expert never visits those states). Frames
[t*:] are the off-manifold states DAgger must supervise; frames before t* are
on-manifold (already covered by the expert dataset) and are dropped. The saved
episode (collect.py schema) contains only [t*:] so build.py --dagger rolls a
fresh oracle exactly where the policy drifts. A per-episode probe record
(teacher-forcing vs closed-loop curves + t*) is written to meta/probe.jsonl.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from so101_nexus.lerobot_dataset import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row

from cf_data.collect import make_env
from cf_data.core import (
    OBJECTIVE_COLORS,
    REACH_PICK,
    Snapshot,
    get_gripper_limits,
    objective_instruction,
    qpos_to_row,
    restore_snapshot,
    save_snapshot,
    stage_to_phase,
)
from cf_data.env import ENV_ID
from models.utils import load_vla_for_inference, pick_device
from simvla_datasets.utils import build_image_transform
from vla_data.oracle import Oracle

CONTROL_DT = 0.02


def preprocess(obs, transform):
    images = torch.stack(
        [transform(Image.fromarray(obs["overhead_camera"])), transform(Image.fromarray(obs["wrist_camera"]))]
    ).unsqueeze(0)
    return images, torch.tensor([[True, True]])


def _row(env) -> np.ndarray:
    return qpos_to_row(env._get_current_qpos())


def _oracle_gt(env, max_steps: int) -> tuple[list[np.ndarray], list[Snapshot]]:
    """Oracle rollout from the current state -> per-control-step proprio + snapshots."""
    oracle = Oracle(ENV_ID, env)
    states: list[np.ndarray] = []
    snaps: list[Snapshot] = []
    steps = 0
    u = env.unwrapped
    while not oracle.finished and steps < max_steps:
        states.append(_row(env).copy())
        snaps.append(Snapshot(env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl[u._actuator_ids].copy()))
        action, _stage = oracle.select_action()
        env.step(action)
        steps += 1
    return states, snaps


def _teacher_forcing(env, model, processor, transform, device, limits, instruction,
                     gt_states, gt_snaps, stride: int) -> tuple[list[int], list[float]]:
    """On-manifold chunkMAE: model on GT states vs GT future, at every `stride`-th frame."""
    frames, maes = [], []
    lang = processor.encode_language([instruction])
    image_mask = torch.tensor([[True, True]])
    for t in range(0, len(gt_states) - 1, stride):
        restore_snapshot(env, gt_snaps[t])
        obs = env._get_obs()
        images = torch.stack(
            [transform(Image.fromarray(obs["overhead_camera"])), transform(Image.fromarray(obs["wrist_camera"]))]
        ).unsqueeze(0)
        gt_future = np.stack(gt_states[t + 1 : t + 11])  # up to 10
        with torch.inference_mode():
            pred = model.generate_actions(
                input_ids=lang["input_ids"].to(device),
                language_attention_mask=lang["language_attention_mask"].to(device),
                image_input=images.to(device),
                image_mask=image_mask.to(device),
                proprio=torch.as_tensor(gt_states[t], dtype=torch.float32, device=device).unsqueeze(0),
                steps=10,
            )[0].float().cpu().numpy()
        n = min(len(gt_future), 10)
        mae = float(np.abs(pred[:n, :5] - gt_future[:n, :5]).mean())
        frames.append(t)
        maes.append(mae)
    return frames, maes


def _closed_loop(env, model, processor, transform, device, limits, instruction,
                 execute_steps: int, max_replans: int, policy_seed: int) -> tuple[list[dict], dict]:
    """Model closed-loop from the current state; record every control step."""
    obs = env._get_obs()
    frames: list[dict] = []
    info: dict = {}
    done = False
    torch.manual_seed(policy_seed)
    u = env.unwrapped
    for _ in range(max_replans):
        try:
            oracle = Oracle(ENV_ID, env)
            stage_name = oracle.stages[0].name if oracle.stages else "finished"
            phase = stage_to_phase(stage_name)
        except RuntimeError:
            phase, stage_name = REACH_PICK, "unreachable"
        images, image_mask = preprocess(obs, transform)
        state = sim_qpos_to_dataset_row(np.asarray(obs["state"], dtype=np.float64), gripper_limits_rad=limits)
        lang = processor.encode_language([instruction])
        with torch.inference_mode():
            actions = model.generate_actions(
                input_ids=lang["input_ids"].to(device),
                language_attention_mask=lang["language_attention_mask"].to(device),
                image_input=images.to(device),
                image_mask=image_mask.to(device),
                proprio=torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                steps=10,
            )[0].float().cpu().numpy()
        for action_row in actions[:execute_steps]:
            frames.append(
                {
                    "overhead": obs["overhead_camera"].copy(),
                    "wrist": obs["wrist_camera"].copy(),
                    "proprio": _row(env),
                    "phase": phase,
                    "stage": stage_name,
                    "qpos": env.data.qpos.copy(),
                    "qvel": env.data.qvel.copy(),
                    "ctrl": env.data.ctrl[u._actuator_ids].copy(),
                }
            )
            command = np.clip(dataset_row_to_sim_qpos(action_row, gripper_limits_rad=limits), u._target_low, u._target_high)
            current = u.data.ctrl[u._actuator_ids].copy()
            for cs in range(2):
                alpha = (cs + 1) / 2
                obs, _, terminated, truncated, info = env.step(current + alpha * (command - current))
                if terminated or truncated:
                    done = True
                    break
            if done:
                break
        if done:
            break
    return frames, info


def _find_threshold(track_mae: np.ndarray, threshold: float, persist: int) -> int | None:
    """First index where track_mae exceeds `threshold` for `persist` consecutive steps."""
    n = len(track_mae)
    run = 0
    for t in range(n):
        if track_mae[t] > threshold:
            run += 1
            if run >= persist:
                return t - persist + 1
        else:
            run = 0
    return None


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

    device = pick_device()
    model, processor = load_vla_for_inference(args.checkpoint, device)
    model.action_space.load_norm_stats(str(args.norm_stats))
    transform = build_image_transform(model.config.image_size, False)

    env = make_env(width=args.width, height=args.height, source_index=0, robot_init_qpos_noise=0.0)
    limits = get_gripper_limits(env)
    max_steps = args.max_replans * args.execute_steps  # bound both GT and closed-loop

    records, probes = [], []
    episode_index = 0
    for scene_index in range(args.scenes):
        seed = args.seed + scene_index
        scene_id = f"scene_{seed:06d}"
        env.reset(seed=seed)
        s0 = save_snapshot(env)
        for objective_id in range(len(OBJECTIVE_COLORS)):
            ep_id = f"{scene_id}_obj{objective_id}"
            instruction = objective_instruction(objective_id)

            # 1. GT oracle trajectory from S0
            restore_snapshot(env, s0)
            env.set_objective(objective_id)
            try:
                gt_states, gt_snaps = _oracle_gt(env, max_steps)
            except RuntimeError as exc:
                print(f"  {ep_id}: oracle GT failed ({exc}), skipping")
                continue
            if len(gt_states) < 8:
                print(f"  {ep_id}: GT too short ({len(gt_states)}), skipping")
                continue

            # 2. teacher-forcing (on-manifold model error)
            tf_frames, tf_maes = _teacher_forcing(
                env, model, processor, transform, device, limits, instruction,
                gt_states, gt_snaps, args.tf_stride,
            )

            # 3. closed-loop (model rollout)
            restore_snapshot(env, s0)
            env.set_objective(objective_id)
            cl_frames, info = _closed_loop(
                env, model, processor, transform, device, limits, instruction,
                args.execute_steps, args.max_replans,
                args.policy_seed + scene_index * 100 + objective_id,
            )
            if not cl_frames:
                print(f"  {ep_id}: closed-loop produced 0 frames, skipping")
                continue

            # 4. trackMAE + off-manifold threshold t*
            n = min(len(cl_frames), len(gt_states))
            cl_proprio = np.stack([f["proprio"] for f in cl_frames])[:n]
            gt_proprio = np.stack(gt_states)[:n]
            track_mae = np.abs(cl_proprio[:, :5] - gt_proprio[:, :5]).mean(axis=1)
            t_star = _find_threshold(track_mae, args.track_mae_threshold, args.persist_steps)
            # model outlasted the expert -> those states are off-manifold by definition
            if t_star is None and len(cl_frames) > len(gt_states):
                t_star = len(gt_states)

            if t_star is None:
                print(f"  {ep_id}: on-manifold throughout (max trackMAE={track_mae.max():.2f}), no DAgger data")
                probes.append(_probe_record(ep_id, objective_id, None, cl_frames, track_mae, tf_frames, tf_maes, info))
                continue
            off = cl_frames[t_star:]
            if len(off) < args.min_offmanifold_frames:
                print(f"  {ep_id}: t*={t_star} but only {len(off)} off-manifold frames, skipping")
                probes.append(_probe_record(ep_id, objective_id, t_star, cl_frames, track_mae, tf_frames, tf_maes, info))
                continue

            rel = _save_episode(out, episode_index, off)
            records.append(
                {
                    "scene_id": scene_id,
                    "scene_index": scene_index,
                    "initial_state_id": f"init_{seed:06d}",
                    "episode_id": ep_id,
                    "objective_id": objective_id,
                    "objective_color": OBJECTIVE_COLORS[objective_id],
                    "instruction": instruction,
                    "is_counterfactual": False,
                    "dagger": True,
                    "t_star": int(t_star),
                    "num_offmanifold_frames": len(off),
                    "success": bool(info.get("success", False)),
                    "is_grasped": bool(info.get("is_grasped", False)),
                    "num_frames": len(off),
                    "final_obj_to_target_dist": float(info.get("obj_to_target_dist", float("nan"))),
                    "episode_index": episode_index,
                    "file": rel,
                }
            )
            episode_index += 1
            probes.append(_probe_record(ep_id, objective_id, t_star, cl_frames, track_mae, tf_frames, tf_maes, info))
            print(
                f"  {ep_id}: t*={t_star} off-manifold={len(off)} frames "
                f"(on-manifold chunkMAE={np.mean(tf_maes):.2f}deg, trackMAE@t*={track_mae[t_star] if t_star < n else float('nan'):.2f}deg)"
            )
    env.close()

    with (out / "meta" / "nominal_episodes.jsonl").open("w", encoding="utf-8") as h:
        for r in records:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "meta" / "probe.jsonl").open("w", encoding="utf-8") as h:
        for r in probes:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    info_json = {
        "format": "cf_nominal_v1",
        "dataset_kind": "dagger_offmanifold",
        "num_objectives": len(OBJECTIVE_COLORS),
        "objective_colors": list(OBJECTIVE_COLORS),
        "target_color": "white",
        "num_scenes_requested": args.scenes,
        "total_episodes": len(records),
        "total_offmanifold_frames": sum(r["num_offmanifold_frames"] for r in records),
        "image_shape": [args.height, args.width, 3],
        "control_dt": CONTROL_DT,
        "track_mae_threshold_deg": args.track_mae_threshold,
        "persist_steps": args.persist_steps,
        "checkpoint": str(args.checkpoint),
    }
    (out / "meta" / "info.json").write_text(json.dumps(info_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {len(records)} DAgger episodes ({info_json['total_offmanifold_frames']} off-manifold frames) -> {out}")
    return out


def _probe_record(ep_id, objective_id, t_star, cl_frames, track_mae, tf_frames, tf_maes, info) -> dict:
    return {
        "episode_id": ep_id,
        "objective_id": objective_id,
        "t_star": None if t_star is None else int(t_star),
        "num_closed_loop_frames": len(cl_frames),
        "on_manifold_chunkMAE_mean": float(np.mean(tf_maes)) if tf_maes else float("nan"),
        "on_manifold_chunkMAE_max": float(np.max(tf_maes)) if tf_maes else float("nan"),
        "closed_loop_trackMAE_max": float(track_mae.max()) if len(track_mae) else float("nan"),
        "closed_loop_trackMAE_curve": [round(float(x), 3) for x in track_mae],
        "tf_frames": tf_frames,
        "tf_chunkMAE_curve": [round(float(x), 3) for x in tf_maes],
        "success": bool(info.get("success", False)),
        "is_grasped": bool(info.get("is_grasped", False)),
        "final_obj_to_target_dist": float(info.get("obj_to_target_dist", float("nan"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute-steps", type=int, default=5)
    parser.add_argument("--max-replans", type=int, default=80)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--track-mae-threshold", type=float, default=5.0, help="closed-loop arm divergence (deg) marking off-manifold")
    parser.add_argument("--persist-steps", type=int, default=2, help="consecutive steps above threshold to confirm t*")
    parser.add_argument("--min-offmanifold-frames", type=int, default=4, help="skip episode if fewer off-manifold frames than this")
    parser.add_argument("--tf-stride", type=int, default=5, help="teacher-forcing chunkMAE cadence")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.scenes <= 0:
        raise ValueError("--scenes must be positive")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("dagger collect timed out")))
    signal.alarm(max(60, 300 * args.scenes))
    print(f"dataset ready: {collect(args)}")


if __name__ == "__main__":
    main()