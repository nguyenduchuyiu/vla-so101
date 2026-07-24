"""Closed-loop rollout from the EXACT training S0, tracked against the GT expert trajectory.

The open-loop probe (probe_overfit.py) showed one-shot action prediction is good
(arm MAE ~0.6 deg). This script tests the closed loop: starting from the exact frame-0
snapshot of a nominal episode, execute the model's predicted chunks with the SAME
2-substep interpolation evaluate_so101.py uses, and compare the reached qpos against
the GT expert states frame-by-frame. Locates where (which replan) and how the rollout
diverges from the expert manifold, and whether it reaches grasp/place/success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from so101_nexus.lerobot_dataset import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row

from cf_data.collect import make_env
from cf_data.core import Snapshot, get_gripper_limits, objective_instruction, restore_snapshot
from models.utils import load_vla_for_inference, pick_device
from simvla_datasets.utils import build_image_transform


def preprocess(obs, transform):
    images = torch.stack(
        [transform(Image.fromarray(obs["overhead_camera"])), transform(Image.fromarray(obs["wrist_camera"]))]
    ).unsqueeze(0)
    return images, torch.tensor([[True, True]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("runs/overfit/ckpt-4000"))
    p.add_argument("--norm-stats", type=Path, default=Path("norm_stats/cf_dense_norm.json"))
    p.add_argument("--data-dir", type=Path, default=Path("data/cf_dense"))
    p.add_argument("--objective-id", type=int, default=0)
    p.add_argument("--execute-steps", type=int, default=5)
    p.add_argument("--max-replans", type=int, default=80)
    p.add_argument("--from-home", action="store_true", help="start from env.reset home pose instead of S0")
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--log", type=Path, default=Path("outputs/probe_closedloop.log"))
    args = p.parse_args()

    metas = [json.loads(l) for l in (args.data_dir / "meta" / "nominal_episodes.jsonl").read_text().splitlines() if l]
    meta = next(m for m in metas if m["objective_id"] == args.objective_id)
    with np.load(args.data_dir / meta["file"]) as npz:
        gt_states = npz["observation.state"].copy()  # (N,6) deg+%
        gt_qpos = npz["snapshot.qpos"][0].copy()
        gt_qvel = npz["snapshot.qvel"][0].copy()
        gt_ctrl = npz["snapshot.ctrl"][0].copy()

    device = pick_device()
    model, processor = load_vla_for_inference(args.checkpoint, device)
    model.action_space.load_norm_stats(str(args.norm_stats))
    transform = build_image_transform(model.config.image_size, False)
    g_mid = float((model.action_space.action_norm_stats.q01[5] + model.action_space.action_norm_stats.q99[5]) / 2)

    env = make_env(width=256, height=256, source_index=args.objective_id, robot_init_qpos_noise=0.0)
    env.reset(seed=0)
    env.set_objective(args.objective_id)
    limits = get_gripper_limits(env)
    instruction = objective_instruction(args.objective_id)

    if not args.from_home:
        restore_snapshot(env, Snapshot(gt_qpos, gt_qvel, gt_ctrl))
    obs = env._get_obs()

    lines = []
    lines.append(f"obj={args.objective_id} from_home={args.from_home} policy_seed={args.policy_seed}")
    lines.append(f"S0 proprio (gt frame0)={np.round(gt_states[0],1).tolist()}")

    arm_names = ["sh_pan", "sh_lift", "elbow", "wr_flex", "wr_roll"]
    replan = 0
    gt_frame = 0  # GT frame index aligned to executed env steps (2 env steps == 1 GT frame)
    first_diverge = None
    info = {}
    torch.manual_seed(args.policy_seed)

    for replan in range(args.max_replans):
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
            )[0].float().cpu().numpy()  # (10,6) deg+%, gripper binarized

        gt_future = gt_states[gt_frame + 1 : gt_frame + 1 + 10]  # may be short near end
        if len(gt_future) == 0:
            lines.append(f"replan {replan}: GT trajectory ended at frame {gt_frame}; stopping.")
            break
        n = min(len(gt_future), args.execute_steps)
        pred_arm0 = actions[0, :5]
        gt_arm0 = gt_future[0, :5]
        chunk_mae = np.abs(actions[:n, :5] - gt_future[:n, :5]).mean()
        grip_pred = actions[0, 5]
        grip_gt = gt_future[0, 5]

        # execute
        done = False
        for action_row in actions[: args.execute_steps]:
            command = dataset_row_to_sim_qpos(action_row, gripper_limits_rad=limits)
            command = np.clip(command, env.unwrapped._target_low, env.unwrapped._target_high)
            current = env.unwrapped.data.ctrl[env.unwrapped._actuator_ids].copy()
            for cs in range(2):
                alpha = (cs + 1) / 2
                obs, _, terminated, truncated, info = env.step(current + alpha * (command - current))
                if terminated or truncated:
                    done = True
                    break
            if done:
                break
        gt_frame += args.execute_steps  # 5 chunk entries executed == 5 GT frames (2 env steps each)

        reached = sim_qpos_to_dataset_row(np.asarray(obs["state"], dtype=np.float64), gripper_limits_rad=limits)
        gt_now = gt_states[gt_frame] if gt_frame < len(gt_states) else gt_states[-1]
        track_mae = np.abs(reached[:5] - gt_now[:5]).mean()
        if first_diverge is None and track_mae > 5.0:
            first_diverge = (replan, gt_frame, float(track_mae))

        lines.append(
            f"r{replan:02d} gf{gt_frame:03d} chunkMAE={chunk_mae:5.2f} trackMAE={track_mae:5.2f} "
            f"reached={np.round(reached[:5],1).tolist()} gt={np.round(gt_now[:5],1).tolist()} "
            f"grip pred={grip_pred:4.1f} gt={grip_gt:4.1f}"
        )
        if done:
            lines.append(f"  env terminated/truncated at replan {replan}")
            break

    env.close()
    lines.append(
        f"\nRESULT replans={replan+1} first_diverge={first_diverge} "
        f"success={bool(info.get('success', False))} is_grasped={bool(info.get('is_grasped', False))} "
        f"is_obj_placed={bool(info.get('is_obj_placed', False))} "
        f"obj_to_target_dist={float(info.get('obj_to_target_dist', np.inf)):.4f}"
    )
    txt = "\n".join(lines)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(txt)
    print(txt)
    print(f"\nlog -> {args.log}")


if __name__ == "__main__":
    main()