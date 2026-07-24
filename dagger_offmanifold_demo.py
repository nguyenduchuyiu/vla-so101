"""Render model-vs-oracle rollouts from OFF-MANIFOLD states the policy visits.

Story (DAgger): the model is correct on the expert manifold but drifts off it
under its own rollout (probe_onmanifold vs probe_closedloop). This script runs
the model closed-loop from S0 to collect the off-manifold states it actually
visits, then from a few of those states re-rolls (a) the MODEL and (b) the
ORACLE (DAgger expert) side by side, and writes one MP4 per anchor:

    +---------------------+---------------------+
    | MODEL (off-manifold) | ORACLE (DAgger exp) |   overhead
    +---------------------+---------------------+
    | MODEL wrist          | ORACLE wrist        |   wrist
    +---------------------+---------------------+

The model keeps drifting / fails to grasp; the oracle re-plans from the drifted
state and completes pick-and-place -- i.e. the supervision DAgger adds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from so101_nexus.lerobot_dataset import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row

from cf_data.collect import make_env
from cf_data.core import (
    OBJECTIVE_COLORS,
    Snapshot,
    get_gripper_limits,
    objective_instruction,
    qpos_to_row,
    restore_snapshot,
    save_snapshot,
)
from cf_data.env import ENV_ID
from models.utils import load_vla_for_inference, pick_device
from simvla_datasets.utils import build_image_transform
from vla_data.oracle import Oracle


def _preprocess(obs, transform):
    images = torch.stack(
        [transform(Image.fromarray(obs["overhead_camera"])), transform(Image.fromarray(obs["wrist_camera"]))]
    ).unsqueeze(0)
    return images, torch.tensor([[True, True]])


def _model_actions(model, processor, transform, device, limits, obs, instruction):
    images, image_mask = _preprocess(obs, transform)
    state = sim_qpos_to_dataset_row(np.asarray(obs["state"], dtype=np.float64), gripper_limits_rad=limits)
    lang = processor.encode_language([instruction])
    with torch.inference_mode():
        return model.generate_actions(
            input_ids=lang["input_ids"].to(device),
            language_attention_mask=lang["language_attention_mask"].to(device),
            image_input=images.to(device),
            image_mask=image_mask.to(device),
            proprio=torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
            steps=10,
        )[0].float().cpu().numpy()


def _pad(seq, n):
    if len(seq) == 0:
        raise RuntimeError("empty rollout")
    while len(seq) < n:
        seq.append(seq[-1].copy())
    return seq[:n]


def rollout_model(env, model, processor, transform, device, limits, instruction, snap, obj, n_steps, execute_steps):
    restore_snapshot(env, snap)
    env.set_objective(obj)
    obs = env._get_obs()
    oh, wr = [], []
    info: dict = {}
    steps = 0
    done = False
    while steps < n_steps:
        actions = _model_actions(model, processor, transform, device, limits, obs, instruction)
        u = env.unwrapped
        for action_row in actions[:execute_steps]:
            oh.append(obs["overhead_camera"].copy())
            wr.append(obs["wrist_camera"].copy())
            command = np.clip(dataset_row_to_sim_qpos(action_row, gripper_limits_rad=limits), u._target_low, u._target_high)
            current = u.data.ctrl[u._actuator_ids].copy()
            for cs in range(2):
                alpha = (cs + 1) / 2
                obs, _, term, trunc, info = env.step(current + alpha * (command - current))
                if term or trunc:
                    done = True
                    break
            steps += 1
            if done or steps >= n_steps:
                break
        if done:
            break
    return np.stack(_pad(oh, n_steps)), np.stack(_pad(wr, n_steps)), info


def rollout_oracle(env, snap, obj, n_steps):
    restore_snapshot(env, snap)
    env.set_objective(obj)
    oracle = Oracle(ENV_ID, env)
    obs = env._get_obs()
    oh, wr = [], []
    info: dict = {}
    for _ in range(n_steps):
        oh.append(obs["overhead_camera"].copy())
        wr.append(obs["wrist_camera"].copy())
        if oracle.finished:
            continue
        action, _stage = oracle.select_action()
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break
    return np.stack(_pad(oh, n_steps)), np.stack(_pad(wr, n_steps)), info


def _font(size):
    p = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    return ImageFont.truetype(str(p), size) if p.exists() else ImageFont.load_default()


def write_video(out: Path, m_oh, m_wr, o_oh, o_wr, anchor_meta, fps, size):
    S = size
    gap, label_h, title_h = 6, 26, 34
    W = 2 * S + gap
    H = title_h + 2 * (label_h + S) + gap
    font = _font(16)
    font_title = _font(20)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {out}")
    n = m_oh.shape[0]
    for i in range(n):
        canvas = Image.new("RGB", (W, H), (12, 12, 12))
        d = ImageDraw.Draw(canvas)
        d.text((W // 2, 6), anchor_meta, fill="white", font=font_title, anchor="ma")
        panels = [
            (m_oh[i], 0, title_h, "MODEL (off-manifold)", (210, 80, 80)),
            (o_oh[i], S + gap, title_h, "ORACLE (DAgger expert)", (80, 180, 110)),
            (m_wr[i], 0, title_h + label_h + S + gap, "MODEL wrist", (210, 80, 80)),
            (o_wr[i], S + gap, title_h + label_h + S + gap, "ORACLE wrist", (80, 180, 110)),
        ]
        for img, x, y, label, color in panels:
            d.rectangle([x, y, x + S, y + label_h], fill=color)
            d.text((x + S // 2, y + label_h // 2), label, fill="white", font=font, anchor="mm")
            canvas.paste(Image.fromarray(img), (x, y + label_h))
        writer.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  saved {out} ({n} frames, {W}x{H})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("runs/overfit/ckpt-4000"))
    p.add_argument("--norm-stats", type=Path, default=Path("norm_stats/cf_dense_norm.json"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/figs"))
    p.add_argument("--objective-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--anchors", type=int, nargs="*", default=[12, 32, 60],
                   help="control-step indices along the model's S0 rollout to use as off-manifold starts")
    p.add_argument("--rollout-steps", type=int, default=50, help="frames per side-by-side rollout")
    p.add_argument("--execute-steps", type=int, default=5)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--policy-seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    model, processor = load_vla_for_inference(args.checkpoint, device)
    model.action_space.load_norm_stats(str(args.norm_stats))
    transform = build_image_transform(model.config.image_size, False)

    env = make_env(width=args.size, height=args.size, source_index=0, robot_init_qpos_noise=0.0)
    env.reset(seed=args.seed)
    limits = get_gripper_limits(env)
    s0 = save_snapshot(env)
    obj = args.objective_id
    instruction = objective_instruction(obj)

    # 1) run the model closed-loop from S0 and snapshot every control step -> off-manifold states
    restore_snapshot(env, s0)
    env.set_objective(obj)
    obs = env._get_obs()
    snaps: list[Snapshot] = []
    torch.manual_seed(args.policy_seed)
    steps = 0
    max_scan = max(args.anchors) + 1
    while steps < max_scan:
        actions = _model_actions(model, processor, transform, device, limits, obs, instruction)
        u = env.unwrapped
        for action_row in actions[:args.execute_steps]:
            snaps.append(Snapshot(env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl[u._actuator_ids].copy()))
            command = np.clip(dataset_row_to_sim_qpos(action_row, gripper_limits_rad=limits), u._target_low, u._target_high)
            current = u.data.ctrl[u._actuator_ids].copy()
            for cs in range(2):
                alpha = (cs + 1) / 2
                obs, _, term, trunc, _ = env.step(current + alpha * (command - current))
                if term or trunc:
                    break
            steps += 1
            if term or trunc or steps >= max_scan:
                break
        if steps >= max_scan:
            break
    if len(snaps) <= max(args.anchors):
        raise RuntimeError(f"model rollout only reached step {len(snaps)-1}; need up to {max(args.anchors)}")

    # 2) from each off-manifold anchor, roll model vs oracle and write a video
    for a in args.anchors:
        if a >= len(snaps):
            print(f"  anchor {a} out of range (snaps={len(snaps)}), skipping")
            continue
        snap = snaps[a]
        restore_snapshot(env, snap)
        env.set_objective(obj)
        anchor_info = env.unwrapped._get_info() if hasattr(env.unwrapped, "_get_info") else {}
        anchor_info = env.unwrapped._get_info() if hasattr(env.unwrapped, "_get_info") else {}
        m_oh, m_wr, m_info = rollout_model(
            env, model, processor, transform, device, limits, instruction, snap, obj, args.rollout_steps, args.execute_steps
        )
        o_oh, o_wr, o_info = rollout_oracle(env, snap, obj, args.rollout_steps)
        meta = (
            f"off-manifold start @ step {a}  |  obj={OBJECTIVE_COLORS[obj]}  |  "
            f"start: grasp={int(anchor_info.get('is_grasped', 0))} dist={float(anchor_info.get('obj_to_target_dist', -1)):.3f}  |  "
            f"MODEL end: grasp={int(m_info.get('is_grasped', 0))} dist={float(m_info.get('obj_to_target_dist', -1)):.3f}  |  "
            f"ORACLE end: grasp={int(o_info.get('is_grasped', 0))} placed={int(o_info.get('is_obj_placed', 0))} dist={float(o_info.get('obj_to_target_dist', -1)):.3f}"
        )
        out = args.out_dir / f"dagger_offmanifold_obj{obj}_step{a:03d}.mp4"
        write_video(out, m_oh, m_wr, o_oh, o_wr, meta, args.fps, args.size)
        print(meta)

    env.close()
    print(f"\ndone. videos in {args.out_dir}/dagger_offmanifold_obj{obj}_step*.mp4")


if __name__ == "__main__":
    main()