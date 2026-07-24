"""Direct probe: does ckpt-4000 reproduce GT action chunks from exact training inputs?

Isolates "model learned the mapping" from "env rollout failure". For each anchor it
feeds the EXACT training input (stored 96x96 anchor image + anchor_proprio +
branch instruction) into the model and compares the predicted 10-step action chunk
against the GT future_chunk. Then it re-renders the same anchor state at 256x256
(native eval resolution) and repeats, to measure the image-resolution shift.

Reports per-joint arm MAE (deg) and gripper behavior. No env rollout -- pure
feed-forward probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cf_data.collect import make_env
from cf_data.core import Snapshot, get_gripper_limits, restore_snapshot
from models.utils import load_vla_for_inference, pick_device
from simvla_datasets.utils import build_image_transform


def load_anchor(data_dir: Path, anchor: dict):
    nominal = np.load(data_dir / anchor["nominal_episode_path"])
    cf = np.load(data_dir / anchor["cf_path"])
    t = int(anchor["anchor_frame"])
    img96 = [
        Image.fromarray(nominal["observation.images.overhead"][t]),
        Image.fromarray(nominal["observation.images.wrist"][t]),
    ]
    proprio = np.array(cf["anchor_proprio"], dtype=np.float32)
    futures = np.array(cf["future_chunks"], dtype=np.float32)  # [n_branch, 32, 6]
    snap = Snapshot(
        nominal["snapshot.qpos"][t],
        nominal["snapshot.qvel"][t],
        nominal["snapshot.ctrl"][t],
    )
    nominal.close()
    cf.close()
    return img96, proprio, futures, snap


def render256(env, snap, transform):
    restore_snapshot(env, snap)
    obs = env._get_obs()
    img256 = [
        transform(Image.fromarray(obs["overhead_camera"])),
        transform(Image.fromarray(obs["wrist_camera"])),
    ]
    return torch.stack(img256).unsqueeze(0), obs


def predict(model, processor, device, images_tensor, image_mask, proprio, instruction, seed):
    lang = processor.encode_language([instruction])
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = model.generate_actions(
            input_ids=lang["input_ids"].to(device),
            language_attention_mask=lang["language_attention_mask"].to(device),
            image_input=images_tensor.to(device),
            image_mask=image_mask.to(device),
            proprio=torch.as_tensor(proprio, dtype=torch.float32, device=device).unsqueeze(0),
            steps=10,
        )[0].float().cpu().numpy()
    return out  # [10, 6] deg+% (gripper binarized)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("runs/overfit/ckpt-4000"))
    p.add_argument("--norm-stats", type=Path, default=Path("norm_stats/cf_dense_norm.json"))
    p.add_argument("--data-dir", type=Path, default=Path("data/cf_dense"))
    p.add_argument("--anchors", type=int, default=3)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--log", type=Path, default=Path("outputs/probe_overfit.log"))
    args = p.parse_args()

    device = pick_device()
    model, processor = load_vla_for_inference(args.checkpoint, device)
    model.action_space.load_norm_stats(str(args.norm_stats))
    transform = build_image_transform(model.config.image_size, False)
    gripper_mid = float(
        (model.action_space.action_norm_stats.q01[5] + model.action_space.action_norm_stats.q99[5]) / 2
    )
    g_closed = float(model.action_space.action_norm_stats.q01[5])
    g_open = float(model.action_space.action_norm_stats.q99[5])

    anchors = [
        json.loads(line)
        for line in (args.data_dir / "meta" / "anchors.jsonl").read_text().splitlines()
        if line
    ]
    rp = [a for a in anchors if a["phase"] == 0][: args.anchors]

    env = make_env(width=256, height=256, source_index=0, robot_init_qpos_noise=0.0)
    env.reset(seed=0)

    lines = []
    lines.append(f"checkpoint={args.checkpoint} image_size={model.config.image_size}")
    lines.append(f"gripper: closed={g_closed:.2f}% open={g_open:.2f}% midpoint={gripper_mid:.2f}%")
    lines.append(f"anchors={len(rp)} seeds={args.seeds}")

    arm_names = ["sh_pan", "sh_lift", "elbow", "wr_flex", "wr_roll"]
    agg = {"96": [], "256": []}
    diff96_256 = []

    for a in rp:
        img96, proprio, futures, snap = load_anchor(args.data_dir, a)
        img96_t = torch.stack([transform(im) for im in img96]).unsqueeze(0)
        img96_mask = torch.tensor([[True, True]])
        img256_t, _ = render256(env, snap, transform)
        img256_mask = torch.tensor([[True, True]])
        lines.append(f"\n=== {a['anchor_id']} frame={a['anchor_frame']} proprio={np.round(proprio,1).tolist()}")
        for bi, branch in enumerate(a["branches"]):
            gt = futures[bi][:10]  # [10,6] deg+%, continuous gripper
            gt_arm = gt[:, :5]
            gt_grip = gt[:, 5]
            for cond, img_t, img_m in (("96", img96_t, img96_mask), ("256", img256_t, img256_mask)):
                errs = []
                for s in range(args.seeds):
                    pred = predict(model, processor, device, img_t, img_m, proprio, branch["instruction"], s)
                    err = np.abs(pred[:, :5] - gt_arm)  # [10,5]
                    errs.append(pred)
                    agg[cond].append(err.mean(axis=0))  # per-joint mean over chunk
                pred_mean = np.mean(errs, axis=0)  # [10,6] over seeds
                mae = np.abs(pred_mean[:, :5] - gt_arm).mean(axis=0)
                pred_grip = pred_mean[0, 5]
                gt_grip_state = "open" if gt_grip.mean() >= gripper_mid else "closed"
                pred_grip_state = "open" if pred_grip >= gripper_mid else "closed"
                grip_ok = gt_grip_state == pred_grip_state
                tag = "OK " if grip_ok else "MISS"
                lines.append(
                    f"  obj{branch['objective_id']} {cond:3s} armMAE={np.round(mae,2).tolist()} "
                    f"grip pred={pred_grip:.1f}% gt[min={gt_grip.min():.1f},max={gt_grip.max():.1f}] {tag}"
                )
            # 96 vs 256 action divergence (arm)
            p96 = predict(model, processor, device, img96_t, img96_mask, proprio, branch["instruction"], 0)
            p256 = predict(model, processor, device, img256_t, img256_mask, proprio, branch["instruction"], 0)
            d = np.abs(p96[:, :5] - p256[:, :5]).mean()
            diff96_256.append(d)

    env.close()

    lines.append("\n=== SUMMARY ===")
    for cond in ("96", "256"):
        arr = np.array(agg[cond])  # [N, 5]
        lines.append(f"arm MAE {cond} per-joint (deg): {np.round(arr.mean(axis=0),3).tolist()}  mean={arr.mean():.3f}")
    lines.append(f"96-vs-256 arm action divergence (deg): mean={np.mean(diff96_256):.3f} max={np.max(diff96_256):.3f}")

    txt = "\n".join(lines)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(txt)
    print(txt)
    print(f"\nlog -> {args.log}")


if __name__ == "__main__":
    main()