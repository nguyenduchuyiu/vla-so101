"""Test model closed-loop rollout from the EXACT snapshot S0 used during training data collection.

Compares ground-truth expert trajectory against model predicted rollout starting from the exact same physics snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from so101_nexus.lerobot_dataset import (
    dataset_row_to_sim_qpos,
    sim_qpos_to_dataset_row,
)

from cf_data.collect import make_env
from cf_data.core import OBJECTIVE_COLORS, Snapshot, objective_instruction, restore_snapshot
from models.utils import load_vla_for_inference, pick_device
from old_vla_data.counterfactual_collector import _gripper_limits
from simvla_datasets.utils import build_image_transform


def preprocess_images(
    obs: dict[str, np.ndarray], transform
) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack(
        [
            transform(Image.fromarray(obs["overhead_camera"])),
            transform(Image.fromarray(obs["wrist_camera"])),
        ]
    ).unsqueeze(0)
    return images, torch.tensor([[True, True]])


def draw_frame_comparison(
    gt_ov: np.ndarray,
    gt_wr: np.ndarray,
    pred_ov: np.ndarray,
    pred_wr: np.ndarray,
    step: int,
    instruction: str,
    width: int = 640,
    height: int = 400,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (20, 25, 35)

    cam_w, cam_h = 140, 140
    # Resizing
    gt_ov_r = cv2.resize(gt_ov, (cam_w, cam_h))
    gt_wr_r = cv2.resize(gt_wr, (cam_w, cam_h))
    pred_ov_r = cv2.resize(pred_ov, (cam_w, cam_h))
    pred_wr_r = cv2.resize(pred_wr, (cam_w, cam_h))

    # Left: GT, Right: Model Pred
    canvas[70 : 70 + cam_h, 20 : 20 + cam_w] = gt_ov_r
    canvas[70 : 70 + cam_h, 170 : 170 + cam_w] = gt_wr_r

    canvas[70 : 70 + cam_h, 330 : 330 + cam_w] = pred_ov_r
    canvas[70 : 70 + cam_h, 480 : 480 + cam_w] = pred_wr_r

    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)

    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    draw.rectangle([(0, 0), (width, 36)], fill=(35, 45, 65))
    draw.text((15, 8), f"Step {step} | Exact S0 Train Comparison", font=font_title, fill=(240, 245, 255))

    draw.text((20, 48), "GT Expert (Overhead)", font=font_sub, fill=(100, 220, 140))
    draw.text((170, 48), "GT Expert (Wrist)", font=font_sub, fill=(100, 220, 140))

    draw.text((330, 48), "Model Pred (Overhead)", font=font_sub, fill=(240, 140, 100))
    draw.text((480, 48), "Model Pred (Wrist)", font=font_sub, fill=(240, 140, 100))

    draw.rectangle([(15, 230), (width - 15, 275)], fill=(30, 36, 50), outline=(50, 65, 90))
    draw.text((25, 235), f"Instruction:", font=font_sub, fill=(140, 180, 240))
    draw.text((105, 235), instruction, font=font_sub, fill=(255, 255, 255))

    return np.array(pil_img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cf_smoke_test"))
    parser.add_argument("--norm-stats", type=Path, default=Path("norm_stats/cf_smoke_test_norm.json"))
    parser.add_argument("--objective-id", type=int, default=0)
    parser.add_argument("--execute-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("outputs/cf_exact_train_comparison.mp4"))
    args = parser.parse_args()

    meta_path = args.data_dir / "meta" / "nominal_episodes.jsonl"
    if not meta_path.exists():
        print(f"Error: {meta_path} does not exist!")
        sys.exit(1)

    metas = [json.loads(line) for line in meta_path.read_text().splitlines() if line]
    target_meta = next((m for m in metas if m["objective_id"] == args.objective_id), metas[0])
    gt_npz_path = args.data_dir / target_meta["file"]

    with np.load(gt_npz_path) as npz:
        gt_states = npz["observation.state"].copy()
        gt_ov_imgs = npz["observation.images.overhead"].copy()
        gt_wr_imgs = npz["observation.images.wrist"].copy()
        gt_qpos = npz["snapshot.qpos"][0].copy()
        gt_qvel = npz["snapshot.qvel"][0].copy()
        gt_ctrl = npz["snapshot.ctrl"][0].copy()

    device = pick_device()
    model, processor = load_vla_for_inference(args.checkpoint, device)
    model.action_space.load_norm_stats(str(args.norm_stats))
    image_transform = build_image_transform(model.config.image_size, False)

    # Initialize environment with seed=0
    env = make_env(width=256, height=256, source_index=args.objective_id, robot_init_qpos_noise=0.0)
    obs, info = env.reset(seed=0)
    env.set_objective(args.objective_id)

    # Restore exact physics snapshot S0 from nominal training episode
    s0 = Snapshot(qpos=gt_qpos, qvel=gt_qvel, ctrl=gt_ctrl)
    restore_snapshot(env, s0)
    obs = env._get_obs()

    instruction = objective_instruction(args.objective_id)
    limits = _gripper_limits(env)

    print(f"Running rollout for instruction: {instruction}")

    comparison_frames = []
    num_steps = min(len(gt_ov_imgs), args.max_steps)

    for step in range(num_steps):
        # Current GT images
        gt_ov = gt_ov_imgs[step]
        gt_wr = gt_wr_imgs[step]

        # Current Model Environment images
        pred_ov = obs["overhead_camera"]
        pred_wr = obs["wrist_camera"]

        frame = draw_frame_comparison(gt_ov, gt_wr, pred_ov, pred_wr, step, instruction)
        comparison_frames.append(frame)

        if step % args.execute_steps == 0:
            images, image_mask = preprocess_images(obs, image_transform)
            state = sim_qpos_to_dataset_row(
                np.asarray(obs["state"], dtype=np.float64),
                gripper_limits_rad=limits,
            )
            language = processor.encode_language([instruction])
            with torch.inference_mode():
                actions = model.generate_actions(
                    input_ids=language["input_ids"].to(device),
                    language_attention_mask=language["language_attention_mask"].to(device),
                    image_input=images.to(device),
                    image_mask=image_mask.to(device),
                    proprio=torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                    steps=10,
                )[0].float().cpu().numpy()

        action_row = actions[step % args.execute_steps]
        command = dataset_row_to_sim_qpos(action_row, gripper_limits_rad=limits)
        command = np.clip(command, env.unwrapped._target_low, env.unwrapped._target_high)

        current = env.unwrapped.data.ctrl[env.unwrapped._actuator_ids].copy()
        for control_step in range(2):
            alpha = (control_step + 1) / 2
            interpolated = current + alpha * (command - current)
            obs, _, terminated, truncated, info = env.step(interpolated)
            if terminated or truncated:
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(args.output, comparison_frames, fps=25)
    print(f"Saved exact comparison video to: {args.output}")


if __name__ == "__main__":
    main()
