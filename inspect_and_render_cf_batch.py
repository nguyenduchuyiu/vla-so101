"""Inspect cf_data minimal batch and render annotated video + image layout for a small batch.

Reads from data/cf_smoke_test (or a specified cf_data directory), creates a SmolVLM dataloader,
inspects batch contents, and renders an annotated video/image displaying overhead & wrist
camera views alongside instruction text, flow_group_id, proprio, and action deltas.
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

from simvla_datasets.dataset_smolvlm import create_smolvlm_dataloader


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert float tensor [3, H, W] in [0, 1] or normalized range to uint8 RGB [H, W, 3]."""
    img = tensor.detach().cpu().numpy()
    if img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        if img.max() <= 1.0 and img.min() >= 0.0:
            img = (img * 255.0).astype(np.uint8)
        else:
            # Standard ImageNet un-normalization if standard mean/std used
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img * std + mean) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def draw_sample_card(
    overhead_np: np.ndarray,
    wrist_np: np.ndarray,
    sample_idx: int,
    instruction: str,
    flow_group_id: int,
    proprio: np.ndarray,
    action: np.ndarray,
    card_width: int = 640,
    card_height: int = 360,
) -> np.ndarray:
    """Create a rendered visual card for 1 batch item with views + annotations."""
    canvas = np.zeros((card_height, card_width, 3), dtype=np.uint8)
    # Fill background with dark navy/slate header theme
    canvas[:card_height] = (25, 30, 42)

    # Resize camera images
    cam_h, cam_w = 180, 240
    ov_res = cv2.resize(overhead_np, (cam_w, cam_h), interpolation=cv2.INTER_CUBIC)
    wr_res = cv2.resize(wrist_np, (cam_w, cam_h), interpolation=cv2.INTER_CUBIC)

    # Position images side by side
    canvas[70 : 70 + cam_h, 20 : 20 + cam_w] = ov_res
    canvas[70 : 70 + cam_h, 260 : 260 + cam_w] = wr_res

    # Use PIL to draw crisp text
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)

    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 13)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title header
    draw.rectangle([(0, 0), (card_width, 36)], fill=(40, 50, 75))
    draw.text(
        (15, 8),
        f"Batch Item #{sample_idx} | Flow Group ID: {flow_group_id}",
        font=font_title,
        fill=(240, 245, 255),
    )

    # Camera Labels
    draw.text((20, 48), "Overhead Camera", font=font_small, fill=(180, 200, 230))
    draw.text((260, 48), "Wrist Camera", font=font_small, fill=(180, 200, 230))

    # Right panel: Metadata & Action info
    rx = 510
    draw.text((rx - 5, 48), "Details", font=font_small, fill=(180, 200, 230))

    # Instruction box
    draw.rectangle([(15, 260), (card_width - 15, 310)], fill=(35, 42, 58), outline=(60, 75, 105))
    draw.text((25, 265), f"Instruction:", font=font_small, fill=(140, 180, 240))

    # Wrap instruction if needed
    words = instruction.split()
    line1, line2 = "", ""
    for w in words:
        if len(line1 + " " + w) < 65:
            line1 += (" " if line1 else "") + w
        else:
            line2 += (" " if line2 else "") + w
    draw.text((105, 265), line1, font=font_sub, fill=(255, 255, 255))
    if line2:
        draw.text((105, 285), line2, font=font_sub, fill=(255, 255, 255))

    # State & Action readout on right side
    arm_deg = ", ".join(f"{v:.1f}°" for v in proprio[:5])
    grip_pct = f"{proprio[5]:.1f}%"
    act_step0_arm = ", ".join(f"{v:+.1f}" for v in action[0, :5])
    act_step0_grip = f"{action[0, 5]:.1f}%"

    draw.text((rx - 5, 75), "Proprio State:", font=font_small, fill=(140, 180, 240))
    draw.text((rx - 5, 92), f"Grip: {grip_pct}", font=font_sub, fill=(220, 230, 250))

    draw.text((rx - 5, 125), "Action [0] Delta:", font=font_small, fill=(140, 180, 240))
    draw.text((rx - 5, 142), f"Arm Δ: [{act_step0_arm}]", font=font_small, fill=(220, 230, 250))
    draw.text((rx - 5, 160), f"Grip target: {act_step0_grip}", font=font_small, fill=(220, 230, 250))

    # Footer
    draw.text((15, 320), f"Proprio joints: [{arm_deg}]", font=font_small, fill=(160, 175, 195))

    return np.array(pil_img)


def inspect_batch(batch: dict) -> None:
    print("=" * 70)
    print("BATCH INSPECTION SUMMARY")
    print("=" * 70)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  Key: {key:<22} | Type: Tensor {tuple(value.shape)} | dtype: {value.dtype}")
        elif isinstance(value, list):
            print(f"  Key: {key:<22} | Type: List len={len(value)} | sample: {repr(value[0])[:50]}...")
        else:
            print(f"  Key: {key:<22} | Type: {type(value)}")
    print("-" * 70)

    instructions = batch["language_instruction"]
    flow_groups = batch["flow_group_id"].tolist()
    proprios = batch["proprio"].cpu().numpy()
    actions = batch["action"].cpu().numpy()
    batch_size = len(instructions)

    print(f"Batch Size: {batch_size}")
    for i in range(batch_size):
        print(f"\n--- Sample #{i} ---")
        print(f"  Language Instruction : {instructions[i]}")
        print(f"  Flow Group ID        : {flow_groups[i]}")
        print(f"  Proprio (deg/grip%)  : {np.round(proprios[i], 2)}")
        print(f"  Action shape         : {actions[i].shape}")
        print(f"  Action step 0 delta  : {np.round(actions[i][0], 2)}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cf_smoke_test"))
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    meta_path = args.data_dir / "meta" / "cf_balanced.json"
    if not meta_path.exists():
        print(f"Error: {meta_path} does not exist!")
        sys.exit(1)

    print(f"Loading dataloader from {meta_path}...")
    loader = create_smolvlm_dataloader(
        batch_size=args.batch_size,
        metas_path=str(meta_path),
        num_actions=10,
        training=True,
        action_mode="so101_delta",
        num_workers=0,
        image_size=128,
        num_views=2,
    )

    batch = next(iter(loader))
    inspect_batch(batch)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Render visual cards for each sample in batch
    cards = []
    batch_size = len(batch["language_instruction"])
    for i in range(batch_size):
        img_input = batch["image_input"][i]  # [V, C, H, W]
        ov_np = denormalize_image(img_input[0])
        wr_np = denormalize_image(img_input[1])

        instr = batch["language_instruction"][i]
        flow_id = int(batch["flow_group_id"][i])
        proprio = batch["proprio"][i].cpu().numpy()
        action = batch["action"][i].cpu().numpy()

        card = draw_sample_card(
            overhead_np=ov_np,
            wrist_np=wr_np,
            sample_idx=i,
            instruction=instr,
            flow_group_id=flow_id,
            proprio=proprio,
            action=action,
        )
        cards.append(card)

    # 1. Save multi-sample grid image
    grid_img = np.concatenate(cards, axis=0)
    grid_path = args.out_dir / "cf_batch_grid.png"
    Image.fromarray(grid_img).save(grid_path)
    print(f"\nSaved batch inspection image grid to: {grid_path}")

    # 2. Save interactive video (showing batch samples sequentially as video frames)
    # Each sample shown for 2 seconds (50 frames at 25 fps)
    video_frames = []
    for card in cards:
        for _ in range(50):
            video_frames.append(card)

    video_path = args.out_dir / "cf_batch_inspect.mp4"
    media.write_video(video_path, video_frames, fps=25)
    print(f"Saved rendered video to: {video_path}")


if __name__ == "__main__":
    main()
