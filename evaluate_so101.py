"""Run a trained SimVLA checkpoint in the local SO101-Nexus environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from PIL import Image
from so101_nexus.lerobot_dataset import (
    dataset_row_to_sim_qpos,
    sim_qpos_to_dataset_row,
)

from models.utils import load_vla_for_inference, pick_device
from simvla_datasets.utils import build_image_transform
from cf_data.collect import make_env
from cf_data.core import OBJECTIVE_COLORS, objective_instruction
from old_vla_data.counterfactual_collector import _gripper_limits


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--norm_stats", type=Path,
        default=Path("norm_stats/cf_smoke_test_norm.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy_seed", type=int)
    parser.add_argument("--objective_id", type=int, choices=range(5), default=0, help="0: red, 1: blue, 2: green, 3: yellow, 4: purple")
    parser.add_argument("--instruction", type=str)
    parser.add_argument("--execute_steps", type=int, default=5)
    parser.add_argument("--max_replans", type=int, default=400)
    parser.add_argument("--robot_noise", type=float, default=0.0, help="Initial robot pose noise (default 0.0 for exact overfit test)")
    parser.add_argument("--output", type=Path, default=Path("outputs/cf_5obj_eval.mp4"))
    args = parser.parse_args()

    if args.execute_steps < 1 or args.execute_steps > 10:
        raise ValueError("--execute_steps must be in 1..10")

    device = pick_device()
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    model, processor = load_vla_for_inference(args.checkpoint, device)
    if model.action_mode not in ("so101_joint", "so101_delta"):
        raise ValueError(f"Checkpoint action mode is {model.action_mode}, expected SO101")
    model.action_space.load_norm_stats(str(args.norm_stats))
    image_transform = build_image_transform(model.config.image_size, False)

    env = make_env(width=256, height=256, source_index=args.objective_id, robot_init_qpos_noise=args.robot_noise)
    frames: list[np.ndarray] = []
    try:
        obs, info = env.reset(seed=args.seed)
        env.set_objective(args.objective_id)
        instruction = args.instruction or objective_instruction(args.objective_id)
        limits = _gripper_limits(env)
        torch.manual_seed(args.seed if args.policy_seed is None else args.policy_seed)

        frames.append(
            np.concatenate(
                [obs["overhead_camera"], obs["wrist_camera"]], axis=1
            )
        )
        for _ in range(args.max_replans):
            # Keep the flow latent fixed across replans for deterministic closed-loop
            # evaluation. Otherwise every observation gets an unrelated sampled plan.
            if args.policy_seed is not None:
                torch.manual_seed(args.policy_seed)
            images, image_mask = preprocess_images(obs, image_transform)
            state = sim_qpos_to_dataset_row(
                np.asarray(obs["state"], dtype=np.float64),
                gripper_limits_rad=limits,
            )
            language = processor.encode_language([instruction])
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                actions = model.generate_actions(
                    input_ids=language["input_ids"].to(device),
                    language_attention_mask=language["language_attention_mask"].to(device),
                    image_input=images.to(device),
                    image_mask=image_mask.to(device),
                    proprio=torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                    steps=10,
                )[0].float().cpu().numpy()

            done = False
            for action_row in actions[: args.execute_steps]:
                command = dataset_row_to_sim_qpos(
                    action_row, gripper_limits_rad=limits
                )
                command = np.clip(
                    command, env.unwrapped._target_low, env.unwrapped._target_high
                )
                # Reconstruct the skipped 50 Hz midpoint between stored 25 Hz commands.
                current = env.unwrapped.data.ctrl[env.unwrapped._actuator_ids].copy()
                for control_step in range(2):
                    alpha = (control_step + 1) / 2
                    interpolated = current + alpha * (command - current)
                    obs, _, terminated, truncated, info = env.step(interpolated)
                    if terminated or truncated:
                        done = True
                        break
                frames.append(
                    np.concatenate(
                        [obs["overhead_camera"], obs["wrist_camera"]], axis=1
                    )
                )
                if done:
                    break
            if done:
                break
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(args.output, frames, fps=25)
    print(f"instruction: {instruction}")
    print(f"success: {bool(info.get('success', False))}")
    print(f"is_obj_placed: {bool(info.get('is_obj_placed', False))}")
    print(f"is_grasped: {bool(info.get('is_grasped', False))}")
    print(f"obj_to_target_dist: {float(info.get('obj_to_target_dist', np.inf)):.6f}")
    print(f"video: {args.output}")


if __name__ == "__main__":
    main()