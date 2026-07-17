"""Run a trained SimVLA checkpoint in the local SO101-Nexus environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from so101_nexus.lerobot_dataset import (
    dataset_row_to_sim_qpos,
    sim_qpos_to_dataset_row,
)
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor
from vla_data.counterfactual_collector import _gripper_limits, _make_env
from vla_data.language import canonical_instruction


def preprocess_images(obs: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(
                (384, 384), interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )
    images = torch.stack(
        [transform(obs["overhead_camera"]), transform(obs["wrist_camera"])]
    ).unsqueeze(0)
    return images, torch.tensor([[True, True]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm_stats", type=Path, default=Path("norm_stats/so101_norm.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source_index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--target_index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--instruction", type=str)
    parser.add_argument("--execute_steps", type=int, default=5)
    parser.add_argument("--max_replans", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("outputs/so101_demo.mp4"))
    args = parser.parse_args()

    if args.execute_steps < 1 or args.execute_steps > 10:
        raise ValueError("--execute_steps must be in 1..10")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device).eval()
    if model.action_mode != "so101_joint":
        raise ValueError(f"Checkpoint action mode is {model.action_mode}, expected so101_joint")
    model.action_space.load_norm_stats(str(args.norm_stats))
    processor = SmolVLMVLAProcessor.from_pretrained(model.config.smolvlm_model_path)

    source_colors = ("red", "orange")
    target_colors = ("green", "white")
    env = _make_env(
        source_colors,
        target_colors,
        args.source_index,
        args.target_index,
        256,
        256,
    )
    frames: list[np.ndarray] = []
    try:
        obs, info = env.reset(seed=args.seed)
        instruction = args.instruction or canonical_instruction(env)
        limits = _gripper_limits(env)
        torch.manual_seed(args.seed)

        frames.append(
            np.concatenate(
                [obs["overhead_camera"], obs["wrist_camera"]], axis=1
            )
        )
        for _ in range(args.max_replans):
            images, image_mask = preprocess_images(obs)
            state = sim_qpos_to_dataset_row(
                np.asarray(obs["state"], dtype=np.float64),
                gripper_limits_rad=limits,
            )
            language = processor.encode_language([instruction])
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                actions = model.generate_actions(
                    input_ids=language["input_ids"].to(device),
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
    print(f"video: {args.output}")


if __name__ == "__main__":
    main()
