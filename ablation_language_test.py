"""Language Ablation Test for SimVLA Checkpoint.

Tests whether the model's action predictions actually respond to different language instructions
when given the EXACT same visual observation and initial robot state S0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from so101_nexus.lerobot_dataset import sim_qpos_to_dataset_row

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cf_smoke_test"))
    parser.add_argument("--norm-stats", type=Path, default=Path("norm_stats/cf_smoke_test_norm.json"))
    args = parser.parse_args()

    meta_path = args.data_dir / "meta" / "nominal_episodes.jsonl"
    if not meta_path.exists():
        print(f"Error: {meta_path} does not exist!")
        sys.exit(1)

    metas = [json.loads(line) for line in meta_path.read_text().splitlines() if line]
    target_meta = metas[0]
    gt_npz_path = args.data_dir / target_meta["file"]

    with np.load(gt_npz_path) as npz:
        gt_qpos = npz["snapshot.qpos"][0].copy()
        gt_qvel = npz["snapshot.qvel"][0].copy()
        gt_ctrl = npz["snapshot.ctrl"][0].copy()

    device = pick_device()
    model, processor = load_vla_for_inference(args.checkpoint, device)
    model.action_space.load_norm_stats(str(args.norm_stats))
    image_transform = build_image_transform(model.config.image_size, False)

    # Initialize env and restore exact S0 snapshot
    env = make_env(width=256, height=256, source_index=0, robot_init_qpos_noise=0.0)
    env.reset(seed=0)
    s0 = Snapshot(qpos=gt_qpos, qvel=gt_qvel, ctrl=gt_ctrl)
    restore_snapshot(env, s0)
    obs = env._get_obs()

    images, image_mask = preprocess_images(obs, image_transform)
    limits = _gripper_limits(env)
    state = sim_qpos_to_dataset_row(
        np.asarray(obs["state"], dtype=np.float64),
        gripper_limits_rad=limits,
    )

    test_instructions = [
        ("Obj 0 (Red)", objective_instruction(0)),
        ("Obj 1 (Blue)", objective_instruction(1)),
        ("Obj 2 (Green)", objective_instruction(2)),
        ("Obj 3 (Yellow)", objective_instruction(3)),
        ("Obj 4 (Purple)", objective_instruction(4)),
        ("Ablation: Empty", ""),
        ("Ablation: Nonsense", "do something completely random and jump"),
    ]

    predicted_actions: dict[str, np.ndarray] = {}

    print("=" * 75)
    print("LANGUAGE ABLATION EXPERIMENT: SAME S0 OBS, VARYING INSTRUCTION")
    print("=" * 75)

    for label, instr in test_instructions:
        torch.manual_seed(42)  # Fixed seed for Flow Matching sampling
        language = processor.encode_language([instr])
        with torch.inference_mode():
            actions = model.generate_actions(
                input_ids=language["input_ids"].to(device),
                language_attention_mask=language["language_attention_mask"].to(device),
                image_input=images.to(device),
                image_mask=image_mask.to(device),
                proprio=torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                steps=10,
            )[0].float().cpu().numpy()  # [10, 6]

        predicted_actions[label] = actions
        first_step_arm = np.round(actions[0, :5], 2)
        first_step_grip = round(float(actions[0, 5]), 1)
        print(f"{label:<20} | Instr: {instr[:45]:<45} | First Step Arm: {first_step_arm} | Grip: {first_step_grip}%")

    print("\n" + "=" * 75)
    print("PAIRWISE ACTION DIFFERENCE MATRIX (L2 Norm of predicted action chunks [10, 6])")
    print("=" * 75)

    labels = [t[0] for t in test_instructions]
    N = len(labels)
    diff_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            diff = np.linalg.norm(predicted_actions[labels[i]] - predicted_actions[labels[j]])
            diff_matrix[i, j] = diff

    header = f"{'':<20}" + "".join(f"{f'[{i}]':>10}" for i in range(N))
    print(header)
    for i in range(N):
        row_str = f"[{i}] {labels[i]:<15}" + "".join(f"{diff_matrix[i, j]:10.3f}" for j in range(N))
        print(row_str)

    print("=" * 75)

    # Conclusion check
    avg_cross_diff = np.mean([diff_matrix[i, j] for i in range(5) for j in range(5) if i != j])
    if avg_cross_diff < 1.0:
        print(f"RESULT: AVERAGE INSTRUCTION DIFFERENCE IS EXTREMELY SMALL ({avg_cross_diff:.4f}).")
        print("--> CONFIRMED: THE MODEL AT THIS CHECKPOINT IS INDEED LANGUAGE-BLIND!")
    else:
        print(f"RESULT: AVERAGE INSTRUCTION DIFFERENCE IS ({avg_cross_diff:.4f}).")
        print("--> THE MODEL RESPONDS TO LANGUAGE DIFFERENCES.")
    print("=" * 75)


if __name__ == "__main__":
    main()
