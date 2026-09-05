"""Run a saved LeRobot SmolVLA checkpoint in the project's SO101 simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from so101_nexus.lerobot_dataset import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row

from cf_data.collect import CONTROL_DT, make_env
from cf_data.core import get_gripper_limits, instruction
from .data import CAMERAS, JOINT_NAMES, from_delta_joint


def load_checkpoint(path: Path, device: str):
    contract = json.loads((path / "so101_contract.json").read_text())
    if contract["joint_order"] != JOINT_NAMES or contract["camera_mapping"] != CAMERAS:
        raise ValueError("Checkpoint does not match the SO101 camera/joint contract")
    if contract["action_semantics"] != "arm_delta_from_chunk_anchor_gripper_absolute":
        raise ValueError("Expected delta-joint SO101 policy")
    config = SmolVLAConfig.from_pretrained(path)
    config.device = device
    policy = SmolVLAPolicy.from_pretrained(path, config=config, strict=True).to(device).eval()
    pre = PolicyProcessorPipeline.from_pretrained(
        path, config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": device}},
    )
    post = PolicyProcessorPipeline.from_pretrained(
        path, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
    )
    return policy, pre, post, contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--source", type=int, choices=range(5), default=0)
    parser.add_argument("--target", type=int, choices=range(3), default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-replans", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("outputs/smolvla_cf.mp4"))
    args = parser.parse_args()
    if args.max_replans < 1:
        raise ValueError("max-replans must be positive")
    policy, pre, post, contract = load_checkpoint(args.checkpoint, args.device)
    if not np.isclose(1 / contract["fps"], CONTROL_DT):
        raise ValueError("Dataset and simulator control rates differ")
    shape = policy.config.input_features["observation.images.camera1"].shape
    env = make_env(width=shape[2], height=shape[1], source_index=args.source, robot_init_qpos_noise=0.0)
    frames = []
    try:
        if env.unwrapped.control_mode != "pd_joint_pos":
            raise ValueError("This delta-to-target adapter requires pd_joint_pos control")
        obs, info = env.reset(seed=args.seed)
        env.unwrapped.set_objective(args.source, args.target)
        obs = env.unwrapped._get_obs()
        limits = get_gripper_limits(env)
        task = instruction(args.source, args.target)
        policy.reset()
        done = False
        for _ in range(args.max_replans):
            raw = {
                "observation.images.camera1": torch.from_numpy(obs["overhead_camera"].copy()).permute(2, 0, 1).float() / 255,
                "observation.images.camera2": torch.from_numpy(obs["wrist_camera"].copy()).permute(2, 0, 1).float() / 255,
                "observation.state": torch.tensor(sim_qpos_to_dataset_row(obs["state"], gripper_limits_rad=limits), dtype=torch.float32),
                "task": task,
            }
            # Fixed latent isolates changes caused by the observation/instruction.
            anchor_state = raw["observation.state"].numpy().copy()
            torch.manual_seed(args.seed)
            with torch.inference_mode():
                actions = post(policy.predict_action_chunk(pre(raw)))[0].cpu().numpy()
            if actions.shape != (policy.config.chunk_size, 6) or not np.isfinite(actions).all():
                raise RuntimeError("Invalid predicted SO101 action chunk")
            targets = from_delta_joint(actions, anchor_state)
            for row in targets[:policy.config.n_action_steps]:
                command = dataset_row_to_sim_qpos(row, gripper_limits_rad=limits)
                command = np.clip(command, env.unwrapped._target_low, env.unwrapped._target_high)
                # CF frames are already at 50 Hz: one command, one environment step.
                obs, _, terminated, truncated, info = env.step(command)
                frames.append(np.concatenate([obs["overhead_camera"], obs["wrist_camera"]], axis=1))
                if terminated or truncated:
                    done = True
                    break
            if done:
                break
        args.output.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(args.output, frames, fps=contract["fps"])
        print({key: info.get(key) for key in ("success", "is_grasped", "obj_to_target_dist")})
        print(f"Video: {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
