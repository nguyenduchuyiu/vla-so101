"""Run a saved SmolVLA or task-conditioned ACT checkpoint in the SO101 simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from scipy.interpolate import make_interp_spline
from so101_nexus.lerobot_dataset import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row

from cf_data.collect import CONTROL_DT, make_env
from cf_data.core import OBJECTIVE_COLORS, TARGET_COLORS, get_gripper_limits, instruction
from .data import CAMERAS, JOINT_NAMES, NUM_OBJECTIVES, from_delta_joint


def load_checkpoint(path: Path, device: str):
    contract = json.loads((path / "so101_contract.json").read_text())
    if contract["joint_order"] != JOINT_NAMES or contract["camera_mapping"] != CAMERAS:
        raise ValueError("Checkpoint does not match the SO101 camera/joint contract")
    if contract["action_semantics"] != "arm_delta_from_frame_state_gripper_absolute":
        raise ValueError("Expected per-frame delta-joint SO101 policy")
    family = contract.get("policy_family", "smolvla")
    if family == "act":
        config = ACTConfig.from_pretrained(path)
        config.device = device
        # Full checkpoint weights are authoritative; do not redownload ImageNet.
        config.pretrained_backbone_weights = None
        policy = ACTPolicy.from_pretrained(path, config=config, strict=True).to(device).eval()
    elif family == "smolvla":
        config = SmolVLAConfig.from_pretrained(path)
        config.device = device
        policy = SmolVLAPolicy.from_pretrained(path, config=config, strict=True).to(device).eval()
    else:
        raise ValueError(f"Unsupported policy family: {family!r}")
    pre = PolicyProcessorPipeline.from_pretrained(
        path, config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": device}},
    )
    post = PolicyProcessorPipeline.from_pretrained(
        path, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
    )
    return policy, pre, post, contract


def _act_conditioned_state(state: torch.Tensor, source: int, target: int, contract: dict) -> torch.Tensor:
    conditioning = contract.get("task_conditioning", {})
    if conditioning.get("object_order") != list(OBJECTIVE_COLORS):
        raise ValueError("ACT checkpoint object ordering does not match the environment")
    if conditioning.get("target_order") != list(TARGET_COLORS):
        raise ValueError("ACT checkpoint target ordering does not match the environment")
    if conditioning.get("total_state_dimensions") != 6 + NUM_OBJECTIVES:
        raise ValueError("ACT checkpoint does not use the expected 21D conditioned state")
    one_hot = torch.zeros(NUM_OBJECTIVES, dtype=state.dtype)
    one_hot[source * len(TARGET_COLORS) + target] = 1
    return torch.cat((state, one_hot))


def resample_policy_targets_for_sim(targets: np.ndarray, policy_fps: int) -> np.ndarray:
    """Interpolate absolute 30 Hz policy targets onto the simulator's 50 Hz clock."""
    duration = len(targets) / policy_fps
    count = max(1, int(round(duration / CONTROL_DT)))
    policy_t = np.arange(len(targets), dtype=np.float64) / policy_fps
    sim_t = np.arange(count, dtype=np.float64) * CONTROL_DT
    result = np.empty((count, targets.shape[1]), dtype=np.float32)
    for channel in range(5):
        result[:, channel] = np.interp(sim_t, policy_t, targets[:, channel])
    previous = np.clip(np.searchsorted(policy_t, sim_t, side="right") - 1, 0, len(targets) - 1)
    result[:, 5] = targets[previous, 5]
    return result


def resample_policy_targets_bspline_for_sim(
    targets: np.ndarray,
    current_state: np.ndarray,
    policy_fps: int,
    spline_fps: int = 100,
) -> np.ndarray:
    """Join a 30 Hz chunk to the current state with a B-spline, then sample at 50 Hz.

    The 100 Hz trajectory is an internal smoothing timeline. The simulator still
    receives commands at its native 50 Hz. Arm spline values are bounded by the
    knot range to prevent cubic overshoot; the gripper remains zero-order held.
    """
    targets = np.asarray(targets, dtype=np.float32)
    current_state = np.asarray(current_state, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[0] < 1 or targets.shape[1] != 6:
        raise ValueError("targets must have shape (steps, 6) with at least one step")
    if current_state.shape != (6,):
        raise ValueError("current_state must have shape (6,)")
    if policy_fps < 1 or spline_fps < 1:
        raise ValueError("policy_fps and spline_fps must be positive")
    sim_fps = int(round(1 / CONTROL_DT))
    if spline_fps % sim_fps:
        raise ValueError(f"spline_fps must be an integer multiple of simulator fps ({sim_fps})")

    # Put the measured state at t=0 and shift predicted targets forward by one
    # policy tick. This removes the discontinuous target jump at every replan.
    knot_t = np.arange(len(targets) + 1, dtype=np.float64) / policy_fps
    knots = np.concatenate([current_state[None], targets], axis=0)
    duration = len(targets) / policy_fps
    dense_count = max(1, int(round(duration * spline_fps)))
    dense_t = np.arange(dense_count, dtype=np.float64) / spline_fps
    dense = np.empty((dense_count, 6), dtype=np.float32)
    degree = min(3, len(knot_t) - 1)
    for channel in range(5):
        values = knots[:, channel]
        smooth = make_interp_spline(knot_t, values, k=degree)(dense_t)
        dense[:, channel] = np.clip(smooth, values.min(), values.max())

    target_t = np.arange(1, len(targets) + 1, dtype=np.float64) / policy_fps
    previous = np.searchsorted(target_t, dense_t, side="right") - 1
    dense[:, 5] = current_state[5]
    has_target = previous >= 0
    dense[has_target, 5] = targets[previous[has_target], 5]

    sim_count = max(1, int(round(duration / CONTROL_DT)))
    sim_t = np.arange(sim_count, dtype=np.float64) * CONTROL_DT
    dense_indices = np.rint(sim_t * spline_fps).astype(np.int64)
    if dense_indices[-1] >= len(dense):
        raise RuntimeError("100 Hz spline timeline does not cover the 50 Hz simulator timeline")
    return dense[dense_indices]


def evaluate_episode(
    policy,
    pre,
    post,
    contract: dict,
    *,
    device: str,
    source: int,
    target: int,
    seed: int,
    max_replans: int,
    execute_steps: int | None = None,
    interpolation: str = "linear",
    spline_fps: int = 100,
    output: Path | None = None,
    env=None,
) -> dict:
    """Run one deterministic closed-loop episode with an already-loaded policy."""
    if max_replans < 1:
        raise ValueError("max_replans must be positive")
    if interpolation not in ("linear", "bspline"):
        raise ValueError(f"Unknown interpolation: {interpolation}")
    if contract["fps"] < 1:
        raise ValueError("Invalid dataset control rate")
    if execute_steps is None:
        execute_steps = policy.config.n_action_steps
    if not 1 <= execute_steps <= policy.config.chunk_size:
        raise ValueError(
            f"execute_steps must be between 1 and chunk_size ({policy.config.chunk_size})"
        )
    family = contract.get("policy_family", "smolvla")
    temporal_ensemble = family == "act" and policy.config.temporal_ensemble_coeff is not None
    if temporal_ensemble and execute_steps != 1:
        raise ValueError("ACT temporal ensembling must query the policy with execute_steps=1")
    owns_env = env is None
    if env is None:
        shape = policy.config.input_features["observation.images.camera1"].shape
        env = make_env(width=shape[2], height=shape[1], source_index=source, robot_init_qpos_noise=0.0)
    frames = []
    command_rows = []
    actual_rows = []
    try:
        if env.unwrapped.control_mode != "pd_joint_pos":
            raise ValueError("This delta-to-target adapter requires pd_joint_pos control")
        obs, info = env.reset(seed=seed)
        env.unwrapped.set_objective(source, target)
        obs = env.unwrapped._get_obs()
        limits = get_gripper_limits(env)
        actual_rows.append(sim_qpos_to_dataset_row(obs["state"], gripper_limits_rad=limits))
        task = instruction(source, target)
        policy.reset()
        absolute_ensembler = (
            ACTTemporalEnsembler(policy.config.temporal_ensemble_coeff, policy.config.chunk_size)
            if temporal_ensemble
            else None
        )
        done = False
        ever_grasped = False
        ever_placed = False
        max_lift_height = float("-inf")
        min_tcp_to_obj_dist = float("inf")
        sim_steps_elapsed = 0
        replans_executed = 0
        for replan_index in range(max_replans):
            robot_state = torch.tensor(
                sim_qpos_to_dataset_row(obs["state"], gripper_limits_rad=limits),
                dtype=torch.float32,
            )
            raw = {
                "observation.images.camera1": torch.from_numpy(obs["overhead_camera"].copy()).permute(2, 0, 1).float() / 255,
                "observation.images.camera2": torch.from_numpy(obs["wrist_camera"].copy()).permute(2, 0, 1).float() / 255,
                "observation.state": robot_state,
            }
            anchor_state = robot_state.numpy().copy()
            if family == "act":
                raw["observation.state"] = _act_conditioned_state(
                    robot_state, source, target, contract
                )
            else:
                raw["task"] = task
            with torch.inference_mode():
                processed = pre(raw)
                # Do not call ACTPolicy.select_action here. Its built-in temporal
                # ensemble operates in normalized delta space, but every predicted
                # chunk is relative to a different current robot state.
                torch.manual_seed(seed)
                actions = post(policy.predict_action_chunk(processed))[0].cpu().numpy()
            expected_steps = policy.config.chunk_size
            if actions.shape != (expected_steps, 6) or not np.isfinite(actions).all():
                raise RuntimeError(
                    f"Invalid predicted SO101 actions: expected {(expected_steps, 6)}, got {actions.shape}"
                )
            targets = from_delta_joint(actions, anchor_state)
            if temporal_ensemble:
                # Convert each anchor-relative chunk to physical absolute targets
                # first, then ensemble overlapping predictions in a common frame.
                policy_targets = (
                    absolute_ensembler.update(torch.from_numpy(targets[None]))[0]
                    .numpy()[None]
                )
            else:
                policy_targets = targets[:execute_steps]
            if family == "act" and execute_steps == 1:
                # Query at the nearest 50 Hz simulator tick to every exact k/30
                # policy timestamp. Interval lengths follow 2,1,2,2,1,... rather
                # than rounding every interval to two ticks (which would be 25 Hz).
                desired_sim_steps = int(
                    round((replan_index + 1) / contract["fps"] / CONTROL_DT)
                )
                count = max(1, desired_sim_steps - sim_steps_elapsed)
                sim_targets = np.repeat(policy_targets, count, axis=0)
            elif interpolation == "bspline":
                sim_targets = resample_policy_targets_bspline_for_sim(
                    policy_targets, anchor_state, contract["fps"], spline_fps
                )
            else:
                sim_targets = resample_policy_targets_for_sim(policy_targets, contract["fps"])
            for row in sim_targets:
                command_rows.append(row.copy())
                command = dataset_row_to_sim_qpos(row, gripper_limits_rad=limits)
                command = np.clip(command, env.unwrapped._target_low, env.unwrapped._target_high)
                obs, _, terminated, truncated, info = env.step(command)
                sim_steps_elapsed += 1
                actual_rows.append(sim_qpos_to_dataset_row(obs["state"], gripper_limits_rad=limits))
                ever_grasped |= bool(info.get("is_grasped", False))
                ever_placed |= bool(info.get("is_obj_placed", False))
                max_lift_height = max(max_lift_height, float(info.get("lift_height", float("-inf"))))
                min_tcp_to_obj_dist = min(
                    min_tcp_to_obj_dist, float(info.get("tcp_to_obj_dist", float("inf")))
                )
                if output is not None:
                    frames.append(np.concatenate([obs["overhead_camera"], obs["wrist_camera"]], axis=1))
                if terminated or truncated:
                    done = True
                    break
            if done:
                break
            replans_executed += 1

        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            media.write_video(output, frames, fps=int(round(1 / CONTROL_DT)))
        commands = np.asarray(command_rows, dtype=np.float32)
        actual = np.asarray(actual_rows, dtype=np.float32)
        command_steps = np.diff(np.concatenate([actual[:1, :5], commands[:, :5]], axis=0), axis=0)
        velocity = np.diff(actual[:, :5], axis=0) / CONTROL_DT
        acceleration = np.diff(velocity, axis=0) / CONTROL_DT
        return {
            "seed": seed,
            "source": source,
            "target": target,
            "task": task,
            "policy_family": family,
            "success": bool(info.get("success", False)),
            "is_grasped": bool(info.get("is_grasped", False)),
            "ever_grasped": ever_grasped,
            "ever_placed": ever_placed,
            "max_lift_height": max_lift_height,
            "min_tcp_to_obj_dist": min_tcp_to_obj_dist,
            "obj_to_target_dist": float(info.get("obj_to_target_dist", float("nan"))),
            "interpolation": (
                "zoh_exact_rational_clock" if family == "act" and execute_steps == 1 else interpolation
            ),
            "execute_steps": execute_steps,
            "temporal_ensemble": temporal_ensemble,
            "nominal_replan_hz": contract["fps"] / execute_steps,
            "effective_sim_replan_hz": (
                (replans_executed + int(done)) / (len(commands) * CONTROL_DT)
                if len(commands)
                else 0.0
            ),
            "motion": {
                "sim_steps": len(commands),
                "max_command_step_deg": float(np.max(np.abs(command_steps))),
                "p95_command_step_deg": float(np.percentile(np.abs(command_steps), 95)),
                "max_actual_speed_deg_s": float(np.max(np.abs(velocity))),
                "p95_actual_speed_deg_s": float(np.percentile(np.abs(velocity), 95)),
                "max_actual_accel_deg_s2": float(np.max(np.abs(acceleration)))
                if len(acceleration)
                else 0.0,
            },
            **({"video": str(output)} if output is not None else {}),
        }
    finally:
        if owns_env:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--source", type=int, choices=range(5), default=0)
    parser.add_argument("--target", type=int, choices=range(3), default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-replans", type=int, default=100)
    parser.add_argument(
        "--execute-steps",
        type=int,
        help="Actions executed per prediction chunk; default comes from checkpoint",
    )
    parser.add_argument(
        "--no-temporal-ensemble",
        action="store_true",
        help="For ACT, use only the newest predicted chunk instead of overlapping-chunk ensembling",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        help="Override ACT ensemble coefficient; negative values favor newer predictions",
    )
    parser.add_argument("--interpolation", choices=["linear", "bspline"], default="linear")
    parser.add_argument("--spline-fps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("outputs/smolvla_cf.mp4"))
    args = parser.parse_args()
    if args.no_temporal_ensemble and args.temporal_ensemble_coeff is not None:
        raise ValueError("Choose either --no-temporal-ensemble or --temporal-ensemble-coeff")
    policy, pre, post, contract = load_checkpoint(args.checkpoint, args.device)
    if args.no_temporal_ensemble and contract.get("policy_family") == "act":
        policy.config.temporal_ensemble_coeff = None
    elif args.temporal_ensemble_coeff is not None and contract.get("policy_family") == "act":
        policy.config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
    result = evaluate_episode(
        policy, pre, post, contract,
        device=args.device, source=args.source, target=args.target, seed=args.seed,
        max_replans=args.max_replans, execute_steps=args.execute_steps,
        interpolation=args.interpolation,
        spline_fps=args.spline_fps, output=args.output,
    )
    print(result)
    print(f"Video: {args.output}")


if __name__ == "__main__":
    main()
