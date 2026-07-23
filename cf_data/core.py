"""Shared constants and helpers for the counterfactual pick-and-place dataset.

Objective ``i`` = pick the ``i``-th cube (color ``OBJECTIVE_COLORS[i]``) and place it
on the single shared target. The env places cubes in ``OBJECTIVE_COLORS`` order, so
``env.set_objective(i)`` selects objective ``i``. Phase labels come from the oracle's
current subgoal stage name (see ``STAGE_TO_PHASE``). Dataset rows use the
``sim_qpos_to_dataset_row`` encoding (arm joints in degrees, gripper 0-100 %).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from so101_nexus.lerobot_dataset import sim_qpos_to_dataset_row

# 5 language objectives: one per source cube. The shared target color is fixed.
OBJECTIVE_COLORS: tuple[str, ...] = ("red", "blue", "green", "yellow", "purple")
TARGET_COLOR: str = "white"
NUM_OBJECTIVES: int = len(OBJECTIVE_COLORS)

PHASE_NAMES: tuple[str, ...] = ("REACH_PICK", "GRASP", "REACH_PLACE", "PLACE")
REACH_PICK: int = 0
GRASP: int = 1
REACH_PLACE: int = 2
PLACE: int = 3

# Map each oracle subgoal stage to one of the 4 trajectory phases.
STAGE_TO_PHASE: dict[str, int] = {
    "approach_object": REACH_PICK,
    "descend_to_object": REACH_PICK,
    "close_gripper": GRASP,
    "lift_object": GRASP,
    "move_above_target": REACH_PLACE,
    "lower_to_target": PLACE,
    "release_object": PLACE,
    "retreat": PLACE,
}

# Gripper control range of the SO101 env equals sim_qpos_to_dataset_row's default
# (rad(-10), rad(100)); we rely on the default so the row encoding is unambiguous.


def get_gripper_limits(env: Any) -> tuple[float, float]:
    """Extract gripper lower and upper joint limits in radians from SO101 env."""
    u = env.unwrapped
    return float(u._target_low[5]), float(u._target_high[5])


_gripper_limits = get_gripper_limits


def objective_instruction(objective_id: int) -> str:
    """Language instruction for one objective (names the source cube color)."""
    return f"pick up the {OBJECTIVE_COLORS[objective_id]} block and place it on the {TARGET_COLOR} target"


def stage_to_phase(stage_name: str) -> int:
    if stage_name == "finished":
        return PLACE
    if stage_name not in STAGE_TO_PHASE:
        raise ValueError(f"unknown oracle stage: {stage_name!r}")
    return STAGE_TO_PHASE[stage_name]


def qpos_to_row(qpos_rad: np.ndarray) -> np.ndarray:
    """Simulator joint radians (6,) -> dataset row (6,) [deg, deg, deg, deg, deg, gripper %]."""
    return np.asarray(
        sim_qpos_to_dataset_row(np.asarray(qpos_rad, dtype=np.float64)), dtype=np.float64
    ).copy()


@dataclass(frozen=True)
class Snapshot:
    """Full MuJoCo state needed to reproduce a frame exactly (render-free restore)."""

    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray  # actuator targets for the 6 controlled joints


def save_snapshot(env) -> Snapshot:
    u = env.unwrapped
    return Snapshot(
        qpos=u.data.qpos.copy(),
        qvel=u.data.qvel.copy(),
        ctrl=u.data.ctrl[u._actuator_ids].copy(),
    )


def restore_snapshot(env, snap: Snapshot) -> None:
    """Restore physics state and recompute derived kinematics (xpos/xmat/site_xpos).

    After assignment we call ``mj_forward`` so ``data.xpos``/``site_xpos`` reflect the
    restored ``qpos``. The restored ``ctrl`` keeps the held actuator targets so the
    next ``env.step`` continues the PD servo from the exact anchor configuration.
    """
    u = env.unwrapped
    u.data.qpos[:] = snap.qpos
    u.data.qvel[:] = snap.qvel
    u.data.ctrl[u._actuator_ids] = snap.ctrl
    mujoco.mj_forward(u.model, u.data)


def step_physics(env, action: np.ndarray) -> None:
    """Replicate ``env.step`` physics (clip + ctrl + N_SUBSTEPS) without rendering/obs.

    Used for counterfactual future rollout: only the proprio chunk (qpos) is needed,
    so we skip the camera renders that ``env.step`` would trigger. The clip uses the
    env's action-space bounds, matching ``SO101NexusMuJoCoBaseEnv.step``.
    """
    u = env.unwrapped
    ctrl = np.clip(np.asarray(action, dtype=np.float64), u.action_space.low, u.action_space.high)
    u.data.ctrl[u._actuator_ids] = ctrl
    for _ in range(u._N_SUBSTEPS):
        mujoco.mj_step(u.model, u.data)


def split_for_scene(scene_index: int, n_scenes: int, *, test_ratio: float = 0.15, val_ratio: float = 0.15) -> str:
    """Deterministic scene-level split: a contiguous block of test scenes, then val,
    then train. All anchors of a scene share its split, so all branches of an anchor
    stay together (plan §11). Guarantees >=1 test and >=1 val scene when n_scenes>=3.
    """
    if n_scenes < 1:
        raise ValueError("n_scenes must be positive")
    n_test = max(1, int(round(n_scenes * test_ratio)))
    n_val = max(1, int(round(n_scenes * val_ratio)))
    if n_test + n_val >= n_scenes:
        # Degenerate: force at least one train scene.
        n_val = max(0, n_scenes - n_test - 1)
    if scene_index < n_test:
        return "test"
    if scene_index < n_test + n_val:
        return "val"
    return "train"


def short_hash(*parts) -> str:
    """Stable short hex digest used for scene/episode id derivation from a seed."""
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return h[:10]