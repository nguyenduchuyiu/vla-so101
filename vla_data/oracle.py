"""Privileged scripted experts for SO101-Nexus MuJoCo tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares


OPEN_GRIPPER_RAD = float(np.deg2rad(40.0))
CLOSED_GRIPPER_RAD = float(np.deg2rad(-10.0))
# Pose observed in successful SO101-Nexus demonstrations.  Fixing the last two
# arm joints turns the first three joints into a stable positional IK chain and
# preserves a side-grasp orientation compatible with the SO-101 jaw geometry.
GRASP_WRIST_RAD = np.deg2rad(np.array([26.0, -110.0], dtype=np.float64))
ARM_REACHED_TOL_RAD = float(np.deg2rad(0.5))


@dataclass(frozen=True)
class Stage:
    name: str
    qpos: np.ndarray
    gripper_rad: float
    steps: int
    max_joint_step: float = 0.015
    completion: str = "arm"


class Oracle:
    """Task-dispatched joint-position oracle using simulator ground truth."""

    def __init__(self, env_id: str, env: Any):
        self.env_id = env_id
        self.env = env
        self.u = env.unwrapped
        self.stages = self._plan()
        self.stage_index = 0
        self.stage_step = 0
        self.command = self.u.data.ctrl[self.u._actuator_ids].copy()

    @property
    def finished(self) -> bool:
        return self.stage_index >= len(self.stages)

    def select_action(self) -> tuple[np.ndarray, str]:
        while not self.finished:
            stage = self.stages[self.stage_index]
            actual = self.u._get_current_qpos()
            if self.stage_step > 0 and self._stage_complete(stage, actual):
                self.stage_index += 1
                self.stage_step = 0
                continue
            if self.stage_step >= stage.steps:
                raise RuntimeError(f"oracle stage did not converge: {stage.name}")
            break
        if self.finished:
            return self.command.astype(np.float32), "finished"

        stage = self.stages[self.stage_index]
        actual = self.u._get_current_qpos()[:5]
        # Servo the actuator target against actual qpos. This compensates the
        # steady-state PD/gravity offset that appears near the table.
        correction = np.clip(
            0.15 * (stage.qpos - actual),
            -stage.max_joint_step,
            stage.max_joint_step,
        )
        self.command[:5] = np.clip(
            self.command[:5] + correction,
            self.u._target_low[:5],
            self.u._target_high[:5],
        )
        self.command[5] = stage.gripper_rad
        name = stage.name
        self.stage_step += 1
        return self.command.astype(np.float32), name

    def _stage_complete(self, stage: Stage, actual: np.ndarray) -> bool:
        if stage.completion == "arm":
            return bool(np.max(np.abs(stage.qpos - actual[:5])) <= ARM_REACHED_TOL_RAD)
        if stage.completion == "grasped":
            return bool(self.u._is_grasping())
        if stage.completion == "released":
            return not bool(self.u._is_grasping())
        raise ValueError(f"unknown stage completion condition: {stage.completion}")

    def _position_ik(self, target: np.ndarray, q0: np.ndarray) -> np.ndarray:
        """Solve TCP position while preserving a demonstrated grasp wrist pose."""
        data = mujoco.MjData(self.u.model)
        data.qpos[:] = self.u.data.qpos
        qpos_addrs = np.array(
            [self.u.model.jnt_qposadr[jid] for jid in self.u._joint_ids[:5]], dtype=int
        )

        def residual(first_three: np.ndarray) -> np.ndarray:
            data.qpos[qpos_addrs] = np.concatenate([first_three, GRASP_WRIST_RAD])
            mujoco.mj_forward(self.u.model, data)
            return 100.0 * (data.site_xpos[self.u._tcp_site_id] - target)

        result = least_squares(
            residual,
            q0[:3],
            bounds=(self.u._target_low[:3], self.u._target_high[:3]),
            max_nfev=400,
        )
        error_m = float(np.linalg.norm(residual(result.x)) / 100.0)
        if error_m > 0.008:
            raise RuntimeError(f"IK target is unreachable (position error {error_m:.4f} m)")
        return np.concatenate([result.x, GRASP_WRIST_RAD])

    def _free_position_ik(self, target: np.ndarray, q0: np.ndarray) -> np.ndarray:
        """Position IK for non-grasp tasks, with all five arm joints free."""
        data = mujoco.MjData(self.u.model)
        data.qpos[:] = self.u.data.qpos
        qpos_addrs = np.array(
            [self.u.model.jnt_qposadr[jid] for jid in self.u._joint_ids[:5]], dtype=int
        )

        def residual(qpos: np.ndarray) -> np.ndarray:
            data.qpos[qpos_addrs] = qpos
            mujoco.mj_forward(self.u.model, data)
            return np.concatenate(
                [100.0 * (data.site_xpos[self.u._tcp_site_id] - target), 0.02 * (qpos - q0)]
            )

        result = least_squares(
            residual,
            q0,
            bounds=(self.u._target_low[:5], self.u._target_high[:5]),
            max_nfev=400,
        )
        error_m = float(np.linalg.norm(residual(result.x)[:3]) / 100.0)
        if error_m > 0.008:
            raise RuntimeError(f"IK target is unreachable (position error {error_m:.4f} m)")
        return result.x

    def _pick_stages(self, obj: np.ndarray, q0: np.ndarray) -> tuple[list[Stage], np.ndarray]:
        above = self._position_ik(obj + [0.0, 0.0, 0.08], q0)
        grasp = self._position_ik(obj + [0.0, 0.0, 0.003], above)
        lifted = self._position_ik(obj + [0.0, 0.0, 0.10], grasp)
        stages = [
            Stage("approach_object", above, OPEN_GRIPPER_RAD, 140),
            Stage("descend_to_object", grasp, OPEN_GRIPPER_RAD, 100),
            Stage("close_gripper", grasp, CLOSED_GRIPPER_RAD, 80, completion="grasped"),
            Stage("lift_object", lifted, CLOSED_GRIPPER_RAD, 140),
        ]
        return stages, lifted

    def _plan(self) -> list[Stage]:
        q0 = self.u._get_current_qpos()[:5].copy()
        if self.env_id == "MuJoCoPickLift-v1":
            obj = self.u._get_target_pose()[:3].copy()
            stages, _ = self._pick_stages(obj, q0)
            return stages
        if self.env_id in {
            "MuJoCoPickAndPlace-v1",
            "MuJoCoCounterfactualPickAndPlace-v1",
        }:
            obj = self.u._get_object_pose()[:3].copy()
            target = self.u._get_target_pos().copy()
            stages, lifted = self._pick_stages(obj, q0)
            above_target = self._position_ik(target + [0.0, 0.0, 0.10], lifted)
            # Keep the object above the environment's placement-height slack
            # while descending. Otherwise Gym can terminate as "placed" while
            # the closed gripper is still holding the cube.
            place = self._position_ik(target + [0.0, 0.0, 0.050], above_target)
            retreat = self._position_ik(target + [0.0, 0.0, 0.10], place)
            stages.extend(
                [
                    Stage("move_above_target", above_target, CLOSED_GRIPPER_RAD, 140),
                    Stage("lower_to_target", place, CLOSED_GRIPPER_RAD, 100),
                    Stage("release_object", place, OPEN_GRIPPER_RAD, 80, completion="released"),
                    Stage("retreat", retreat, OPEN_GRIPPER_RAD, 140),
                ]
            )
            return stages
        if self.env_id == "MuJoCoTouch-v1":
            obj = self.u._get_target_pose()[:3].copy()
            above = self._position_ik(obj + [0.0, 0.0, 0.08], q0)
            touch = self._position_ik(obj + [0.0, 0.0, 0.02], above)
            return [
                Stage("approach_object", above, OPEN_GRIPPER_RAD, 120),
                Stage("touch_object", touch, OPEN_GRIPPER_RAD, 100),
            ]
        if self.env_id == "MuJoCoMove-v1":
            # Aim slightly beyond the success plane to absorb PD/gravity
            # steady-state error, especially for the backward direction.
            target = self.u._target_pos.copy() + 0.02 * self.u._dir_vec
            move = self._free_position_ik(target, q0)
            direction = self.u.config.direction
            max_joint_step = {
                "down": 0.006,
                "forward": 0.003,
                "backward": 0.004,
            }.get(direction, 0.002)
            return [
                Stage(
                    "move_end_effector",
                    move,
                    OPEN_GRIPPER_RAD,
                    400,
                    max_joint_step=max_joint_step,
                )
            ]
        raise ValueError(f"No oracle implemented for {self.env_id!r}")
