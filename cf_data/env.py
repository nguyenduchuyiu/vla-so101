"""Counterfactual pick-and-place env: N cubes (one per objective) + 1 shared target.

Same scene layout (object poses, robot pose, cameras) is produced for every
objective from a single scene seed, because ``_task_reset`` places objects at
fixed anchors jittered only by ``self.np_random`` (seeded once per reset). The
collector captures the post-reset state once and restores it before running the
oracle toward each objective, so all N nominal trajectories share the same
starting state. ``source_index`` selects which cube the oracle picks; the
target is shared (``target_index`` is always 0).
"""

from __future__ import annotations

import tempfile
from typing import ClassVar

import mujoco
import numpy as np
from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.config import ControlMode, PickAndPlaceConfig
from so101_nexus.constants import COLOR_MAP
from so101_nexus.mujoco.base_env import SO101NexusMuJoCoBaseEnv
from so101_nexus.mujoco.spawn_utils import place_freejoint_slot
from so101_nexus.object_slots import build_object_scene_xml, extract_object_slots
from so101_nexus.scene import MUJOCO_SCENE_OPTION_XML

ENV_ID = "MuJoCoCounterfactualPickAndPlace-v1"  # reused by the oracle's pick-and-place branch
_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()
_TARGET_Z = 0.001
_PLACE_Z_SLACK = 0.01
# Objects spread laterally at a reachable depth; target centered behind them.
_SOURCE_X = 0.28
_SOURCE_Y_RANGE = 0.16
_TARGET_XY = (0.40, 0.0)
_JITTER = 0.012


def _source_anchors(num_objects: int) -> list[tuple[float, float]]:
    # Evenly spaced across y at fixed x. Spacing 0.32/(N-1) keeps min separation
    # (spacing - 2*JITTER) above min_object_target_separation for N up to ~8.
    if num_objects < 2:
        raise ValueError("need at least 2 objects")
    span = 2 * _SOURCE_Y_RANGE
    return [
        (_SOURCE_X, -_SOURCE_Y_RANGE + i * span / (num_objects - 1))
        for i in range(num_objects)
    ]


def _target_body_xml(color: str, radius: float) -> str:
    r, g, b, a = COLOR_MAP[color]
    return (
        '    <body name="cf_target_0" pos="0.4 0 0.001">\n'
        f'      <geom name="cf_target_geom_0" type="cylinder" '
        f'size="{radius} 0.001" rgba="{r} {g} {b} {a}" contype="0" conaffinity="0"/>\n'
        "    </body>\n"
    )


class CFMultiObjectEnv(SO101NexusMuJoCoBaseEnv):
    """N visible cubes and one visible shared target; task = pick source cube, place on target."""

    config: PickAndPlaceConfig
    default_config_cls: ClassVar[type[PickAndPlaceConfig]] = PickAndPlaceConfig

    def __init__(
        self,
        config: PickAndPlaceConfig,
        *,
        source_index: int,
        target_colors: tuple[str, ...],
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        if len(target_colors) != 1:
            raise ValueError("CFMultiObjectEnv uses a single shared target; pass one target color")
        objects = config.object_pool()
        if len(objects) < 2:
            raise ValueError("CFMultiObjectEnv requires at least 2 source objects")
        if not 0 <= source_index < len(objects):
            raise IndexError(source_index)
        self._init_common(
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )
        self.source_index = source_index
        self.target_index = 0  # shared target is always index 0
        self.target_colors = tuple(target_colors)
        slot_names = [f"cf_source_{i}" for i in range(len(objects))]
        xml = build_object_scene_xml(
            objects,
            slot_names,
            COLOR_MAP["gray"],
            option_xml=MUJOCO_SCENE_OPTION_XML,
            robot_xml_path=str(_SO101_XML),
            model_name="cf_multi_object_pick_place",
            extra_bodies=_target_body_xml(self.target_colors[0], config.target_disc_radius),
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", dir=_SO101_DIR, delete=True
        ) as handle:
            handle.write(xml)
            handle.flush()
            self.model = mujoco.MjModel.from_xml_path(handle.name)
        self.data = mujoco.MjData(self.model)
        self._slots = extract_object_slots(self.model, slot_names, objects)
        self._target_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cf_target_0")
        ]
        self._source_anchors = _source_anchors(len(objects))
        self._obj_geom_id = self._slots[source_index].geom_id
        self._initial_obj_z = self._slots[source_index].spawn_z
        self._finish_model_setup()

    def set_objective(self, source_index: int) -> None:
        """Switch which cube the oracle picks (used for nominal + counterfactual replay)."""
        if not 0 <= source_index < len(self._slots):
            raise IndexError(source_index)
        self.source_index = source_index
        self._obj_geom_id = self._slots[source_index].geom_id
        self._initial_obj_z = self._slots[source_index].spawn_z

    @property
    def task_description(self) -> str:
        return f"Pick up the {self._slots[self.source_index].obj!r} and place it on the {self.target_colors[0]} target."

    def _get_object_pose(self) -> np.ndarray:
        addr = self._slots[self.source_index].qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    def _get_target_pos(self) -> np.ndarray:
        return self.data.xpos[self._target_body_ids[self.target_index]].copy()

    def _get_component_data(self, component: object) -> np.ndarray:
        from so101_nexus.observations import ObjectOffset, ObjectPose, TargetOffset, TargetPosition

        if isinstance(component, ObjectPose):
            return self._get_object_pose()
        if isinstance(component, ObjectOffset):
            return self._get_object_pose()[:3] - self._get_tcp_pose()[:3]
        if isinstance(component, TargetPosition):
            return self._get_target_pos()
        if isinstance(component, TargetOffset):
            return self._get_target_pos() - self._get_object_pose()[:3]
        return super()._get_component_data(component)

    def _task_reset(self) -> None:
        rng = self.np_random
        for slot, anchor in zip(self._slots, self._source_anchors, strict=True):
            xy = (
                anchor[0] + float(rng.uniform(-_JITTER, _JITTER)),
                anchor[1] + float(rng.uniform(-_JITTER, _JITTER)),
            )
            place_freejoint_slot(self.model, self.data, slot, rng, xy)
        self.model.body_pos[self._target_body_ids[0]] = [
            _TARGET_XY[0] + float(rng.uniform(-_JITTER, _JITTER)),
            _TARGET_XY[1] + float(rng.uniform(-_JITTER, _JITTER)),
            _TARGET_Z,
        ]
        self._obj_geom_id = self._slots[self.source_index].geom_id
        self._initial_obj_z = self._slots[self.source_index].spawn_z

    def _refresh_reset_reference_state(self) -> None:
        self._initial_obj_z = float(self._get_object_pose()[2])

    def _get_info(self) -> dict:
        obj_pos = self._get_object_pose()[:3]
        target_pos = self._get_target_pos()
        tcp_pos = self._get_tcp_pose()[:3]
        distance = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
        placed = distance <= self.config.goal_thresh and (
            obj_pos[2] < self._initial_obj_z + _PLACE_Z_SLACK
        )
        info = {
            "obj_to_target_dist": distance,
            "is_obj_placed": placed,
            "is_grasped": self._is_grasping(),
            "is_robot_static": self._is_robot_static(),
            "lift_height": float(obj_pos[2] - self._initial_obj_z),
            "tcp_to_obj_dist": float(np.linalg.norm(obj_pos - tcp_pos)),
            "success": placed and self._is_robot_static(),
        }
        if self._privileged_state is not None:
            info["privileged_state"] = self._privileged_state
        return info

    def _compute_reward(self, info: dict) -> float:
        from so101_nexus.rewards import reach_progress

        components = self.config.reward.compute_components(
            reach_progress=reach_progress(
                info["tcp_to_obj_dist"], scale=self.config.reward.tanh_shaping_scale
            ),
            is_grasped=info["is_grasped"] > 0.5,
            task_progress=float(info["is_obj_placed"]),
            is_complete=info["success"],
            action_delta_norm=info.get("action_delta_norm", 0.0),
            energy_norm=info.get("energy_norm", 0.0),
        )
        info["reward_components"] = components
        return float(sum(components.values()))