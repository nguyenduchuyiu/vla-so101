"""Counterfactual multi-object/multi-target SO101-Nexus MuJoCo environment."""

from __future__ import annotations

import tempfile
from typing import ClassVar

import gymnasium
import mujoco
import numpy as np
from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.config import ControlMode, PickAndPlaceConfig
from so101_nexus.constants import COLOR_MAP
from so101_nexus.mujoco.base_env import SO101NexusMuJoCoBaseEnv
from so101_nexus.mujoco.spawn_utils import place_freejoint_slot
from so101_nexus.object_slots import ObjectSlot, build_object_scene_xml, extract_object_slots
from so101_nexus.objects import CubeObject
from so101_nexus.rewards import reach_progress
from so101_nexus.scene import MUJOCO_SCENE_OPTION_XML

ENV_ID = "MuJoCoCounterfactualPickAndPlace-v1"
_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()
_TARGET_Z = 0.001
_PLACE_Z_SLACK = 0.01


def _target_bodies(colors: tuple[str, ...], radius: float) -> str:
    bodies: list[str] = []
    for index, color in enumerate(colors):
        r, g, b, a = COLOR_MAP[color]
        bodies.append(
            f'    <body name="cf_target_{index}" pos="0.4 0 {_TARGET_Z}">\n'
            f'      <geom name="cf_target_geom_{index}" type="cylinder" '
            f'size="{radius} 0.001" rgba="{r} {g} {b} {a}" '
            'contype="0" conaffinity="0"/>\n'
            "    </body>\n"
        )
    return "".join(bodies)


class CounterfactualPickAndPlaceEnv(SO101NexusMuJoCoBaseEnv):
    """Two visible cubes and two visible trays; task selection does not alter scene."""

    config: PickAndPlaceConfig
    default_config_cls: ClassVar[type[PickAndPlaceConfig]] = PickAndPlaceConfig

    def __init__(
        self,
        config: PickAndPlaceConfig,
        *,
        source_index: int,
        target_index: int,
        target_colors: tuple[str, ...],
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        objects = config.object_pool()
        if len(objects) < 2 or not all(isinstance(obj, CubeObject) for obj in objects):
            raise ValueError("counterfactual env requires at least two CubeObject sources")
        if not 0 <= source_index < len(objects):
            raise IndexError(source_index)
        if not 0 <= target_index < len(target_colors):
            raise IndexError(target_index)
        self._init_common(
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )
        self.source_index = source_index
        self.target_index = target_index
        self.target_colors = target_colors
        slot_names = [f"cf_source_{index}" for index in range(len(objects))]
        xml = build_object_scene_xml(
            objects,
            slot_names,
            COLOR_MAP["gray"],
            option_xml=MUJOCO_SCENE_OPTION_XML,
            robot_xml_path=str(_SO101_XML),
            model_name="counterfactual_pick_place",
            extra_bodies=_target_bodies(target_colors, config.target_disc_radius),
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", dir=_SO101_DIR, delete=True
        ) as handle:
            handle.write(xml)
            handle.flush()
            self.model = mujoco.MjModel.from_xml_path(handle.name)
        self.data = mujoco.MjData(self.model)
        self._slots: list[ObjectSlot] = extract_object_slots(self.model, slot_names, objects)
        self._target_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"cf_target_{index}")
            for index in range(len(target_colors))
        ]
        self._obj_geom_id = self._slots[source_index].geom_id
        self._initial_obj_z = self._slots[source_index].spawn_z
        self._finish_model_setup()

    @property
    def task_description(self) -> str:
        return (
            f"Pick up the {self._slots[self.source_index].obj!r} and place it on the "
            f"{self.target_colors[self.target_index]} tray."
        )

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
        # Anchors give a feasible, separated layout. Jitter is sampled solely
        # from scene seed and never depends on source/target task selection.
        source_anchors = [(0.31, -0.10), (0.31, 0.10), (0.35, 0.0)]
        target_anchors = [(0.41, -0.10), (0.41, 0.10), (0.43, 0.0)]
        jitter = 0.012
        for index, slot in enumerate(self._slots):
            anchor = source_anchors[index]
            xy = (
                anchor[0] + float(rng.uniform(-jitter, jitter)),
                anchor[1] + float(rng.uniform(-jitter, jitter)),
            )
            place_freejoint_slot(self.model, self.data, slot, rng, xy)
        for index, body_id in enumerate(self._target_body_ids):
            anchor = target_anchors[index]
            self.model.body_pos[body_id] = [
                anchor[0] + float(rng.uniform(-jitter, jitter)),
                anchor[1] + float(rng.uniform(-jitter, jitter)),
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


def register() -> None:
    if ENV_ID not in gymnasium.registry:
        gymnasium.register(
            id=ENV_ID,
            entry_point="vla_data.counterfactual_env:CounterfactualPickAndPlaceEnv",
            max_episode_steps=1200,
        )


register()
