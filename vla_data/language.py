"""Task schemas and controlled language generation.

Instructions are generated from the post-reset simulator state.  No attribute
is guessed before scene randomization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    skill: str
    object_name: str | None = None
    target_name: str | None = None
    direction: str | None = None
    distance_m: float | None = None
    source: dict[str, Any] | None = None
    target: dict[str, Any] | None = None
    relation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    def render_context(self) -> dict[str, Any]:
        context = self.to_dict()
        if self.source is not None:
            context["object_name"] = " ".join(
                str(self.source[key])
                for key in ("size", "color", "object_type")
                if self.source.get(key)
            )
        if self.target is not None:
            context["target_name"] = " ".join(
                str(self.target[key]) for key in ("color", "object_type") if self.target.get(key)
            )
        return context


_TEMPLATES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "pick_lift": (
        (
            "Pick up the {object_name}.",
            "Grasp the {object_name} and lift it.",
            "Lift the {object_name} off the table.",
        ),
        ("Raise the {object_name} from the table.",),
    ),
    "pick_and_place": (
        (
            "Pick up the {object_name} and place it on the {target_name}.",
            "Move the {object_name} onto the {target_name}.",
            "Put the {object_name} on the {target_name}.",
        ),
        ("Grasp the {object_name}, then set it down on the {target_name}.",),
    ),
    "counterfactual_pick_and_place": (
        (
            "Pick up the {object_name} and place it on the {target_name}.",
            "Move the {object_name} onto the {target_name}.",
            "Put the {object_name} on the {target_name}.",
            "Place the {object_name} on top of the {target_name}.",
            "Grasp the {object_name}, then set it on the {target_name}.",
            "Route the {object_name} to the {target_name}.",
        ),
        (
            "Take the {object_name} and position it atop the {target_name}.",
            "Set the {object_name} down on the {target_name}.",
        ),
    ),
    "touch": (
        (
            "Touch the {object_name}.",
            "Move the gripper to the {object_name}.",
            "Reach for the {object_name}.",
        ),
        ("Bring the end-effector into contact with the {object_name}.",),
    ),
    "move": (
        (
            "Move the end-effector {direction} by {distance_m:.2f} meters.",
            "Shift the gripper {direction} by {distance_m:.2f} meters.",
        ),
        ("Translate the end-effector {distance_m:.2f} meters {direction}.",),
    ),
}


def task_spec_from_env(env_id: str, env: Any) -> TaskSpec:
    """Read the selected task entities after ``env.reset``."""
    unwrapped = env.unwrapped
    if env_id == "MuJoCoPickLift-v1":
        slot = unwrapped._slots[unwrapped._target_slot_idx]
        return TaskSpec(skill="pick_lift", object_name=repr(slot.obj))
    if env_id == "MuJoCoPickAndPlace-v1":
        slot = unwrapped._slots[unwrapped._target_slot_idx]
        return TaskSpec(
            skill="pick_and_place",
            object_name=repr(slot.obj),
            target_name=f"{unwrapped.target_color_name} circle",
        )
    if env_id == "MuJoCoCounterfactualPickAndPlace-v1":
        slot = unwrapped._slots[unwrapped.source_index]
        obj = slot.obj
        return TaskSpec(
            skill="counterfactual_pick_and_place",
            source={
                "object_type": "cube",
                "color": obj.color,
                "size": "small",
            },
            target={
                "object_type": "tray",
                "color": unwrapped.target_colors[unwrapped.target_index],
            },
            relation="on",
        )
    if env_id == "MuJoCoTouch-v1":
        slot = unwrapped._slots[unwrapped._target_slot_idx]
        return TaskSpec(skill="touch", object_name=repr(slot.obj))
    if env_id == "MuJoCoMove-v1":
        return TaskSpec(
            skill="move",
            direction=unwrapped.config.direction,
            distance_m=float(unwrapped._target_displacement),
        )
    raise ValueError(f"No task schema extractor for {env_id!r}")


def make_instruction(spec: TaskSpec, rng: np.random.Generator, *, held_out: bool) -> str:
    """Render a meaning-preserving template from an exact task schema."""
    train_templates, held_out_templates = _TEMPLATES[spec.skill]
    pool = held_out_templates if held_out else train_templates
    template = pool[int(rng.integers(0, len(pool)))]
    return template.format(**spec.render_context())


def canonical_instruction(env: Any) -> str:
    """Return SO101-Nexus' post-randomization canonical task string."""
    return str(env.unwrapped.task_description)
