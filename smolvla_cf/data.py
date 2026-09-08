"""SO101 counterfactual dataset adapters and delta-joint conversion."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cf_data.core import OBJECTIVE_COLORS as OBJECT_COLORS
from cf_data.core import TARGET_COLORS

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMERAS = {
    "observation.images.camera1": "observation.images.overhead",
    "observation.images.camera2": "observation.images.wrist",
}
NUM_OBJECTIVES = len(OBJECT_COLORS) * len(TARGET_COLORS)


def objective_id_from_task(task: str) -> int:
    """Collapse linguistic task variants to one of the 5 x 3 objectives."""
    words = set(re.findall(r"[a-z]+", task.lower()))
    objects = [i for i, color in enumerate(OBJECT_COLORS) if color in words]
    targets = [i for i, color in enumerate(TARGET_COLORS) if color in words]
    if len(objects) != 1 or len(targets) != 1:
        raise ValueError(f"Cannot identify exactly one object and target in task: {task!r}")
    return objects[0] * len(TARGET_COLORS) + targets[0]


def to_delta_joint(future, anchor_state):
    """Arm targets relative to one reference frame state; gripper stays absolute."""
    action = np.array(future, dtype=np.float32, copy=True)
    action[..., :5] -= np.asarray(anchor_state)[..., :5]
    return action


def from_delta_joint(action, anchor_state):
    """Invert once against the same reference state, never cumulatively."""
    future = np.array(action, dtype=np.float32, copy=True)
    future[..., :5] += np.asarray(anchor_state)[..., :5]
    return future


class CounterfactualDataset(Dataset):
    """One sample = anchor RGB/state + branch instruction + delta-joint chunk.

    Keep every branch separate, and never query adjacent records as future frames.
    Images stay uint8 in RAM; conversion to float happens only for a sampled batch.
    """

    def __init__(self, root: str | Path, chunk_size: int = 32, split: str = "train"):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info["joint_order"] != JOINT_NAMES:
            raise ValueError("Expected SO101 joint order: " + ", ".join(JOINT_NAMES))
        if self.info["action_semantics"] != "pd_joint_pos_command_recoverable_from_snapshot_ctrl":
            raise ValueError("Expected SO101 pd_joint_pos command data")
        if self.info["units"] != {
            "observation.state[0:5]": "degrees",
            "observation.state[5]": "gripper_percent_0_100",
        }:
            raise ValueError("Expected SO101 arm degrees and gripper percentage")
        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"Unknown split: {split}")
        records = [json.loads(line) for line in (self.root / "meta/anchors.jsonl").read_text().splitlines() if line]
        self.anchors = [r for r in records if split == "all" or r["split"] == split]
        if not self.anchors:
            raise ValueError(f"No anchors in split={split!r}; use --split all only for an explicit overfit experiment")
        if chunk_size < 1 or any(chunk_size > r["horizon"] for r in self.anchors):
            raise ValueError("chunk_size must be positive and no longer than the stored CF horizon")
        self.chunk_size = chunk_size
        self.split = split
        self.images: dict[int, dict[str, np.ndarray]] = {}
        self.states = []
        self.actions = []
        self.samples = []
        by_episode: dict[str, list[int]] = {}
        for i, anchor in enumerate(self.anchors):
            by_episode.setdefault(anchor["nominal_episode_path"], []).append(i)
            with np.load(self.root / anchor["cf_path"]) as cf:
                state = cf["anchor_proprio"].astype(np.float32)
                actions = cf["future_chunks"][:, :chunk_size].astype(np.float32)
                if state.shape != (6,) or actions.shape != (len(anchor["branches"]), chunk_size, 6):
                    raise ValueError(f"Invalid state/action shape at {anchor['anchor_id']}")
                if not np.isfinite(state).all() or not np.isfinite(actions).all():
                    raise ValueError(f"Nonfinite joints at {anchor['anchor_id']}")
                for j, branch in enumerate(anchor["branches"]):
                    if int(cf["objective_ids"][j]) != branch["objective_id"] or int(cf["target_ids"][j]) != branch["target_id"]:
                        raise ValueError(f"Branch instruction/action mismatch at {anchor['anchor_id']}")
                    if not branch["instruction"].strip():
                        raise ValueError("Empty branch instruction")
                    self.samples.append((i, j))
                self.states.append(state)
                self.actions.append(actions)
        for episode_path, indices in by_episode.items():
            with np.load(self.root / episode_path) as episode:
                states = episode["observation.state"]
                for i in indices:
                    if not np.allclose(states[self.anchors[i]["anchor_frame"]], self.states[i], atol=1e-4):
                        raise ValueError("Anchor state does not match its observation frame")
                for key, source in CAMERAS.items():
                    frames = episode[source]
                    for i in indices:
                        self.images.setdefault(i, {})[key] = frames[self.anchors[i]["anchor_frame"]].copy()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor_index, branch_index = self.samples[index]
        anchor = self.anchors[anchor_index]
        return {
            **{key: torch.from_numpy(value).permute(2, 0, 1).float() / 255 for key, value in self.images[anchor_index].items()},
            "observation.state": torch.from_numpy(self.states[anchor_index].copy()),
            "action": torch.from_numpy(to_delta_joint(self.actions[anchor_index][branch_index], self.states[anchor_index])),
            "task": anchor["branches"][branch_index]["instruction"],
        }

    def stats(self):
        # Match the branch-uniform training distribution, using only this split.
        states = np.stack([self.states[i] for i, _ in self.samples])
        actions = np.concatenate([to_delta_joint(a, s) for a, s in zip(self.actions, self.states)], axis=0).reshape(-1, 6)
        return {
            key: {"mean": torch.from_numpy(values.mean(0)), "std": torch.from_numpy(values.std(0).clip(1e-6))}
            for key, values in (("observation.state", states), ("action", actions))
        }

    def contract(self):
        return {
            "robot_type": "so101", "joint_order": JOINT_NAMES,
            "arm_units": "degrees", "gripper_units": "percent_0_100",
            "action_semantics": "arm_delta_from_chunk_anchor_gripper_absolute",
            "target_source": "expert pd_joint_pos actuator commands",
            "fps": self.info["fps"], "camera_mapping": CAMERAS,
            "chunk_size": self.chunk_size, "split": self.split,
            "num_anchors": len(self.anchors), "num_samples": len(self),
        }


class LeRobotCFDataset(Dataset):
    """Read full 30 Hz branch trajectories and their explicit padded chunks."""

    def __init__(self, root, chunk_size=50, split="train"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root)
        self._contract = json.loads((self.root / "meta/cf_contract.json").read_text())
        if self._contract["action_semantics"] != "arm_delta_from_frame_state_gripper_absolute":
            raise ValueError("Expected per-frame delta-joint CF data")
        if chunk_size != self._contract["chunk_size"]:
            raise ValueError("Training chunk_size must match the exported CF chunk")
        records = [json.loads(line) for line in (self.root / "meta/cf_samples.jsonl").read_text().splitlines()]
        if any(row.get("row_index") != i for i, row in enumerate(records)):
            raise ValueError("CF manifest row indices are not contiguous")
        self.indices = [row["row_index"] for row in records if split == "all" or row["split"] == split]
        if not self.indices:
            raise ValueError(f"No anchors in split={split!r}; --split all is only for explicit overfit experiments")
        self.dataset = LeRobotDataset(repo_id=self._contract["repo_id"], root=self.root)
        if len(self.dataset) != len(records):
            raise ValueError("CF sample manifest does not match LeRobot rows")
        self.info = {"image_shape": self._contract["image_shape"]}
        self.split = split
        self.chunk_size = chunk_size
        self._stats = json.loads((self.root / "meta/cf_norm_stats.json").read_text())[split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.dataset[self.indices[index]]
        return {
            **{key: row[key] for key in CAMERAS},
            "observation.state": row["observation.state"],
            "task": row["task"],
            "action": row["cf.action_chunk"],
            "action_is_pad": row["action_is_pad"],
        }

    def stats(self):
        return {key: {name: torch.tensor(value, dtype=torch.float32) for name, value in stats.items()}
                for key, stats in self._stats.items()}

    def contract(self):
        return {**self._contract, "split": self.split, "num_samples": len(self)}


class ACTCFDataset(Dataset):
    """ACT view of the same CF rows, with the objective appended to proprioception.

    The observation alone is ambiguous because every scene is paired with all 15
    counterfactual objectives. Conditioning the CVAE and decoder on a 15-way
    objective vector prevents those action labels from being averaged together.
    """

    def __init__(self, root, chunk_size=50, split="train"):
        self.base = LeRobotCFDataset(root, chunk_size, split)
        self.root = self.base.root
        self.info = self.base.info
        self.split = split
        self.chunk_size = chunk_size

        task_to_index = {
            str(task): int(row.task_index)
            for task, row in self.base.dataset.meta.tasks.iterrows()
        }
        index_to_objective = {
            task_index: objective_id_from_task(task)
            for task, task_index in task_to_index.items()
        }
        # Go through Arrow once. Indexing a formatted Hugging Face column row by
        # row is orders of magnitude slower for this 200k-row dataset.
        all_task_indices = (
            self.base.dataset.hf_dataset.data.column("task_index")
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        selected_task_indices = all_task_indices[np.asarray(self.base.indices)]
        self.objective_ids = [index_to_objective[int(i)] for i in selected_task_indices]

        # Each instruction variant must map to the same semantic 5 x 3 space.
        if set(index_to_objective.values()) != set(range(NUM_OBJECTIVES)):
            raise ValueError("LeRobot task table does not cover all 15 object/target objectives")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = self.base[index]
        task = row.pop("task")
        objective_id = self.objective_ids[index]
        if objective_id_from_task(task) != objective_id:
            raise ValueError("Cached objective does not match the row instruction")
        task_one_hot = torch.zeros(NUM_OBJECTIVES, dtype=row["observation.state"].dtype)
        task_one_hot[objective_id] = 1
        row["observation.state"] = torch.cat((row["observation.state"], task_one_hot))
        return row

    def stats(self):
        stats = self.base.stats()
        objective_ids = torch.tensor(self.objective_ids, dtype=torch.long)
        objective_vectors = torch.nn.functional.one_hot(
            objective_ids, num_classes=NUM_OBJECTIVES
        ).float()
        task_mean = objective_vectors.mean(0)
        task_std = objective_vectors.std(0, unbiased=False).clamp_min(1e-6)
        state_stats = stats["observation.state"]
        state_stats["mean"] = torch.cat((state_stats["mean"], task_mean))
        state_stats["std"] = torch.cat((state_stats["std"], task_std))

        image_stats = json.loads((self.root / "meta/stats.json").read_text())
        for key in CAMERAS:
            stats[key] = {
                name: torch.tensor(image_stats[key][name], dtype=torch.float32)
                for name in ("mean", "std")
            }
        return stats

    def contract(self):
        return {
            **self.base.contract(),
            "policy_family": "act",
            "task_conditioning": {
                "representation": "15_way_object_target_one_hot_appended_to_observation.state",
                "object_order": list(OBJECT_COLORS),
                "target_order": list(TARGET_COLORS),
                "robot_state_dimensions": 6,
                "task_dimensions": NUM_OBJECTIVES,
                "total_state_dimensions": 6 + NUM_OBJECTIVES,
            },
        }
