"""Balanced counterfactual domain handler for the cf_data dataset (plan §12-C).

Reads ``meta/anchors.jsonl`` written by ``cf_data/build.py`` and yields one training
sample per branch. Branches of the same anchor are yielded consecutively and share
``flow_group_id`` (the anchor's index), so the flow-matching head samples a single
t/noise per counterfactual group and the only per-branch difference is the language
instruction and the future proprioception chunk (plan §6/§10).

The sample carries only the keys the model consumes:
  language_instruction, image_input[V,C,S,S], image_mask[V], proprio[D],
  abs_trajectory[num_actions+1, D], flow_group_id.

``abs_trajectory = [anchor_proprio, future_chunk[:num_actions]]`` (repeat-last
padded), so ``action_slice`` -> proprio=anchor_proprio, action=future chunk, and the
``so101_delta`` action space then forms ``delta = action - proprio`` (plan §8).
All branches of an anchor share the anchor image and proprio (criterion 6), so the
training meta must set ``disable_image_augmentation: true`` to keep the shared image
identical across branches.

The plan §10 metadata (phase, objective_id, episode_id, scene_id, anchor_id,
branch_id, is_counterfactual) is stored in ``anchors.jsonl``; this handler is a pure
model feeder and intentionally does not attach it to the yielded sample (the model's
forward signature has no ``**kwargs``).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from .base import DomainHandler


class CFBalancedHandler(DomainHandler):
    dataset_name = "cf_balanced"

    def __init__(self, meta: dict, num_views: int = 2) -> None:
        super().__init__(meta, num_views)
        if num_views < 2:
            raise ValueError("cf_balanced requires two views")
        self.root = Path(meta["dataset_root"])
        anchors_path = self.root / meta.get("anchors_file", "meta/anchors.jsonl")
        self.anchors = [
            json.loads(line)
            for line in anchors_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if not self.anchors:
            raise ValueError(f"no anchors in {anchors_path}")
        # flow_group_id = the anchor's stable position in this list; all branches of
        # an anchor share it so the flow head samples one t/noise per group.
        for ai, anchor in enumerate(self.anchors):
            anchor["_group_id"] = ai
        self.cache: dict[str, dict[str, np.ndarray]] = {}

    def _npz(self, rel: str) -> dict[str, np.ndarray]:
        path = str(self.root / rel)
        if path not in self.cache:
            with np.load(path) as npz:
                self.cache[path] = {key: npz[key].copy() for key in npz.files}
        return self.cache[path]

    @staticmethod
    def _chunk(future: np.ndarray, num_actions: int) -> np.ndarray:
        """future[:num_actions] with repeat-last padding (matches the SO101 handler)."""
        result = future[:num_actions]
        if len(result) < num_actions:
            result = np.concatenate(
                [result, np.repeat(result[-1:], num_actions - len(result), axis=0)]
            )
        return result

    def _sample(self, anchor: dict, branch: dict, branch_pos: int, num_actions: int, image_aug) -> dict:
        nominal = self._npz(anchor["nominal_episode_path"])
        cf = self._npz(anchor["cf_path"])
        t = int(anchor["anchor_frame"])
        # All branches of an anchor share the anchor image and proprio (criterion 6).
        images = [
            Image.fromarray(nominal["observation.images.overhead"][t]),
            Image.fromarray(nominal["observation.images.wrist"][t]),
        ]
        images = [image_aug(image) for image in images]
        while len(images) < self.num_views:
            images.append(torch.zeros_like(images[0]))
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:2] = True
        proprio = np.array(cf["anchor_proprio"], dtype=np.float32)
        future = self._chunk(np.array(cf["future_chunks"][branch_pos], dtype=np.float32), num_actions)
        abs_trajectory = np.concatenate([proprio[None], future], axis=0).astype(np.float32)
        return {
            "language_instruction": branch["instruction"],
            "image_input": torch.stack(images),
            "image_mask": image_mask,
            "proprio": torch.as_tensor(proprio, dtype=torch.float32),
            "abs_trajectory": torch.as_tensor(abs_trajectory, dtype=torch.float32),
            "flow_group_id": torch.tensor(int(anchor["_group_id"]), dtype=torch.long),
        }

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int = 10,
        image_aug=None,
        **kwargs,
    ) -> Iterable[dict]:
        if traj_idx != 0:
            raise IndexError("cf_balanced exposes one synthetic trajectory")
        if image_aug is None:
            raise ValueError("image_aug is required")
        while True:
            order = list(range(len(self.anchors)))
            random.shuffle(order)
            for ai in order:
                anchor = self.anchors[ai]
                for branch_pos, branch in enumerate(anchor["branches"]):
                    yield self._sample(anchor, branch, branch_pos, num_actions, image_aug)


__all__ = ["CFBalancedHandler"]