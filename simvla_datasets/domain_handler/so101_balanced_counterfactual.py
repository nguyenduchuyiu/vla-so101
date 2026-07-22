"""Batch-structured SO101 counterfactual and phase-balanced sampler."""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from .base import DomainHandler


class SO101BalancedCounterfactualHandler(DomainHandler):
    dataset_name = "so101_balanced_counterfactual"

    def __init__(self, meta: dict, num_views: int = 2) -> None:
        super().__init__(meta, num_views)
        if num_views < 2:
            raise ValueError("balanced SO101 sampling requires two views")
        self.batch_size = int(meta["sampler_batch_size"])
        if self.batch_size % 8:
            raise ValueError("sampler_batch_size must be divisible by 8")
        self.cache: dict[str, dict[str, np.ndarray]] = {}

    def _episode(self, path: str) -> dict[str, np.ndarray]:
        if path not in self.cache:
            with np.load(path) as episode:
                self.cache[path] = {key: episode[key].copy() for key in episode.files}
        return self.cache[path]

    @staticmethod
    def _chunk(actions: np.ndarray, timestep: int, horizon: int) -> np.ndarray:
        result = actions[timestep : timestep + horizon]
        if len(result) < horizon:
            result = np.concatenate(
                [result, np.repeat(result[-1:], horizon - len(result), axis=0)]
            )
        return result

    def _sample(
        self,
        *,
        observation_path: str,
        action_record: dict,
        timestep: int,
        num_actions: int,
        image_aug,
        flow_group_id: int,
    ) -> dict:
        observation = self._episode(observation_path)
        target = self._episode(action_record["path"])
        images = [
            Image.fromarray(observation["observation.images.overhead"][timestep]),
            Image.fromarray(observation["observation.images.wrist"][timestep]),
        ]
        images = [image_aug(image) for image in images]
        while len(images) < self.num_views:
            images.append(torch.zeros_like(images[0]))
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:2] = True
        state = observation["observation.state"][timestep]
        future = self._chunk(target["action"], timestep, num_actions)
        trajectory = np.concatenate([state[None], future], axis=0)
        return {
            "language_instruction": action_record["language_instruction"],
            "image_input": torch.stack(images),
            "image_mask": image_mask,
            "proprio": torch.as_tensor(state, dtype=torch.float32),
            "abs_trajectory": torch.as_tensor(trajectory, dtype=torch.float32),
            "flow_group_id": torch.tensor(flow_group_id, dtype=torch.long),
        }

    def _decision_pair(
        self,
        pair: dict,
        *,
        num_actions: int,
        image_aug,
        group_id: int,
    ) -> Iterable[dict]:
        for side in ("left", "right"):
            yield self._sample(
                observation_path=pair["observation_path"],
                action_record=pair[side],
                timestep=int(pair["timestep"]),
                num_actions=num_actions,
                image_aug=image_aug,
                flow_group_id=group_id,
            )

    @staticmethod
    def _balanced_pair_choice(pairs: list[dict], kind: str) -> dict:
        """Balance the unaffected task axis before choosing a decision window."""
        key = "target_index" if kind == "source" else "source_index"
        groups: dict[int, list[dict]] = {}
        for pair in pairs:
            groups.setdefault(int(pair["left"][key]), []).append(pair)
        return random.choice(random.choice(list(groups.values())))

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int = 10,
        image_aug=None,
        action_mode: str = "so101_delta",
        **kwargs,
    ) -> Iterable[dict]:
        if traj_idx != 0:
            raise IndexError("balanced sampler exposes one synthetic trajectory")
        if action_mode not in ("so101_joint", "so101_delta"):
            raise ValueError(f"unsupported action mode: {action_mode}")
        pairs_per_decision_kind = self.batch_size // 8
        execution_count = self.batch_size // 2
        phases = list(self.meta["execution_by_phase"])
        while True:
            group_id = 0
            for _ in range(pairs_per_decision_kind):
                yield from self._decision_pair(
                    self._balanced_pair_choice(
                        self.meta["source_decision_pairs"], "source"
                    ),
                    num_actions=num_actions,
                    image_aug=image_aug,
                    group_id=group_id,
                )
                group_id += 1
            for _ in range(pairs_per_decision_kind):
                yield from self._decision_pair(
                    self._balanced_pair_choice(
                        self.meta["target_decision_pairs"], "target"
                    ),
                    num_actions=num_actions,
                    image_aug=image_aug,
                    group_id=group_id,
                )
                group_id += 1
            for _ in range(execution_count):
                phase = random.choice(phases)
                phase_episode = random.choice(self.meta["execution_by_phase"][phase])
                record = phase_episode["episode"]
                timestep = int(random.choice(phase_episode["timesteps"]))
                yield self._sample(
                    observation_path=record["path"],
                    action_record=record,
                    timestep=timestep,
                    num_actions=num_actions,
                    image_aug=image_aug,
                    flow_group_id=group_id,
                )
                group_id += 1


__all__ = ["SO101BalancedCounterfactualHandler"]
