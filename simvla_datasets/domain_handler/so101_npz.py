"""SO-101 language NPZ data handler."""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from .base import DomainHandler


class SO101NPZHandler(DomainHandler):
    dataset_name = "so101_npz"

    def __init__(self, meta: dict, num_views: int = 2) -> None:
        super().__init__(meta, num_views)
        if num_views < 2:
            raise ValueError("SO101NPZHandler requires overhead and wrist views")
        self.episodes = meta["datalist"]

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int = 10,
        training: bool = True,
        image_aug=None,
        action_mode: str = "so101_joint",
        samples_per_episode: int | None = None,
        **kwargs,
    ) -> Iterable[dict]:
        if action_mode not in ("so101_joint", "so101_delta"):
            raise ValueError(f"SO101 NPZ data requires an SO101 action mode, got {action_mode}")

        record = self.episodes[traj_idx]
        with np.load(record["path"]) as episode:
            state = episode["observation.state"]
            actions = episode["action"]
            overhead = episode["observation.images.overhead"]
            wrist = episode["observation.images.wrist"]

            if state.shape[1:] != (6,) or actions.shape[1:] != (6,):
                raise ValueError("SO101 state and action must be [T, 6]")
            if not (len(state) == len(actions) == len(overhead) == len(wrist)):
                raise ValueError("SO101 episode arrays are not time-aligned")

            indices = list(range(len(state)))
            if training:
                if samples_per_episode is not None:
                    if samples_per_episode < 1:
                        raise ValueError("samples_per_episode must be positive")
                    if samples_per_episode < len(indices):
                        # Always retain gripper transition frames. They are rare but
                        # determine whether the robot actually grasps/releases.
                        early_count = min(2, samples_per_episode)
                        mandatory = list(range(early_count))
                        gripper_delta = np.abs(np.diff(actions[:, 5]))
                        transition_rows = np.flatnonzero(gripper_delta > 5.0) + 1
                        for row in transition_rows:
                            window = list(range(
                                max(0, int(row) - 1),
                                min(len(indices), int(row) + 2),
                            ))
                            mandatory.extend(window)
                            # Large per-episode budgets can afford extra transition
                            # repetitions; small budgets retain whole-task coverage.
                            if samples_per_episode >= 48:
                                focus = list(range(
                                    int(row), min(len(indices), int(row) + 4)
                                ))
                                mandatory.extend(focus * 3)
                        indices = mandatory[:samples_per_episode]
                        selected = set(indices)
                        remaining = samples_per_episode - len(indices)
                        if remaining:
                            candidates = np.array(
                                [idx for idx in range(early_count, len(state))
                                 if idx not in selected]
                            )
                            bins = np.array_split(
                                candidates,
                                remaining,
                            )
                            indices.extend(int(random.choice(part)) for part in bins if len(part))
                random.shuffle(indices)

            image_mask = torch.zeros(self.num_views, dtype=torch.bool)
            image_mask[:2] = True
            instruction = record["language_instruction"]

            for idx in indices:
                images = [Image.fromarray(overhead[idx]), Image.fromarray(wrist[idx])]
                if image_aug:
                    images = [image_aug(image) for image in images]
                while len(images) < self.num_views:
                    images.append(torch.zeros_like(images[0]))

                # action_slice expects row zero to be current proprioception.
                # Collector rows pair observation[t] with command[t], so the
                # supervised chunk must start at action[t].
                future_actions = actions[idx : idx + num_actions]
                if len(future_actions) < num_actions:
                    pad = np.repeat(
                        future_actions[-1:], num_actions - len(future_actions), axis=0
                    )
                    future_actions = np.concatenate([future_actions, pad], axis=0)
                action_chunk = np.concatenate([state[idx : idx + 1], future_actions], axis=0)
                yield {
                    "language_instruction": instruction,
                    "image_input": torch.stack(images),
                    "image_mask": image_mask,
                    "proprio": torch.as_tensor(state[idx], dtype=torch.float32),
                    "abs_trajectory": torch.as_tensor(action_chunk, dtype=torch.float32),
                }
