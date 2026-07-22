"""SmolVLM-VLA training dataset.

Yields per-sample dicts:
  {
    'language_instruction': str,
    'image_input': FloatTensor[V, C, S, S],  # S = image_size (default 384)
    'image_mask': BoolTensor[V],
    'proprio': FloatTensor[dim_proprio],
    'action': FloatTensor[T, dim_action],
  }
"""

from __future__ import annotations
from typing import Dict, Iterable
import io
import json
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
from mmengine import fileio
from .utils import action_slice, build_image_transform
from .domain_config import DATA_WEIGHTS
from .domain_handler.registry import get_handler_cls


class SmolVLMDataReader(IterableDataset):
    """Infinite data reader for SmolVLM-VLA training."""

    def __init__(
        self,
        metas_path: str,
        num_actions: int = 10,
        num_views: int = 3,
        training: bool = True,
        action_mode: str = "so101_delta",
        image_size: int = 384,
        samples_per_episode: int | None = None,
    ):
        self.num_views = num_views
        self.training = training
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.image_size = image_size
        self.samples_per_episode = samples_per_episode
        self.metas: Dict[str, dict] = {}

        print(f"[SmolVLM Dataset] Image size: {self.image_size}x{self.image_size}")
        print(f"[SmolVLM Dataset] Action mode: {action_mode}")

        # Load metadata: a directory of *.json, a single meta json, or a json
        # listing multiple meta paths.
        if fileio.isdir(metas_path):
            meta_files = fileio.list_dir_or_file(
                metas_path, suffix=".json", recursive=True, list_dir=False
            )
            root = metas_path
        elif metas_path.endswith('.json'):
            with open(metas_path, 'r') as f:
                content = json.load(f)
            if isinstance(content, list):
                meta_files, root = content, ""
            else:
                meta_files, root = [metas_path], ""
        else:
            meta_files, root = [metas_path], ""

        for file in meta_files:
            with io.BytesIO(fileio.get(fileio.join_path(root, file))) as f:
                meta = json.load(f)
            print(f"== dataset {meta['dataset_name']} with {len(meta['datalist'])} trajs")
            self.metas[meta["dataset_name"]] = meta

        self.image_aug = build_image_transform(self.image_size, True)
        self.image_no_aug = build_image_transform(self.image_size, False)

    def _iter_one_dataset(self, dataset_name: str) -> Iterable[dict]:
        """Iterate over one dataset."""
        meta = self.metas[dataset_name]
        Handler = get_handler_cls(dataset_name)
        image_aug = (
            self.image_no_aug
            if meta.get("disable_image_augmentation", False)
            else self.image_aug
        )
        while True:
            traj_indices = list(range(len(meta["datalist"])))
            if self.training and not meta.get("preserve_order", False):
                random.shuffle(traj_indices)

            handler = Handler(meta=meta, num_views=self.num_views)
            for traj_idx in traj_indices:
                try:
                    for sample in handler.iter_episode(
                        traj_idx,
                        num_actions=self.num_actions,
                        training=self.training,
                        image_aug=image_aug,
                        lang_aug_map=meta.get("lang_aug_map"),
                        action_mode=self.action_mode,
                        samples_per_episode=self.samples_per_episode,
                    ):
                        has_proprio = "proprio" in sample
                        slice_result = action_slice(sample["abs_trajectory"])

                        if has_proprio:
                            sample["action"] = slice_result["action"]
                        else:
                            sample.update(slice_result)
                        del sample["abs_trajectory"]

                        yield sample
                except Exception as e:
                    # One corrupted episode must not abort the whole stream.
                    print(f"[dataset {dataset_name}] skipped traj {traj_idx}: {e}")
                    continue

            if not self.training:
                return

    def __iter__(self):
        """Main iteration."""
        names = list(self.metas.keys())
        if not self.training:
            for n in names:
                yield from self._iter_one_dataset(n)
        else:
            gens = [iter(self._iter_one_dataset(n)) for n in names]
            ws = [DATA_WEIGHTS.get(n, 1.0) for n in names]
            s = sum(ws)
            ws = [w / s for w in ws]
            while True:
                i = random.choices(range(len(names)), weights=ws, k=1)[0]
                yield next(gens[i])


def create_smolvlm_dataloader(
    batch_size: int,
    metas_path: str,
    num_actions: int,
    training: bool,
    action_mode: str,
    num_workers: int = 4,
    image_size: int = 384,
    num_views: int = 3,
    samples_per_episode: int | None = None,
):
    """Create a DataLoader for SmolVLM-VLA training.

    action_mode: SO101 action space, e.g. "so101_joint" or "so101_delta".
    """
    from torch.utils.data import DataLoader

    def worker_init_fn(worker_id: int):
        base_seed = torch.initial_seed() % (2**32)
        np.random.seed(base_seed)
        random.seed(base_seed)
        torch.manual_seed(base_seed)

    dataset = SmolVLMDataReader(
        metas_path=metas_path,
        num_actions=num_actions,
        training=training,
        action_mode=action_mode,
        image_size=image_size,
        num_views=num_views,
        samples_per_episode=samples_per_episode,
    )

    structured = [
        meta for meta in dataset.metas.values()
        if meta.get("dataset_name") in ("so101_balanced_counterfactual", "cf_balanced")
    ]
    if structured:
        if num_workers != 0:
            raise ValueError(
                "structured counterfactual datasets require num_workers=0 so worker "
                "interleaving cannot split anchor groups"
            )
        # so101_balanced_counterfactual structures each batch to a fixed composition
        # and pins batch_size to sampler_batch_size; cf_balanced only needs the
        # num_workers=0 guarantee above (it streams balanced samples in random
        # anchor order, so any batch_size is fine).
        pinned = {
            int(meta["sampler_batch_size"])
            for meta in structured
            if meta.get("dataset_name") == "so101_balanced_counterfactual"
        }
        if pinned and pinned != {batch_size}:
            raise ValueError(
                "so101_balanced_counterfactual metadata requires batch_size="
                f"{sorted(pinned)}, got {batch_size}"
            )

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1

    return DataLoader(**loader_kwargs)


__all__ = [
    "SmolVLMDataReader",
    "create_smolvlm_dataloader",
]