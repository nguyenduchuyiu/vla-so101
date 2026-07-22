"""Compute normalization statistics (mean, std, q01, q99) for a dataset using so101_delta mode.

Iterates through samples produced by create_smolvlm_dataloader (or cf_balanced handler),
computes delta actions (delta_arm = future_arm - proprio_arm, gripper_absolute = future_gripper),
and writes a norm_stats JSON file compatible with SO101DeltaActionSpace and train_smolvlm.py.

Usage:
    python -m cf_data.compute_norm_stats --data-dir data/cf_smoke_test --output norm_stats/cf_smoke_test_norm.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from models.action_hub import SO101DeltaActionSpace
from simvla_datasets.dataset_smolvlm import create_smolvlm_dataloader


def compute_norm_stats(data_dir: Path, output_path: Path, max_samples: int = 10000) -> dict:
    meta_path = data_dir / "meta" / "cf_balanced.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run build.py first.")

    loader = create_smolvlm_dataloader(
        batch_size=16,
        metas_path=str(meta_path),
        num_actions=10,
        training=False,
        action_mode="so101_delta",
        num_workers=0,
        image_size=96,
        num_views=2,
    )

    space = SO101DeltaActionSpace()

    all_states: list[np.ndarray] = []
    all_deltas: list[np.ndarray] = []

    print(f"Collecting samples from {meta_path}...")
    sample_count = 0

    for batch in loader:
        proprio = batch["proprio"]  # [B, 6]
        action = batch["action"]    # [B, H, 6] (absolute joint targets)

        # Compute so101_delta: arm delta, gripper absolute
        delta = space._to_delta(proprio, action)  # [B, H, 6]

        all_states.append(proprio.cpu().numpy())
        all_deltas.append(delta.reshape(-1, 6).cpu().numpy())

        sample_count += proprio.shape[0]
        if sample_count >= max_samples:
            break

    states_arr = np.concatenate(all_states, axis=0)  # [N, 6]
    deltas_arr = np.concatenate(all_deltas, axis=0)  # [N * H, 6]

    state_mean = np.mean(states_arr, axis=0).tolist()
    state_std = np.std(states_arr, axis=0).tolist()
    state_q01 = np.quantile(states_arr, 0.01, axis=0).tolist()
    state_q99 = np.quantile(states_arr, 0.99, axis=0).tolist()

    action_mean = np.mean(deltas_arr, axis=0).tolist()
    action_std = np.std(deltas_arr, axis=0).tolist()
    action_q01 = np.quantile(deltas_arr, 0.01, axis=0).tolist()
    action_q99 = np.quantile(deltas_arr, 0.99, axis=0).tolist()

    norm_dict = {
        "norm_stats": {
            "state": {
                "mean": state_mean,
                "std": state_std,
                "q01": state_q01,
                "q99": state_q99,
            },
            "actions": {
                "mean": action_mean,
                "std": action_std,
                "q01": action_q01,
                "q99": action_q99,
            },
        },
        "metadata": {
            "representation": "arm_delta_from_current_proprio_gripper_absolute",
            "num_state_samples": int(states_arr.shape[0]),
            "num_action_targets": int(deltas_arr.shape[0]),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(norm_dict, indent=2), encoding="utf-8")
    print(f"Successfully computed norm stats and saved to: {output_path}")
    return norm_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cf_smoke_test"))
    parser.add_argument("--output", type=Path, default=Path("norm_stats/cf_smoke_test_norm.json"))
    parser.add_argument("--max-samples", type=int, default=10000)
    args = parser.parse_args()

    compute_norm_stats(args.data_dir, args.output, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
