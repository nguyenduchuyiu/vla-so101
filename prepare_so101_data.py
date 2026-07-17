"""Create SimVLA metadata and train-only normalization stats for SO-101 NPZ data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--meta_output", type=Path, required=True)
    parser.add_argument("--stats_output", type=Path, required=True)
    args = parser.parse_args()

    root = args.data_dir.resolve()
    records_path = root / "meta" / "episodes.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    train_records = [record for record in records if record["split"] == "train"]
    if not train_records:
        raise ValueError(f"No train episodes in {records_path}")

    datalist = []
    states = []
    actions = []
    for record in train_records:
        episode_path = root / record["file"]
        with np.load(episode_path) as episode:
            state = episode["observation.state"]
            action = episode["action"]
        if state.shape[1:] != (6,) or action.shape != state.shape:
            raise ValueError(f"Invalid state/action shape in {episode_path}")
        states.append(state)
        actions.append(action)
        datalist.append(
            {
                "path": str(episode_path),
                "episode_index": record["episode_index"],
                "language_instruction": record["language_instruction"],
            }
        )

    state = np.concatenate(states).astype(np.float64)
    action = np.concatenate(actions).astype(np.float64)

    meta = {
        "dataset_name": "so101_npz",
        "data_dir": str(root),
        "datalist": datalist,
        "num_episodes": len(datalist),
        "state_dim": 6,
        "action_dim": 6,
        "fps": 25,
        "idx_for_delta": [],
    }

    def summarize(values: np.ndarray) -> dict:
        return {
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }

    stats = {
        "norm_stats": {
            "state": summarize(state),
            "actions": summarize(action),
        },
        "metadata": {
            "data_dir": str(root),
            "split": "train",
            "num_episodes": len(datalist),
            "num_steps": len(state),
            "state_dim": 6,
            "action_dim": 6,
        },
    }

    args.meta_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.meta_output.write_text(json.dumps(meta, indent=2) + "\n")
    args.stats_output.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"prepared {len(datalist)} train episodes / {len(state)} frames")


if __name__ == "__main__":
    main()
