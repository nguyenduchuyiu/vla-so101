"""Build normalization stats for SO101 arm deltas and absolute gripper commands."""

import argparse
import json
from pathlib import Path

import numpy as np


def stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()
    meta = json.loads(args.meta.read_text())
    states, targets = [], []
    for row in meta["datalist"]:
        with np.load(row["path"]) as episode:
            state = episode["observation.state"].astype(np.float64)
            action = episode["action"].astype(np.float64)
        states.append(state)
        for offset in range(1, args.horizon + 1):
            future = np.concatenate([action[offset:], np.repeat(action[-1:], offset, 0)])
            target = future.copy()
            target[:, :5] -= state[:, :5]
            targets.append(target)
    state_values = np.concatenate(states)
    target_values = np.concatenate(targets)
    payload = {"norm_stats": {"state": stats(state_values), "actions": stats(target_values)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
