"""Validate every exported LeRobot row against its source CF branch."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import LeRobotCFDataset, from_delta_joint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/lerobot_cf"))
    args = parser.parse_args()
    dataset = LeRobotCFDataset(args.data, split="all")
    raw_root = Path(dataset.contract()["raw_source"])
    anchors = [json.loads(line) for line in (raw_root / "meta/anchors.jsonl").read_text().splitlines() if line]
    manifest = [json.loads(line) for line in (args.data / "meta/cf_samples.jsonl").read_text().splitlines() if line]
    if len(dataset) != len(manifest):
        raise ValueError("LeRobot rows and CF manifest differ in length")

    states, actions = [], []
    grouped: dict[str, list[dict]] = {}
    for index, mapping in enumerate(manifest):
        sample = dataset[index]
        anchor = anchors[mapping["source_anchor_index"]]
        branch_index = mapping["source_branch_index"]
        branch = anchor["branches"][branch_index]
        with np.load(raw_root / anchor["cf_path"]) as source:
            expected = source["future_chunks"][branch_index]
            if int(source["objective_ids"][branch_index]) != branch["objective_id"]:
                raise ValueError(f"Source id mismatch at row {index}")
            if int(source["target_ids"][branch_index]) != branch["target_id"]:
                raise ValueError(f"Target id mismatch at row {index}")
        decoded = from_delta_joint(sample["action"].numpy(), sample["observation.state"].numpy())
        if not np.allclose(decoded, expected, atol=1e-4, rtol=1e-5):
            raise ValueError(f"Action chunk mismatch at row {index}")
        if sample["task"] != branch["instruction"]:
            raise ValueError(f"Instruction mismatch at row {index}")
        grouped.setdefault(anchor["anchor_id"], []).append(sample)
        states.append(sample["observation.state"].numpy())
        actions.append(sample["action"].numpy())

    for anchor_id, group in grouped.items():
        reference = group[0]
        for sample in group[1:]:
            if not torch.equal(reference["observation.state"], sample["observation.state"]):
                raise ValueError(f"State differs inside anchor {anchor_id}")
            for camera in ("observation.images.camera1", "observation.images.camera2"):
                if not torch.equal(reference[camera], sample[camera]):
                    raise ValueError(f"Image differs inside anchor {anchor_id}")

    for key, values in (("observation.state", np.stack(states)), ("action", np.concatenate(actions))):
        expected_mean = values.astype(np.float64).mean(0)
        expected_std = values.astype(np.float64).std(0).clip(1e-6)
        stats = dataset.stats()[key]
        np.testing.assert_allclose(stats["mean"], expected_mean, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(stats["std"], expected_std, rtol=1e-5, atol=1e-6)

    multi_branch = sum(len(group) > 1 for group in grouped.values())
    print(json.dumps({
        "status": "PASS", "rows": len(dataset), "anchors": len(grouped),
        "multi_branch_anchors": multi_branch,
        "action_semantics": dataset.contract()["action_semantics"],
        "chunk_size": dataset.chunk_size,
    }, indent=2))


if __name__ == "__main__":
    main()
