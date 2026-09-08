"""Validate exported LeRobot trajectories against their raw 50 Hz sources."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import LeRobotCFDataset, from_delta_joint
from .export import _action_chunks, _load_nominal_suffix, load_branch_trajectory, resample_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/lerobot_cf"))
    args = parser.parse_args()
    stored_contract = json.loads((args.data / "meta/cf_contract.json").read_text())
    dataset = LeRobotCFDataset(
        args.data, chunk_size=int(stored_contract["chunk_size"]), split="all"
    )
    contract = dataset.contract()
    raw_root = Path(contract["raw_source"])
    info = json.loads((raw_root / "meta/info.json").read_text())
    anchors = [
        json.loads(line)
        for line in (raw_root / "meta/anchors.jsonl").read_text().splitlines()
        if line
    ]
    nominal_records = {
        record["episode_id"]: record
        for record in (
            json.loads(line)
            for line in (raw_root / "meta/nominal_episodes.jsonl").read_text().splitlines()
            if line
        )
    }
    manifest = [
        json.loads(line)
        for line in (args.data / "meta/cf_samples.jsonl").read_text().splitlines()
        if line
    ]
    if len(dataset) != len(manifest):
        raise ValueError("LeRobot rows and CF manifest differ in length")

    states, actions = [], []
    expected_cache: dict[tuple, tuple[dict, np.ndarray, np.ndarray, str]] = {}
    for row_index, mapping in enumerate(manifest):
        if mapping["sample_kind"] == "nominal":
            record = nominal_records[mapping["source_episode_id"]]
            key = ("nominal", mapping["source_episode_id"])
        else:
            anchor_index = mapping["source_anchor_index"]
            branch_index = mapping["source_branch_index"]
            anchor = anchors[anchor_index]
            branch = anchor["branches"][branch_index]
            key = ("counterfactual", anchor_index, branch_index)
        if key not in expected_cache:
            if mapping["sample_kind"] == "nominal":
                source = _load_nominal_suffix(raw_root, record["file"], 0, float(info["control_dt"]))
                expected_task = record["instruction"]
            else:
                source = load_branch_trajectory(raw_root, anchor, branch, float(info["control_dt"]))
                expected_task = branch["instruction"]
            trajectory = resample_trajectory(source, contract["fps"])
            chunks, padding = _action_chunks(
                trajectory["action"], trajectory["observation.state"], contract["chunk_size"],
                trajectory["terminal_action"],
            )
            expected_cache[key] = trajectory, chunks, padding, expected_task
        trajectory, chunks, padding, expected_task = expected_cache[key]
        frame = mapping["frame_index"]
        row = dataset.dataset[row_index]
        state = row["observation.state"].numpy()
        chunk = row["cf.action_chunk"].numpy()
        np.testing.assert_allclose(state, trajectory["observation.state"][frame], atol=1e-5)
        np.testing.assert_allclose(chunk, chunks[frame], atol=1e-4, rtol=1e-5)
        for camera, source_key in (
            ("observation.images.camera1", "observation.images.overhead"),
            ("observation.images.camera2", "observation.images.wrist"),
        ):
            expected_image = trajectory[source_key][frame].transpose(2, 0, 1).astype(np.float32) / 255
            actual_image = row[camera].numpy()
            if contract["image_storage"] == "image":
                np.testing.assert_allclose(actual_image, expected_image, atol=1 / 255)
            elif np.mean(np.abs(actual_image - expected_image)) > 0.03:
                raise ValueError(f"video image drift is too large at row {row_index}, camera {camera}")
        np.testing.assert_array_equal(row["action_is_pad"].numpy(), padding[frame])
        if bool(row["next.done"]) != bool(trajectory["terminal"][frame]):
            raise ValueError(f"terminal mismatch at row {row_index}")
        if row["task"] != expected_task:
            raise ValueError(f"instruction mismatch at row {row_index}")
        decoded = from_delta_joint(chunk, state)
        absolute_indices = np.minimum(np.arange(contract["chunk_size"]) + frame, len(trajectory["action"]) - 1)
        expected_absolute = trajectory["action"][absolute_indices].copy()
        expected_absolute[padding[frame]] = trajectory["terminal_action"]
        np.testing.assert_allclose(decoded, expected_absolute, atol=1e-4, rtol=1e-5)
        states.append(state)
        actions.append(chunk[~padding[frame]])

    for key, values in (
        ("observation.state", np.stack(states)),
        ("action", np.concatenate(actions)),
    ):
        expected_mean = values.astype(np.float64).mean(0)
        expected_std = values.astype(np.float64).std(0).clip(1e-6)
        stats = dataset.stats()[key]
        np.testing.assert_allclose(stats["mean"], expected_mean, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(stats["std"], expected_std, rtol=1e-5, atol=1e-6)

    print(
        json.dumps(
            {
                "status": "PASS",
                "episodes": contract["num_episodes"],
                "frames": len(dataset),
                "anchors": contract["num_anchors"],
                "source_fps": contract["source_fps"],
                "fps": contract["fps"],
                "chunk_size": dataset.chunk_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
