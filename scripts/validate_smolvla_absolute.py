"""Check absolute SO101 labels and LeRobot chunk boundaries against raw episodes."""

import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from smolvla_cf.export import _load_nominal_suffix


root = Path("data/lerobot_so101_absolute_50hz")
contract = json.loads((root / "meta/absolute_contract.json").read_text())
dataset = LeRobotDataset(
    contract["repo_id"], root=root,
    delta_timestamps={"action": [index / 50 for index in range(83)]},
)
assert len(dataset) == contract["frames"]
offset = 0
for episode in contract["manifest"]:
    raw = _load_nominal_suffix(
        Path(contract["raw_source"]), episode["source_file"], 0, 0.02
    )
    assert len(raw["action"]) == episode["frames"]
    for frame in {0, len(raw["action"]) // 2, len(raw["action"]) - 1}:
        sample = dataset[offset + frame]
        expected = raw["action"][frame : frame + 83]
        np.testing.assert_allclose(sample["action"][: len(expected)], expected, atol=1e-5)
        np.testing.assert_allclose(
            sample["observation.state"], raw["observation.state"][frame], atol=1e-5
        )
        assert sample["observation.images.camera1"].shape == (3, 256, 256)
        assert sample["observation.images.camera2"].shape == (3, 256, 256)
    offset += len(raw["action"])
assert offset == len(dataset)
print(f"validated {len(contract['manifest'])} episodes, {offset} frames, chunk 83")
