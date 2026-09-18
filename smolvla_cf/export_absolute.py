"""Export successful SO101 nominal episodes with absolute actuator targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .data import JOINT_NAMES
from .export import _load_nominal_suffix


def export_absolute(
    source: Path,
    output: Path,
    repo_id: str,
    *,
    use_videos: bool = True,
    records: list[dict] | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    info = json.loads((source / "meta/info.json").read_text())
    if info["fps"] != 50 or info["control_dt"] != 0.02:
        raise ValueError("Expected the original 50 Hz SO101 nominal trajectories")
    if records is None:
        records = [
            json.loads(line)
            for line in (source / "meta/nominal_episodes.jsonl").read_text().splitlines()
            if line
        ]
        records = [row for row in records if row["split"] == "train"]
    if not records or any(row["split"] != "train" for row in records):
        raise ValueError("Export requires nonempty train episodes only")

    height, width, _ = info["image_shape"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "observation.images.camera1": {
            "dtype": "video" if use_videos else "image",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
        "observation.images.camera2": {
            "dtype": "video" if use_videos else "image",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        fps=50,
        robot_type="so101",
        features=features,
        use_videos=use_videos,
        streaming_encoding=use_videos,
        encoder_threads=2,
        video_files_size_in_mb=1 if use_videos else None,
        rgb_encoder=(
            RGBEncoderConfig(vcodec="h264", g=30, crf=18, preset="fast")
            if use_videos else None
        ),
    )
    manifest = []
    try:
        for episode_index, record in enumerate(records):
            trajectory = _load_nominal_suffix(source, record["file"], 0, 0.02)
            states = trajectory["observation.state"]
            actions = trajectory["action"]
            if states.shape != actions.shape or states.shape[1] != 6:
                raise ValueError(f"Invalid SO101 state/action shape: {record['file']}")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError(f"Nonfinite state/action: {record['file']}")
            for index in range(len(actions)):
                dataset.add_frame({
                    "observation.state": states[index],
                    "action": actions[index],
                    "observation.images.camera1": trajectory["observation.images.overhead"][index],
                    "observation.images.camera2": trajectory["observation.images.wrist"][index],
                    "task": record["instruction"],
                })
            dataset.save_episode()
            manifest.append({
                "episode_index": episode_index,
                "source_file": record["file"],
                "scene_index": record["scene_index"],
                "source_id": record["source_id"],
                "target_id": record["target_id"],
                "frames": len(actions),
            })
    finally:
        dataset.finalize()

    contract = {
        "repo_id": repo_id,
        "raw_source": str(source.resolve()),
        "fps": 50,
        "action_semantics": "absolute_pd_joint_position_target",
        "units": info["units"],
        "camera_mapping": {
            "observation.images.camera1": "observation.images.overhead",
            "observation.images.camera2": "observation.images.wrist",
        },
        "episodes": len(manifest),
        "frames": sum(row["frames"] for row in manifest),
        "manifest": manifest,
    }
    (output / "meta/absolute_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/nominal_spatial"))
    parser.add_argument("--output", type=Path, default=Path("data/lerobot_so101_absolute_50hz"))
    parser.add_argument("--repo-id", default="local/so101_nominal_absolute_50hz")
    parser.add_argument("--image-storage", choices=("video", "image"), default="video")
    args = parser.parse_args()
    contract = export_absolute(
        args.source, args.output, args.repo_id,
        use_videos=args.image_storage == "video",
    )
    print(json.dumps({key: contract[key] for key in ("episodes", "frames", "fps", "action_semantics")}))


if __name__ == "__main__":
    main()
