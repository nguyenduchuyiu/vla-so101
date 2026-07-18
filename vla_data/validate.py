"""Validate SO101 language dataset structure, units, and grounding labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_grounding(record: dict) -> None:
    spec = record["task_spec"]
    instruction = record["language_instruction"].lower()
    canonical = record["canonical_instruction"].lower()
    for key in ("object_name", "target_name", "direction"):
        value = spec.get(key)
        if value is not None and value.lower() not in instruction:
            raise ValueError(
                f"episode {record['episode_index']}: instruction lost {key}={value!r}"
            )
        if value is not None and key != "direction" and value.lower() not in canonical:
            raise ValueError(
                f"episode {record['episode_index']}: canonical label lost {key}={value!r}"
            )
    for role in ("source", "target"):
        entity = spec.get(role)
        if entity is None:
            continue
        for key in ("size", "color", "object_type"):
            value = entity.get(key)
            if value is not None and str(value).lower() not in instruction:
                raise ValueError(
                    f"episode {record['episode_index']}: instruction lost "
                    f"{role}.{key}={value!r}"
                )
    if spec["skill"] == "move":
        expected = float(spec["distance_m"])
        if f"{expected:.2f}" not in instruction:
            raise ValueError(
                f"episode {record['episode_index']}: instruction lost move distance {expected:.2f}"
            )


def validate_dataset(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError("dataset must contain meta/info.json and meta/episodes.jsonl")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    records = _load_jsonl(episodes_path)
    if info.get("format") != "so101_language_npz_v1":
        raise ValueError(f"unsupported format: {info.get('format')!r}")
    if len(records) != info["total_episodes"]:
        raise ValueError("total_episodes does not match episodes.jsonl")

    total_frames = 0
    task_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    loaded: dict[int, dict[str, np.ndarray]] = {}
    for expected_index, record in enumerate(records):
        if record["episode_index"] != expected_index:
            raise ValueError("episode indices must be contiguous and ordered")
        if not record["success"]:
            raise ValueError(f"episode {expected_index} is not a successful oracle rollout")
        if record["task"] == "pick_and_place" and record["final_info"].get("is_grasped", 0.0) > 0.5:
            raise ValueError(
                f"episode {expected_index}: placement terminated before the object was released"
            )
        _validate_grounding(record)
        path = root / record["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as episode:
            required = {
                "observation.state",
                "action",
                "observation.environment_state",
                "observation.images.overhead",
                "observation.images.wrist",
                "reward",
                "success",
                "done",
                "timestamp",
                "oracle_stage",
            }
            missing = required - set(episode.files)
            if missing:
                raise ValueError(f"episode {expected_index} missing keys {sorted(missing)}")
            lengths = {key: len(episode[key]) for key in required}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"episode {expected_index} has misaligned arrays: {lengths}")
            n = next(iter(lengths.values()))
            if n != record["num_frames"] or n < 2:
                raise ValueError(f"episode {expected_index} frame count mismatch")
            state = episode["observation.state"]
            action = episode["action"]
            if state.shape != (n, 6) or action.shape != (n, 6):
                raise ValueError(f"episode {expected_index}: state/action must be (T, 6)")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"episode {expected_index}: non-finite state/action")
            for name, values in (("state", state), ("action", action)):
                if np.min(values[:, 5]) < -0.5 or np.max(values[:, 5]) > 100.5:
                    raise ValueError(
                        f"episode {expected_index}: {name} gripper is not percent [0,100]"
                    )
                if np.max(np.abs(values[:, :5])) > 181.0:
                    raise ValueError(
                        f"episode {expected_index}: {name} arm joints are not plausible degrees"
                    )
            image_shape = tuple(info["image_shape"])
            for key in ("observation.images.overhead", "observation.images.wrist"):
                image = episode[key]
                if image.dtype != np.uint8 or image.shape != (n, *image_shape):
                    raise ValueError(
                        f"episode {expected_index}: {key} must be uint8 (T,{image_shape})"
                    )
            timestamps = episode["timestamp"]
            if np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"episode {expected_index}: timestamps are not increasing")
            if episode["done"][-1] != 1.0 or episode["success"][-1] != 1.0:
                raise ValueError(f"episode {expected_index}: terminal success marker is missing")
            if np.any(episode["done"][:-1] != 0.0):
                raise ValueError(f"episode {expected_index}: done appears before the last frame")
            if float(np.max(np.ptp(action, axis=0))) < 1.0:
                raise ValueError(f"episode {expected_index}: action trajectory is effectively static")
            if info.get("dataset_kind") == "grouped_counterfactual":
                loaded[expected_index] = {
                    "initial_state": state[0].copy(),
                    "initial_overhead": episode["observation.images.overhead"][0].copy(),
                    "initial_wrist": episode["observation.images.wrist"][0].copy(),
                    "action": action.copy(),
                }
        total_frames += n
        task_counts[record["task"]] += 1
        split_counts[record["split"]] += 1

    if total_frames != info["total_frames"]:
        raise ValueError("total_frames does not match stored episode arrays")
    report = {
        "episodes": len(records),
        "frames": total_frames,
        "tasks": dict(task_counts),
        "splits": dict(split_counts),
        "status": "valid",
    }
    if info.get("dataset_kind") == "grouped_counterfactual":
        groups: dict[int, list[dict]] = {}
        for record in records:
            groups.setdefault(int(record["counterfactual_group_id"]), []).append(record)
        expected_per_group = int(info.get("episodes_per_group", 4))
        for group_id, group in groups.items():
            if len(group) != expected_per_group:
                raise ValueError(
                    f"counterfactual group {group_id} has {len(group)} episodes, "
                    f"expected {expected_per_group}"
                )
            if len({record["split"] for record in group}) != 1:
                raise ValueError(f"counterfactual group {group_id} leaks across splits")
            if len({record["scene_seed"] for record in group}) != 1:
                raise ValueError(f"counterfactual group {group_id} uses multiple scene seeds")
            scene_specs = {
                json.dumps(record["scene_spec"], sort_keys=True) for record in group
            }
            if len(scene_specs) != 1:
                raise ValueError(f"counterfactual group {group_id} scene specs differ")
            task_pairs = {
                (
                    record["task_spec"]["source"]["color"],
                    record["task_spec"]["target"]["color"],
                )
                for record in group
            }
            if len(task_pairs) != expected_per_group:
                raise ValueError(f"counterfactual group {group_id} does not enumerate unique tasks")
            reference = loaded[group[0]["episode_index"]]
            for record in group[1:]:
                current = loaded[record["episode_index"]]
                if not np.array_equal(reference["initial_state"], current["initial_state"]):
                    raise ValueError(
                        f"counterfactual group {group_id} initial robot states differ"
                    )
                for camera in ("initial_overhead", "initial_wrist"):
                    pixel_error = np.abs(
                        reference[camera].astype(np.int16) - current[camera].astype(np.int16)
                    )
                    if int(pixel_error.max()) > 1:
                        raise ValueError(
                            f"counterfactual group {group_id} {camera} differs materially"
                        )
            for left_index, left in enumerate(group):
                for right in group[left_index + 1 :]:
                    left_action = loaded[left["episode_index"]]["action"]
                    right_action = loaded[right["episode_index"]]["action"]
                    common = min(len(left_action), len(right_action))
                    divergence = float(
                        np.mean(np.abs(left_action[:common] - right_action[:common]))
                    )
                    if divergence < 0.1:
                        raise ValueError(
                            f"counterfactual group {group_id} has non-divergent action pairs"
                        )

        train_pairs = {
            (
                record["task_spec"]["source"]["color"],
                record["task_spec"]["target"]["color"],
            )
            for record in records
            if record["split"] == "train"
        }
        heldout_pair_cases = 0
        for record in records:
            if record.get("heldout_pair"):
                heldout_pair_cases += 1
                pair = (
                    record["task_spec"]["source"]["color"],
                    record["task_spec"]["target"]["color"],
                )
                if pair in train_pairs:
                    raise ValueError(f"held-out semantic pair {pair} leaked into train")
        train_source_colors = {
            record["task_spec"]["source"]["color"]
            for record in records
            if record["split"] == "train"
        }
        train_target_colors = {
            record["task_spec"]["target"]["color"]
            for record in records
            if record["split"] == "train"
        }
        heldout_color_cases = sum(
            1
            for record in records
            if record["split"] == "test_heldout_color"
            and (
                record["task_spec"]["source"]["color"] not in train_source_colors
                or record["task_spec"]["target"]["color"] not in train_target_colors
            )
        )
        if len(groups) >= 12:
            required_splits = {
                "train",
                "test_seen_semantics_new_scene",
                "test_paraphrase",
                "test_heldout_pair",
                "test_heldout_color",
            }
            missing_splits = required_splits - set(split_counts)
            if missing_splits:
                raise ValueError(f"counterfactual dataset missing splits {sorted(missing_splits)}")
            if heldout_pair_cases == 0:
                raise ValueError("counterfactual dataset has no held-out pair evaluation cases")
            if heldout_color_cases == 0:
                raise ValueError("counterfactual dataset has no held-out color evaluation cases")
        report["counterfactual_groups"] = len(groups)
        report["counterfactual_contract"] = "same initial observation, different task/action"
        report["heldout_pair_cases"] = heldout_pair_cases
        report["heldout_color_cases"] = heldout_color_cases
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    report = validate_dataset(args.dataset.resolve())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
