"""Export full nominal/CF branch trajectories to LeRobot at an exact 30 Hz."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import datasets
import numpy as np
from lerobot.configs import RGBEncoderConfig
from lerobot.datasets import aggregate as lerobot_aggregate
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from cf_data.core import qpos_to_row, split_for_scene

from .data import CAMERAS, JOINT_NAMES


def _nearest_indices(source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_t, target_t, side="left")
    right = np.clip(right, 0, len(source_t) - 1)
    left = np.clip(right - 1, 0, len(source_t) - 1)
    choose_left = np.abs(target_t - source_t[left]) <= np.abs(source_t[right] - target_t)
    return np.where(choose_left, left, right)


def _linear_channels(values: np.ndarray, source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(target_t, source_t, values[:, channel]) for channel in range(values.shape[1])],
        axis=1,
    ).astype(np.float32)


def resample_trajectory(trajectory: dict[str, np.ndarray], target_fps: int = 30) -> dict[str, np.ndarray]:
    """Resample a pre-action trajectory onto ``t_k = k / target_fps``.

    RGB and gripper state use nearest samples. Arm state and absolute arm targets
    are linearly interpolated. The absolute gripper command uses zero-order hold.
    """
    if target_fps < 1:
        raise ValueError("target_fps must be positive")
    source_t = np.asarray(trajectory["timestamp"], dtype=np.float64)
    if len(source_t) == 0 or np.any(np.diff(source_t) <= 0):
        raise ValueError("trajectory timestamps must be nonempty and strictly increasing")
    required = (
        "observation.state",
        "observation.images.overhead",
        "observation.images.wrist",
        "action",
        "phase",
        "terminal",
    )
    if any(len(trajectory[key]) != len(source_t) for key in required):
        raise ValueError("trajectory modalities must have the same length")
    count = int(np.floor(source_t[-1] * target_fps + 1e-7)) + 1
    target_t = np.arange(count, dtype=np.float64) / target_fps
    nearest = _nearest_indices(source_t, target_t)
    previous = np.clip(np.searchsorted(source_t, target_t, side="right") - 1, 0, len(source_t) - 1)

    source_state = np.asarray(trajectory["observation.state"], dtype=np.float32)
    source_action = np.asarray(trajectory["action"], dtype=np.float32)
    state = np.empty((count, 6), dtype=np.float32)
    state[:, :5] = _linear_channels(source_state[:, :5], source_t, target_t)
    state[:, 5] = source_state[nearest, 5]
    action = np.empty((count, 6), dtype=np.float32)
    action[:, :5] = _linear_channels(source_action[:, :5], source_t, target_t)
    action[:, 5] = source_action[previous, 5]

    terminal = np.zeros(count, dtype=bool)
    terminal[-1] = bool(np.asarray(trajectory["terminal"])[-1])
    return {
        "timestamp": target_t.astype(np.float32),
        "observation.state": state,
        "observation.images.overhead": trajectory["observation.images.overhead"][nearest],
        "observation.images.wrist": trajectory["observation.images.wrist"][nearest],
        "action": action,
        "terminal_action": source_action[-1].copy(),
        "phase": np.asarray(trajectory["phase"])[nearest].astype(np.int8),
        "terminal": terminal,
        "source_nearest_index": nearest.astype(np.int64),
    }


def _load_nominal_suffix(root: Path, path: str, start: int, source_dt: float) -> dict[str, np.ndarray]:
    with np.load(root / path) as episode:
        states = episode["observation.state"][start:].astype(np.float32)
        if len(states) == 0:
            raise ValueError(f"empty nominal suffix: {path} frame {start}")
        ctrl = episode["snapshot.ctrl"]
        commands = np.concatenate([ctrl[start + 1 :], ctrl[-1:]], axis=0)
        actions = np.stack([qpos_to_row(row) for row in commands]).astype(np.float32)
        terminal = np.zeros(len(states), dtype=bool)
        terminal[-1] = True
        return {
            "observation.state": states,
            "observation.images.overhead": episode["observation.images.overhead"][start:].copy(),
            "observation.images.wrist": episode["observation.images.wrist"][start:].copy(),
            "action": actions,
            "phase": episode["phase"][start:].astype(np.int8),
            "terminal": terminal,
            "timestamp": (np.arange(len(states)) * source_dt).astype(np.float32),
        }


def load_branch_trajectory(root: Path, anchor: dict, branch: dict, source_dt: float) -> dict[str, np.ndarray]:
    kind = branch.get("trajectory_kind")
    path = branch.get("trajectory_path")
    if not kind or not path:
        raise ValueError(
            "CF observations are missing; rebuild the source with the full-trajectory cf_data.build"
        )
    if kind == "nominal_suffix":
        return _load_nominal_suffix(
            root, path, int(branch.get("trajectory_start_frame", anchor["anchor_frame"])), source_dt
        )
    if kind != "counterfactual":
        raise ValueError(f"unknown trajectory kind: {kind}")
    with np.load(root / path) as episode:
        return {key: episode[key].copy() for key in episode.files}


def _action_chunks(
    actions: np.ndarray, states: np.ndarray, horizon: int, terminal_action: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(horizon, dtype=np.int64)
    indices = np.arange(len(actions), dtype=np.int64)[:, None] + offsets[None, :]
    is_pad = indices >= len(actions)
    absolute = actions[np.minimum(indices, len(actions) - 1)].copy()
    absolute[is_pad] = actions[-1] if terminal_action is None else terminal_action
    delta = absolute.astype(np.float32, copy=True)
    delta[..., :5] -= states[:, None, :5]
    return delta, is_pad


def _dataset_features(info: dict, horizon: int, use_videos: bool) -> dict:
    height, width, _ = info["image_shape"]
    visual_dtype = "video" if use_videos else "image"
    return {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "cf.action_chunk": {
            "dtype": "float32",
            "shape": (horizon, 6),
            "names": ["future_step", "joint"],
        },
        "action_is_pad": {"dtype": "bool", "shape": (horizon,), "names": ["future_step"]},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        **{
            key: {
                "dtype": visual_dtype,
                "shape": (3, height, width),
                "names": ["channels", "height", "width"],
            }
            for key in CAMERAS
        },
    }


def _build_export_tasks(source_root: Path, anchors: list[dict], nominal_records: list[dict]) -> list[dict]:
    scene_indices = sorted({int(record["scene_index"]) for record in nominal_records})
    scene_rank = {scene_index: rank for rank, scene_index in enumerate(scene_indices)}
    tasks = []
    for record in nominal_records:
        tasks.append(
            {
                "trajectory_kind": "nominal_suffix",
                "trajectory_path": record["file"],
                "trajectory_start_frame": 0,
                "task": record["instruction"],
                "split": split_for_scene(scene_rank[int(record["scene_index"])], len(scene_indices)),
                "provenance": {
                    "sample_kind": "nominal",
                    "is_counterfactual": False,
                    "source_episode_id": record["episode_id"],
                    "source_episode_path": record["file"],
                },
            }
        )

    for anchor_index, anchor in enumerate(anchors):
        for branch_index, branch in enumerate(anchor["branches"]):
            if not branch["is_counterfactual"]:
                continue
            kind = branch.get("trajectory_kind")
            path = branch.get("trajectory_path")
            if not kind or not path:
                raise ValueError(
                    "CF observations are missing; rebuild the source with the full-trajectory cf_data.build"
                )
            tasks.append(
                {
                    "trajectory_kind": kind,
                    "trajectory_path": path,
                    "trajectory_start_frame": int(
                        branch.get("trajectory_start_frame", anchor["anchor_frame"])
                    ),
                    "task": branch["instruction"],
                    "split": anchor["split"],
                    "provenance": {
                        "sample_kind": "counterfactual",
                        "is_counterfactual": True,
                        "anchor_id": anchor["anchor_id"],
                        "branch_id": branch["branch_id"],
                        "source_anchor_index": anchor_index,
                        "source_branch_index": branch_index,
                    },
                }
            )
    return tasks


def _load_export_task(source_root: Path, task: dict, source_dt: float) -> dict[str, np.ndarray]:
    if task["trajectory_kind"] == "nominal_suffix":
        return _load_nominal_suffix(
            source_root,
            task["trajectory_path"],
            int(task["trajectory_start_frame"]),
            source_dt,
        )
    if task["trajectory_kind"] != "counterfactual":
        raise ValueError(f"unknown trajectory kind: {task['trajectory_kind']}")
    with np.load(source_root / task["trajectory_path"]) as episode:
        return {key: episode[key].copy() for key in episode.files}


def _empty_moments() -> dict:
    return {
        split: {
            key: {"count": 0, "sum": np.zeros(6), "sum_sq": np.zeros(6)}
            for key in ("observation.state", "action")
        }
        for split in ("train", "val", "test", "all")
    }


def _update_moments(moment: dict, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 6)
    moment["count"] += len(values)
    moment["sum"] += values.sum(axis=0)
    moment["sum_sq"] += np.square(values).sum(axis=0)


def _write_shard_impl(
    source_root: str,
    output: str,
    repo_id: str,
    tasks: list[dict],
    info: dict,
    horizon: int,
    target_fps: int,
    use_videos: bool,
    encoder_threads: int,
) -> dict:
    """Write one independent LeRobot shard. This function is process-safe."""
    datasets.disable_progress_bars()
    logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)
    logging.getLogger("libav").setLevel(logging.ERROR)
    try:
        import av

        av.logging.set_level(av.logging.ERROR)
    except ImportError:
        pass
    source_root_path = Path(source_root)
    output_path = Path(output)
    source_dt = float(info["control_dt"])
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        fps=target_fps,
        robot_type="so101",
        features=_dataset_features(info, horizon, use_videos),
        use_videos=use_videos,
        streaming_encoding=use_videos,
        encoder_queue_maxsize=512,
        encoder_threads=encoder_threads,
        # Avoid repeatedly rewriting one ever-growing MP4 while appending episodes.
        # Small files are cheap for the final no-reencode aggregation to remap.
        video_files_size_in_mb=1 if use_videos else None,
        rgb_encoder=(
            RGBEncoderConfig(
                vcodec="h264",
                g=30,
                crf=18,
                preset="fast",
                extra_options={"x264-params": "log-level=none"},
            )
            if use_videos
            else None
        ),
    )
    moments = _empty_moments()
    manifest_path = output_path / "meta/cf_samples.shard.jsonl"
    row_index = 0
    try:
        with manifest_path.open("w") as manifest_file:
            for episode_index, task in enumerate(tasks):
                source = _load_export_task(source_root_path, task, source_dt)
                trajectory = resample_trajectory(source, target_fps)
                chunks, padding = _action_chunks(
                    trajectory["action"],
                    trajectory["observation.state"],
                    horizon,
                    trajectory["terminal_action"],
                )
                for frame_index in range(len(trajectory["action"])):
                    dataset.add_frame(
                        {
                            "observation.images.camera1": trajectory[
                                "observation.images.overhead"
                            ][frame_index],
                            "observation.images.camera2": trajectory["observation.images.wrist"][
                                frame_index
                            ],
                            "observation.state": trajectory["observation.state"][frame_index],
                            "action": chunks[frame_index, 0],
                            "cf.action_chunk": chunks[frame_index],
                            "action_is_pad": padding[frame_index],
                            "next.done": np.asarray(
                                [trajectory["terminal"][frame_index]], dtype=bool
                            ),
                            "task": task["task"],
                        }
                    )
                    manifest_file.write(
                        json.dumps(
                            {
                                "row_index": row_index,
                                "episode_index": episode_index,
                                "frame_index": frame_index,
                                "split": task["split"],
                                "source_nearest_index": int(
                                    trajectory["source_nearest_index"][frame_index]
                                ),
                                **task["provenance"],
                            }
                        )
                        + "\n"
                    )
                    row_index += 1
                dataset.save_episode()
                for stats_split in (task["split"], "all"):
                    _update_moments(
                        moments[stats_split]["observation.state"],
                        trajectory["observation.state"],
                    )
                    _update_moments(moments[stats_split]["action"], chunks[~padding])
    finally:
        dataset.finalize()

    return {
        "root": str(output_path),
        "repo_id": repo_id,
        "episodes": len(tasks),
        "frames": row_index,
        "moments": moments,
    }


def _write_shard(
    source_root: str,
    output: str,
    repo_id: str,
    tasks: list[dict],
    info: dict,
    horizon: int,
    target_fps: int,
    use_videos: bool,
    encoder_threads: int,
    quiet: bool = False,
) -> dict:
    if not quiet:
        return _write_shard_impl(
            source_root,
            output,
            repo_id,
            tasks,
            info,
            horizon,
            target_fps,
            use_videos,
            encoder_threads,
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.parent / f".{output_path.name}.stderr.log"
    with log_path.open("w") as log_file:
        saved_stderr_fd = os.dup(2)
        try:
            os.dup2(log_file.fileno(), 2)
            with contextlib.redirect_stderr(log_file):
                return _write_shard_impl(
                    source_root,
                    output,
                    repo_id,
                    tasks,
                    info,
                    horizon,
                    target_fps,
                    use_videos,
                    encoder_threads,
                )
        finally:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)


def _merge_moments(results: list[dict]) -> dict:
    merged = _empty_moments()
    for result in results:
        for split, split_values in result["moments"].items():
            for key, values in split_values.items():
                merged_value = merged[split][key]
                merged_value["count"] += values["count"]
                merged_value["sum"] += values["sum"]
                merged_value["sum_sq"] += values["sum_sq"]
    return merged


def _aggregate_shards(results: list[dict], output: Path, repo_id: str) -> None:
    """Aggregate shards without losing LeRobot's shaped-array Arrow schema.

    LeRobot's v3 aggregator currently round-trips data through pandas before
    writing it with ``pa.Table.from_pandas``. That fails for Array2D features
    such as ``cf.action_chunk``. Rebuild only the data Arrow table through the
    dataset feature schema, while retaining the upstream video/metadata remap.
    """
    first_meta = lerobot_aggregate.LeRobotDatasetMetadata(
        results[0]["repo_id"], root=Path(results[0]["root"])
    )
    hf_features = get_hf_features_from_features(first_meta.features)
    original_writer = lerobot_aggregate.to_parquet_one_row_group_per_episode

    def write_shaped_data(df, path):
        dataset = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=hf_features)
        table = dataset.with_format("arrow")[:]
        write_table_one_row_group_per_episode(table, path)

    lerobot_aggregate.to_parquet_one_row_group_per_episode = write_shaped_data
    try:
        lerobot_aggregate.aggregate_datasets(
            repo_ids=[result["repo_id"] for result in results],
            roots=[Path(result["root"]) for result in results],
            aggr_repo_id=repo_id,
            aggr_root=output,
            concatenate_videos=False,
            concatenate_data=False,
        )
    finally:
        lerobot_aggregate.to_parquet_one_row_group_per_episode = original_writer


def _write_export_metadata(
    output: Path,
    source_root: Path,
    repo_id: str,
    info: dict,
    anchors: list[dict],
    results: list[dict],
    horizon: int,
    target_fps: int,
    use_videos: bool,
    num_nominal_episodes: int,
) -> None:
    row_offset = 0
    episode_offset = 0
    manifest_path = output / "meta/cf_samples.jsonl"
    with manifest_path.open("w") as destination:
        for result in results:
            local_manifest = Path(result["root"]) / "meta/cf_samples.shard.jsonl"
            for line in local_manifest.read_text().splitlines():
                row = json.loads(line)
                row["row_index"] += row_offset
                row["episode_index"] += episode_offset
                destination.write(json.dumps(row) + "\n")
            row_offset += result["frames"]
            episode_offset += result["episodes"]

    merged = _merge_moments(results)
    all_stats = {}
    for split, split_values in merged.items():
        if not split_values["observation.state"]["count"]:
            continue
        all_stats[split] = {}
        for key, values in split_values.items():
            count = values["count"]
            mean = values["sum"] / count
            variance = np.maximum(values["sum_sq"] / count - np.square(mean), 0.0)
            all_stats[split][key] = {
                "mean": mean.tolist(),
                "std": np.maximum(np.sqrt(variance), 1e-6).tolist(),
            }

    source_dt = float(info["control_dt"])
    source_fps = int(round(1 / source_dt))
    contract = {
        "robot_type": "so101",
        "joint_order": JOINT_NAMES,
        "arm_units": "degrees",
        "gripper_units": "percent_0_100",
        "action_semantics": "arm_delta_from_frame_state_gripper_absolute",
        "target_source": f"expert pd_joint_pos actuator commands resampled from {source_fps} Hz",
        "source_fps": source_fps,
        "fps": target_fps,
        "camera_mapping": CAMERAS,
        "chunk_size": horizon,
        "split": "all",
        "num_anchors": len(anchors),
        "num_episodes": episode_offset,
        "num_nominal_episodes": num_nominal_episodes,
        "num_counterfactual_episodes": episode_offset - num_nominal_episodes,
        "num_samples": row_offset,
        "repo_id": repo_id,
        "image_shape": info["image_shape"],
        "storage": "lerobot_v3_unique_full_observation_trajectories",
        "image_storage": "video" if use_videos else "image",
        "video_encoding": (
            {"codec": "h264", "gop": 30, "crf": 18, "preset": "fast"} if use_videos else None
        ),
        "resampling": {
            "timeline": "t_k=k/fps",
            "rgb": "nearest",
            "arm_state": "linear",
            "gripper_state": "nearest",
            "absolute_arm_target": "linear",
            "absolute_gripper_target": "zero_order_hold",
            "terminal_padding": "repeat_last_action",
        },
        "raw_source": str(source_root.resolve()),
    }
    (output / "meta/cf_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (output / "meta/cf_norm_stats.json").write_text(json.dumps(all_stats, indent=2) + "\n")


def _run_parallel_export(
    source_root: Path,
    output: Path,
    repo_id: str,
    tasks: list[dict],
    info: dict,
    horizon: int,
    target_fps: int,
    use_videos: bool,
    encoder_threads: int,
    workers: int,
    episodes_per_shard: int,
) -> tuple[list[dict], Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.export-shards-", dir=output.parent)
    )
    task_shards = [
        tasks[index : index + episodes_per_shard]
        for index in range(0, len(tasks), episodes_per_shard)
    ]
    results_by_index: dict[int, dict] = {}
    succeeded = False
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for shard_index, shard_tasks in enumerate(task_shards):
                shard_root = temporary_root / f"shard-{shard_index:05d}"
                shard_repo_id = f"{repo_id}_shard_{shard_index:05d}"
                future = executor.submit(
                    _write_shard,
                    str(source_root),
                    str(shard_root),
                    shard_repo_id,
                    shard_tasks,
                    info,
                    horizon,
                    target_fps,
                    use_videos,
                    encoder_threads,
                    True,
                )
                futures[future] = shard_index

            description = f"Encode {len(task_shards)} shards on {workers} workers"
            with tqdm(total=len(tasks), desc=description) as progress:
                for future in concurrent.futures.as_completed(futures):
                    shard_index = futures[future]
                    result = future.result()
                    results_by_index[shard_index] = result
                    progress.update(result["episodes"])

        results = [results_by_index[index] for index in range(len(task_shards))]
        _aggregate_shards(results, output, repo_id)
        succeeded = True
        return results, temporary_root
    finally:
        if not succeeded:
            print(f"Parallel export shards kept for diagnosis: {temporary_root}", flush=True)


def export_dataset(
    source_root,
    output,
    repo_id="local/so101_cf",
    horizon=50,
    target_fps=30,
    use_videos=True,
    encoder_threads=4,
    workers=1,
    episodes_per_shard=24,
):
    source_root = Path(source_root)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a fresh export directory: {output}")
    if target_fps < 1 or horizon < 1:
        raise ValueError("target_fps and horizon must be positive")
    if workers < 1 or encoder_threads < 1 or episodes_per_shard < 1:
        raise ValueError("workers, encoder_threads, and episodes_per_shard must be positive")
    info = json.loads((source_root / "meta/info.json").read_text())
    anchors = [
        json.loads(line)
        for line in (source_root / "meta/anchors.jsonl").read_text().splitlines()
        if line
    ]
    nominal_records = [
        json.loads(line)
        for line in (source_root / "meta/nominal_episodes.jsonl").read_text().splitlines()
        if line
    ]
    tasks = _build_export_tasks(source_root, anchors, nominal_records)
    if not tasks:
        raise ValueError("No successful nominal or counterfactual trajectories to export")

    temporary_root = None
    if workers == 1:
        results = [
            _write_shard(
                str(source_root),
                str(output),
                repo_id,
                tasks,
                info,
                horizon,
                target_fps,
                use_videos,
                encoder_threads,
            )
        ]
    else:
        results, temporary_root = _run_parallel_export(
            source_root,
            output,
            repo_id,
            tasks,
            info,
            horizon,
            target_fps,
            use_videos,
            encoder_threads,
            min(workers, len(tasks)),
            episodes_per_shard,
        )

    _write_export_metadata(
        output,
        source_root,
        repo_id,
        info,
        anchors,
        results,
        horizon,
        target_fps,
        use_videos,
        len(nominal_records),
    )
    if temporary_root is not None:
        shutil.rmtree(temporary_root)
    for result in results:
        local_manifest = Path(result["root"]) / "meta/cf_samples.shard.jsonl"
        if local_manifest != output / "meta/cf_samples.shard.jsonl":
            continue
        local_manifest.unlink()
    episode_count = sum(result["episodes"] for result in results)
    frame_count = sum(result["frames"] for result in results)
    print(
        f"Exported {episode_count} full branch episodes / {frame_count} frames "
        f"at {target_fps} Hz to {output}",
        flush=True,
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/cf_nominal"))
    parser.add_argument("--output", type=Path, default=Path("data/lerobot_cf"))
    parser.add_argument("--repo-id", default="local/so101_cf")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-storage", choices=("video", "image"), default="video")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="independent trajectory/video export workers (default: up to 4)",
    )
    parser.add_argument(
        "--episodes-per-shard",
        type=int,
        default=24,
        help="episodes per temporary LeRobot shard; smaller avoids costly repeated video concatenation",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=2,
        help="H.264 codec threads per camera, per worker",
    )
    args = parser.parse_args()
    export_dataset(
        args.source,
        args.output,
        args.repo_id,
        args.chunk_size,
        args.fps,
        use_videos=args.image_storage == "video",
        encoder_threads=args.encoder_threads,
        workers=args.workers,
        episodes_per_shard=args.episodes_per_shard,
    )


if __name__ == "__main__":
    main()
