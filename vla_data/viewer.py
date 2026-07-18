"""Render a recorded SO101 language episode as a side-by-side MP4."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def _records(root: Path) -> list[dict]:
    path = root / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def list_episodes(root: Path) -> None:
    print("episode  split             task             frames  instruction")
    for record in _records(root):
        print(
            f"{record['episode_index']:>7}  {record['split']:<16}  "
            f"{record['task']:<15}  {record['num_frames']:>6}  "
            f"{record['language_instruction']}"
        )


def render_episode(root: Path, episode_index: int, output: Path) -> dict:
    records = _records(root)
    matches = [record for record in records if record["episode_index"] == episode_index]
    if not matches:
        raise IndexError(f"episode {episode_index} not found; valid range is 0..{len(records) - 1}")
    record = matches[0]
    episode_path = root / record["file"]
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(info["fps"])
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required; install it with: sudo apt install ffmpeg")

    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(episode_path, allow_pickle=False) as episode:
        overhead = episode["observation.images.overhead"]
        wrist = episode["observation.images.wrist"]
        if overhead.shape != wrist.shape:
            raise ValueError(f"camera shapes differ: {overhead.shape} vs {wrist.shape}")
        frames = np.concatenate([overhead, wrist], axis=2)
        n, height, width, channels = frames.shape
        if channels != 3 or frames.dtype != np.uint8:
            raise ValueError("viewer expects uint8 RGB camera frames")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        if process.stdin is None:
            raise RuntimeError("failed to open ffmpeg input")
        try:
            process.stdin.write(frames.tobytes())
            process.stdin.close()
            return_code = process.wait()
        except BrokenPipeError as exc:
            process.wait()
            raise RuntimeError("ffmpeg failed while encoding the episode") from exc
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with code {return_code}")

    print(f"episode:     {episode_index}")
    print(f"task:        {record['task']}")
    print(f"instruction: {record['language_instruction']}")
    print(f"task spec:   {json.dumps(record['task_spec'], ensure_ascii=False)}")
    print(f"frames:      {n} at {fps} FPS ({n / fps:.2f} s)")
    print(f"video:       {output}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true", help="list episodes without rendering")
    parser.add_argument("--play", action="store_true", help="open the rendered MP4 with ffplay")
    args = parser.parse_args()
    root = args.dataset.resolve()
    if args.list:
        list_episodes(root)
        return
    output = (
        args.output.resolve()
        if args.output is not None
        else (Path.cwd() / "outputs" / f"demo_episode_{args.episode:06d}.mp4")
    )
    render_episode(root, args.episode, output)
    if args.play:
        ffplay = shutil.which("ffplay")
        if ffplay is None:
            raise RuntimeError("ffplay is not installed; open the printed MP4 path manually")
        subprocess.run([ffplay, "-autoexit", str(output)], check=True)


if __name__ == "__main__":
    main()
