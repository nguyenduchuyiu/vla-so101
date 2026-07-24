"""Counterfactual branch-point demo video.

From a single shared start state (S0), the 5 nominal episodes (one per objective
cube) all begin identically then diverge as the oracle reaches for a different cube.
Re-renders each frame at 256x256 from the saved per-frame snapshots (the stored
overhead is only 96x96) and composes a 5-panel side-by-side video with the language
instruction labelled on each panel.

This is the counterfactual / branch-point idea: same (scene, robot, image, proprio),
swap the language instruction, get 5 different futures.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cf_data.collect import make_env
from cf_data.core import OBJECTIVE_COLORS, Snapshot, objective_instruction, restore_snapshot

PANEL_COLORS = {
    "red": (205, 60, 60),
    "blue": (60, 110, 210),
    "green": (60, 170, 75),
    "yellow": (205, 175, 55),
    "purple": (155, 75, 195),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cf_dense"))
    p.add_argument("--out", type=Path, default=Path("outputs/figs/cf_branch_demo.mp4"))
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--max-frames", type=int, default=0, help="0 = min episode length")
    args = p.parse_args()

    metas = [json.loads(l) for l in (args.data_dir / "meta" / "nominal_episodes.jsonl").read_text().splitlines() if l]
    metas = sorted(metas, key=lambda m: m["objective_id"])[:5]
    episodes = [np.load(args.data_dir / m["file"]) for m in metas]
    n_min = min(len(e["snapshot.qpos"]) for e in episodes)
    n_max = args.max_frames if args.max_frames > 0 else n_min
    n_max = min(n_max, n_min)

    env = make_env(width=args.size, height=args.size, source_index=0, robot_init_qpos_noise=0.0)
    env.reset(seed=0)

    S = args.size
    strip_h, gap, title_h = 34, 6, 40
    W = 5 * S + 4 * gap
    H = title_h + strip_h + S

    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18) if \
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf").exists() else ImageFont.load_default()
    font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22) if \
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf").exists() else ImageFont.load_default()
    font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16) if \
        Path("/System/Library/Fonts/Helvetica.ttc").exists() else ImageFont.load_default()

    # pre-render every panel trajectory into RAM (stride applied)
    frames_idx = list(range(0, n_max, args.stride))
    panels = []  # list of (N_frames, S, S, 3) per objective
    for ei, e in enumerate(episodes):
        qpos, qvel, ctrl = e["snapshot.qpos"], e["snapshot.qvel"], e["snapshot.ctrl"].astype(np.float64)
        imgs = []
        for fi, f in enumerate(frames_idx):
            restore_snapshot(env, Snapshot(qpos[f], qvel[f], ctrl[f]))
            obs = env._get_obs()
            imgs.append(obs["overhead_camera"].copy())
            if fi % 40 == 0:
                print(f"  obj {ei} frame {fi}/{len(frames_idx)}")
        panels.append(np.stack(imgs))
    env.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {args.out}")

    for fi in range(len(frames_idx)):
        canvas = Image.new("RGB", (W, H), (15, 15, 15))
        d = ImageDraw.Draw(canvas)
        d.text((W // 2, 8), "Counterfactual branch-point:  same start state, 5 instructions -> 5 oracle futures",
               fill="white", font=font_title, anchor="la")
        for oi, m in enumerate(metas):
            color = OBJECTIVE_COLORS[oi]
            x0 = oi * (S + gap)
            # color strip with objective name
            d.rectangle([x0, title_h, x0 + S, title_h + strip_h], fill=PANEL_COLORS[color])
            d.text((x0 + S // 2, title_h + strip_h // 2), color.upper(),
                   fill="white", font=font, anchor="mm")
            # image
            img = Image.fromarray(panels[oi][fi])
            canvas.paste(img, (x0, title_h + strip_h))
        frame = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()
    print(f"saved {args.out}  ({len(frames_idx)} frames, {args.fps} fps, {W}x{H})")
    print("instructions:")
    for m in metas:
        print(f"  {m['objective_color']:6s}: {objective_instruction(m['objective_id'])}")


if __name__ == "__main__":
    main()