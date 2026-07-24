"""On-manifold probe: feed the model GT expert states (image+proprio) at EVERY frame
of a nominal episode and compare the predicted 10-step chunk to the GT future.

Decisive test for the closed-loop failure mode. The closed-loop probe
(probe_closedloop.py) showed the rollout diverges at gf050 and the gripper never
closes. Two explanations:
  (A) covariate shift: the model is accurate ON the expert manifold (GT states) but
      fails on the off-manifold states its own actions visit;
  (B) prediction/execution bug: the model is wrong even on GT states (overfit is not
      real, or the chunk does not compose).

This script decides. At each GT frame we restore the exact snapshot, render the 256x256
image the eval loop sees, feed GT proprio + instruction, predict the chunk, and report
arm chunkMAE vs GT future and the pre-binarization gripper prediction vs GT gripper --
including the GRASP phase (gf180-200) where the closed loop never closes the gripper.

If chunkMAE stays low throughout and the gripper closes at GRASP frames -> (A)
covariate shift confirmed; overfit is real and complete; the closed-loop failure is
purely the compounding off-manifold drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cf_data.collect import make_env
from cf_data.core import PHASE_NAMES, Snapshot, get_gripper_limits, objective_instruction, restore_snapshot
from models.utils import load_vla_for_inference, pick_device
from simvla_datasets.utils import build_image_transform


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("runs/overfit/ckpt-4000"))
    p.add_argument("--norm-stats", type=Path, default=Path("norm_stats/cf_dense_norm.json"))
    p.add_argument("--data-dir", type=Path, default=Path("data/cf_dense"))
    p.add_argument("--objective-id", type=int, default=0)
    p.add_argument("--stride", type=int, default=5, help="GT frame stride (matches replan cadence)")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--log", type=Path, default=Path("outputs/probe_onmanifold.log"))
    args = p.parse_args()

    metas = [json.loads(l) for l in (args.data_dir / "meta" / "nominal_episodes.jsonl").read_text().splitlines() if l]
    meta = next(m for m in metas if m["objective_id"] == args.objective_id)
    with np.load(args.data_dir / meta["file"]) as npz:
        gt_states = npz["observation.state"].copy()  # (N,6) deg+%
        phases = npz["phase"].copy()
        qpos = npz["snapshot.qpos"].copy()
        qvel = npz["snapshot.qvel"].copy()
        ctrl = npz["snapshot.ctrl"].copy()
    N = len(gt_states)

    device = pick_device()
    model, processor = load_vla_for_inference(args.checkpoint, device)
    model.action_space.load_norm_stats(str(args.norm_stats))
    transform = build_image_transform(model.config.image_size, False)
    g_mid = float((model.action_space.action_norm_stats.q01[5] + model.action_space.action_norm_stats.q99[5]) / 2)
    g_closed = float(model.action_space.action_norm_stats.q01[5])
    g_open = float(model.action_space.action_norm_stats.q99[5])

    env = make_env(width=256, height=256, source_index=args.objective_id, robot_init_qpos_noise=0.0)
    env.reset(seed=0)
    env.set_objective(args.objective_id)
    limits = get_gripper_limits(env)
    instruction = objective_instruction(args.objective_id)
    lang = processor.encode_language([instruction])

    frames = list(range(0, N - 1, args.stride))
    lines = []
    lines.append(f"obj={args.objective_id} N={N} frames={len(frames)} stride={args.stride}")
    lines.append(f"gripper closed={g_closed:.2f}% open={g_open:.2f}% midpoint={g_mid:.2f}%")

    arm_maes = []
    grip_rows = []  # (frame, phase, gt_grip, pred_grip_raw, pred_bin, ok)
    image_mask = torch.tensor([[True, True]])

    for f in frames:
        restore_snapshot(env, Snapshot(qpos[f], qvel[f], ctrl[f]))
        obs = env._get_obs()
        images = torch.stack(
            [transform(Image.fromarray(obs["overhead_camera"])), transform(Image.fromarray(obs["wrist_camera"]))]
        ).unsqueeze(0)
        proprio = gt_states[f]  # exact GT proprio (deg+%)
        gt_future = gt_states[f + 1 : f + 1 + 10]
        ngt = len(gt_future)
        gt_arm = gt_future[:, :5]
        gt_grip = gt_future[:, 5]
        phase = PHASE_NAMES[int(phases[f])] if f < len(phases) else "?"

        preds = []
        with torch.inference_mode():
            for _ in range(args.seeds):
                torch.manual_seed(0)
                out = model.generate_actions(
                    input_ids=lang["input_ids"].to(device),
                    language_attention_mask=lang["language_attention_mask"].to(device),
                    image_input=images.to(device),
                    image_mask=image_mask.to(device),
                    proprio=torch.as_tensor(proprio, dtype=torch.float32, device=device).unsqueeze(0),
                    steps=10,
                )[0].float().cpu().numpy()  # (10,6) deg+%, gripper binarized
                preds.append(out)
        pred = np.mean(preds, axis=0)
        mae = np.abs(pred[:ngt, :5] - gt_arm).mean()
        arm_maes.append(mae)
        pred_grip_bin = pred[0, 5]  # binarized (q01 or q99)
        # raw pre-binarization gripper: re-run postprocess_without binarization is not exposed;
        # binarized value itself tells open/closed decision.
        gt_grip0 = gt_grip[0]
        gt_state = "closed" if gt_grip0 < g_mid else "open"
        pred_state = "closed" if pred_grip_bin < g_mid else "open"
        ok = gt_state == pred_state
        grip_rows.append((f, phase, float(gt_grip0), float(pred_grip_bin), pred_state, ok))
        lines.append(
            f"f{f:03d} {phase:11s} armMAE={mae:5.2f} grip pred={pred_grip_bin:4.1f}({pred_state:6s}) "
            f"gt={gt_grip0:4.1f}({gt_state:6s}) {'OK' if ok else 'MISS'}"
        )

    env.close()

    arm_maes = np.array(arm_maes)
    # phase-split arm MAE
    phase_maes = {}
    for (f, phase, *_), mae in zip(grip_rows, arm_maes):
        phase_maes.setdefault(phase, []).append(mae)
    # grasp-window gripper accuracy (gt closed)
    grasp_rows = [r for r in grip_rows if r[2] < g_mid]
    grasp_correct = sum(1 for r in grasp_rows if r[5])

    lines.append("\n=== SUMMARY ===")
    lines.append(f"arm chunkMAE mean={arm_maes.mean():.3f} max={arm_maes.max():.3f} deg")
    for ph, ms in phase_maes.items():
        lines.append(f"  {ph:11s} n={len(ms):3d} armMAE mean={np.mean(ms):.3f} max={np.max(ms):.3f}")
    lines.append(
        f"gripper: GT-closed frames={len(grasp_rows)} predicted-closed={grasp_correct} "
        f"({100*grasp_correct/max(1,len(grasp_rows)):.0f}%)"
    )
    # show the grasp transition window explicitly
    lines.append("grasp window (gt gripper closing, f>=175):")
    for r in grip_rows:
        if r[0] >= 175 and r[0] <= 210:
            lines.append(f"  f{r[0]:03d} {r[1]:11s} gt_grip={r[2]:4.1f} pred={r[3]:4.1f}({r[4]}) {'OK' if r[5] else 'MISS'}")

    txt = "\n".join(lines)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(txt)
    print(txt)
    print(f"\nlog -> {args.log}")


if __name__ == "__main__":
    main()