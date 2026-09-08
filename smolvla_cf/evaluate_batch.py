"""Evaluate one SmolVLA or ACT checkpoint over tasks in multiple scenes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cf_data.collect import make_env
from cf_data.core import OBJECTIVE_COLORS, TARGET_COLORS

from .evaluate import evaluate_episode, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-replans", type=int, default=30)
    parser.add_argument(
        "--execute-steps",
        type=int,
        help="Actions executed per prediction chunk; default comes from checkpoint",
    )
    parser.add_argument("--no-temporal-ensemble", action="store_true")
    parser.add_argument("--temporal-ensemble-coeff", type=float)
    parser.add_argument("--interpolation", choices=["linear", "bspline"], default="linear")
    parser.add_argument("--spline-fps", type=int, default=100)
    parser.add_argument("--save-videos", choices=["none", "all"], default="none")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted matching batch run")
    args = parser.parse_args()
    if args.no_temporal_ensemble and args.temporal_ensemble_coeff is not None:
        raise ValueError("Choose either --no-temporal-ensemble or --temporal-ensemble-coeff")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    args.output.mkdir(parents=True, exist_ok=True)

    run_config = {
        "checkpoint": str(args.checkpoint.resolve()),
        "seeds": args.seeds,
        "max_replans": args.max_replans,
        "execute_steps": args.execute_steps,
        "no_temporal_ensemble": args.no_temporal_ensemble,
        "temporal_ensemble_coeff": args.temporal_ensemble_coeff,
        "interpolation": args.interpolation,
        "spline_fps": args.spline_fps,
        "save_videos": args.save_videos,
    }
    config_path = args.output / "run_config.json"
    if args.resume:
        if not config_path.exists():
            raise FileNotFoundError(f"Cannot resume without {config_path}")
        if json.loads(config_path.read_text()) != run_config:
            raise ValueError("Resume arguments do not match the existing batch run")
    else:
        config_path.write_text(json.dumps(run_config, indent=2) + "\n")

    policy, pre, post, contract = load_checkpoint(args.checkpoint, args.device)
    if args.no_temporal_ensemble and contract.get("policy_family") == "act":
        policy.config.temporal_ensemble_coeff = None
    elif args.temporal_ensemble_coeff is not None and contract.get("policy_family") == "act":
        policy.config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
    shape = policy.config.input_features["observation.images.camera1"].shape
    env = make_env(width=shape[2], height=shape[1], source_index=0, robot_init_qpos_noise=0.0)
    results_path = args.output / "results.jsonl"
    records = [json.loads(line) for line in results_path.read_text().splitlines()] \
        if args.resume and results_path.exists() else []
    completed = {(row["seed"], row["source"], row["target"]) for row in records}
    try:
        with results_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
            total = len(args.seeds) * len(OBJECTIVE_COLORS) * len(TARGET_COLORS)
            for seed in args.seeds:
                for source, source_color in enumerate(OBJECTIVE_COLORS):
                    for target, target_color in enumerate(TARGET_COLORS):
                        if (seed, source, target) in completed:
                            continue
                        stem = f"scene-{seed:06d}_{source_color}-to-{target_color}"
                        video = args.output / "videos" / f"{stem}.mp4"
                        output = video if args.save_videos == "all" else None
                        record = evaluate_episode(
                            policy, pre, post, contract,
                            device=args.device, source=source, target=target, seed=seed,
                            max_replans=args.max_replans, execute_steps=args.execute_steps,
                            interpolation=args.interpolation,
                            spline_fps=args.spline_fps, output=output, env=env,
                        )
                        records.append(record)
                        handle.write(json.dumps(record) + "\n")
                        handle.flush()
                        print(
                            f"[{len(records):02d}/{total}] seed={seed} "
                            f"{source_color}->{target_color}: "
                            f"{'PASS' if record['success'] else 'FAIL'} "
                            f"dist={record['obj_to_target_dist']:.4f}",
                            flush=True,
                        )
    finally:
        env.close()

    by_scene = defaultdict(list)
    by_objective = defaultdict(list)
    for record in records:
        by_scene[str(record["seed"])].append(record["success"])
        key = f"{OBJECTIVE_COLORS[record['source']]}->{TARGET_COLORS[record['target']]}"
        by_objective[key].append(record["success"])
    summary = {
        "checkpoint": str(args.checkpoint),
        "seeds": args.seeds,
        "interpolation": sorted({record["interpolation"] for record in records}),
        "episodes": len(records),
        "successes": sum(record["success"] for record in records),
        "success_rate": float(np.mean([record["success"] for record in records])),
        "mean_final_distance": float(np.mean([record["obj_to_target_dist"] for record in records])),
        "success_rate_by_scene": {key: float(np.mean(value)) for key, value in by_scene.items()},
        "success_rate_by_objective": {
            key: float(np.mean(value)) for key, value in by_objective.items()
        },
        "results": str(results_path),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
