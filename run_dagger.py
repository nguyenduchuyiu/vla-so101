"""End-to-end DAgger pipeline for vla-so101, driven by dagger_config.yaml.

Outer loop (one "iteration" = rollout + collect + train), run `run.iterations` times:

  iter 0  : FRESH expert rollout -- cf_data/collect.py runs the oracle from each
            scene's S0 (on-manifold expert trajectories), then cf_data/build.py
            builds the CF anchor dataset. No prebuilt dataset is reused. Train E
            epochs on it, resuming from run.base_ckpt if set (else scratch).
  iter t>=1:
    1. rollout  : dagger_collect.py runs the model closed-loop from each S0 AND a
                  GT oracle pass + teacher-forcing pass to locate the off-manifold
                  threshold t* (first control step where closed-loop trackMAE >
                  offmanifold.track_mae_threshold_deg). Only frames [t*:] (the
                  states the policy drifts to but the expert never visits) are
                  saved as DAgger anchors; on-manifold frames are skipped.
    2. collect  : cf_data/build.py --dagger rolls a FRESH oracle from each
                  off-manifold anchor snapshot -> expert future chunk.
    3. merge    : symlink bootstrap + all dagger dirs into one cf_balanced dataset
                  (DAgger accumulates: D = D_expert U D_t).
    4. train    : train_smolvlm.py --epochs E on the merged set, resuming from the
                  latest checkpoint, with a tqdm bar.

Usage: python run_dagger.py [dagger_config.yaml]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def run_cmd(
    cmd: list[str],
    log_path: Path | None,
    label: str,
    env: dict[str, str] | None = None,
) -> None:
    """Run a subprocess; if log_path is None inherit the terminal (live tqdm). Fail fast."""
    print(f"\n[dagger] {label}")
    print(f"[dagger] $ {' '.join(map(str, cmd))}")
    if log_path is None:
        proc = subprocess.run(cmd, env=env)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed (rc={proc.returncode}); see {log_path or 'terminal'}")


def latest_ckpt(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("ckpt-*"), key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        raise RuntimeError(f"no checkpoint found in {run_dir}")
    return ckpts[-1]


def training_environment(launch: dict) -> dict[str, str]:
    """Restrict child processes to the requested, freest NVIDIA GPUs.

    An existing CUDA_VISIBLE_DEVICES always wins. Otherwise ``gpu_ids`` can pin
    an explicit list; with ``auto_select_gpus`` (the default), nvidia-smi is
    queried and devices are ranked by free memory, then lower utilization.
    """
    env = os.environ.copy()
    requested = launch.get("num_processes", "auto")
    auto_count = isinstance(requested, str) and requested.lower() == "auto"
    if not auto_count:
        count = int(requested)
        if count < 1:
            raise ValueError("launch.num_processes must be positive or 'auto'")
    visible_is_set = "CUDA_VISIBLE_DEVICES" in env
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    explicit = launch.get("gpu_ids")

    if visible_is_set:
        chosen = [value.strip() for value in visible.split(",") if value.strip()]
        if not chosen:
            count = 1 if auto_count else count
            if count > 1:
                raise ValueError(
                    "CUDA_VISIBLE_DEVICES disables CUDA, but "
                    f"launch.num_processes={count}"
                )
            print("[dagger] CUDA disabled by CUDA_VISIBLE_DEVICES")
            launch["_resolved_num_processes"] = count
            return env
        count = len(chosen) if auto_count else count
        if len(chosen) < count:
            raise ValueError(
                f"CUDA_VISIBLE_DEVICES exposes {len(chosen)} GPU(s), but "
                f"launch.num_processes={count}"
            )
        print(f"[dagger] respecting CUDA_VISIBLE_DEVICES={visible}")
        launch["_resolved_num_processes"] = count
        return env

    if explicit is not None:
        chosen = [str(value) for value in explicit]
        count = len(chosen) if auto_count else count
        if not chosen:
            raise ValueError("launch.gpu_ids cannot be empty")
        if len(chosen) != count:
            raise ValueError(
                f"launch.gpu_ids must contain exactly {count} entries, got {chosen}"
            )
        print(f"[dagger] using configured GPU IDs: {','.join(chosen)}")
    elif launch.get("auto_select_gpus", True):
        query = [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(query, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            if auto_count or count == 1:
                count = 1
                print(
                    "[dagger] nvidia-smi unavailable; leaving device selection "
                    "to PyTorch (CPU/MPS/CUDA default)"
                )
                launch["_resolved_num_processes"] = count
                return env
            raise RuntimeError(
                "multi-GPU automatic selection requires a working nvidia-smi; "
                "set launch.gpu_ids or export CUDA_VISIBLE_DEVICES"
            ) from exc
        devices = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                continue
            index, free_mb, used_mb, utilization = map(int, fields)
            devices.append((index, free_mb, used_mb, utilization))
        max_used = int(launch.get("max_gpu_memory_used_mb", 1024))
        max_util = int(launch.get("max_gpu_utilization", 20))
        eligible = [
            device
            for device in devices
            if device[2] <= max_used and device[3] <= max_util
        ]
        count = len(eligible) if auto_count else count
        if count < 1:
            raise RuntimeError(
                "no free GPU matched launch.max_gpu_memory_used_mb="
                f"{max_used} and launch.max_gpu_utilization={max_util}"
            )
        if len(eligible) < count:
            raise RuntimeError(
                f"requested {count} GPU process(es), but only {len(eligible)} "
                f"matched the free-GPU thresholds (nvidia-smi reported "
                f"{len(devices)} total)"
            )
        eligible.sort(key=lambda item: (item[2], item[3], -item[1], item[0]))
        selected = eligible[:count]
        chosen = [str(index) for index, _, _, _ in selected]
        details = ", ".join(
            f"GPU {index} ({free_mb} MiB free, {used_mb} MiB used, {util}% util)"
            for index, free_mb, used_mb, util in selected
        )
        print(f"[dagger] auto-selected {details}")
    else:
        if auto_count:
            count = 1
        launch["_resolved_num_processes"] = count
        return env

    env["CUDA_VISIBLE_DEVICES"] = ",".join(chosen)
    launch["_resolved_num_processes"] = count
    return env


def merge_datasets(sources: list[Path], accum: Path, horizon: int) -> Path:
    """Symlink bootstrap + dagger dirs into one cf_balanced dataset under `accum`.

    Episode/cf_anchor filenames are prefixed with the source dir name (episode
    counters restart per collect run, so names collide across sources). anchors.jsonl
    paths are rewritten to the symlinked locations. Returns the merged cf_balanced.json path.
    """
    if accum.exists():
        shutil.rmtree(accum)
    (accum / "episodes").mkdir(parents=True)
    (accum / "cf_anchors").mkdir()
    (accum / "meta").mkdir()

    lines: list[str] = []
    num_anchors = 0
    num_branches = 0
    for src in sources:
        src = src.resolve()
        tag = src.name
        for ep in (src / "episodes").glob("*.npz"):
            (accum / "episodes" / f"{tag}_{ep.name}").symlink_to(ep)
        for an in (src / "cf_anchors").glob("*.npz"):
            (accum / "cf_anchors" / f"{tag}_{an.name}").symlink_to(an)
        for line in (src / "meta" / "anchors.jsonl").read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            rec = json.loads(line)
            rec["nominal_episode_path"] = f"episodes/{tag}_{Path(rec['nominal_episode_path']).name}"
            rec["cf_path"] = f"cf_anchors/{tag}_{Path(rec['cf_path']).name}"
            num_anchors += 1
            num_branches += len(rec["branches"])
            lines.append(json.dumps(rec, ensure_ascii=False))

    if not lines:
        raise RuntimeError(f"merge produced 0 anchors from {sources}")
    (accum / "meta" / "anchors.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "dataset_name": "cf_balanced",
        "data_dir": str(accum),
        "datalist": [{"sampler": "cf_balanced"}],
        "dataset_root": str(accum),
        "anchors_file": "meta/anchors.jsonl",
        "horizon_stored": horizon,
        "disable_image_augmentation": True,
        "preserve_order": False,
        "num_anchors": num_anchors,
        "num_branches": num_branches,
        "sources": [str(s) for s in sources],
    }
    out = accum / "meta" / "cf_balanced.json"
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[dagger] merged {num_anchors} anchors / {num_branches} branches from {len(sources)} sources -> {accum}")
    return out


def build_train_cmd(cfg: dict, train_meta: Path, output_dir: Path, resume_ckpt: Path | None,
                    norm_path: Path, finetune: bool = False) -> list[str]:
    t, ln, m = cfg["train"], cfg["launch"], cfg["train"]["model"]
    num_processes = int(ln.get("_resolved_num_processes", ln["num_processes"]))
    lr = t["finetune_learning_rate"] if finetune else t["learning_rate"]
    # YAML parses unquoted no/yes/on/off as bools; accelerate wants the literal
    # string "no"/"fp16"/"bf16"/"fp8". Map the bool trap back to "no".
    mp = ln["mixed_precision"]
    mp = "no" if mp is False else str(mp)
    cmd = [
        "accelerate", "launch",
        "--num_processes", str(num_processes),
        "--main_process_port", str(ln["main_process_port"]),
        "--mixed_precision", mp,
        "train_smolvlm.py",
        "--output_dir", str(output_dir),
        "--train_metas_path", str(train_meta),
        "--smolvlm_model_path", m["smolvlm_model_path"],
        "--action_mode", m["action_mode"],
        "--batch_size", str(t["batch_size"]),
        "--num_workers", str(t["num_workers"]),
        "--learning_rate", str(lr),
        "--learning_coef", str(t["learning_coef"]),
        "--weight_decay", str(t["weight_decay"]),
        "--max_grad_norm", str(t["max_grad_norm"]),
        "--warmup_steps", str(t["warmup_steps"]),
        "--freeze_steps", str(t["freeze_steps"]),
        "--min_lr_ratio", str(t["min_lr_ratio"]),
        "--save_interval", str(t["save_interval"]),
        "--log_interval", str(t["log_interval"]),
        "--norm_stats_path", str(norm_path),
        "--epochs", str(t["epochs"]),
        "--seed", str(t["train_seed"]),
        "--num_actions", str(m["num_actions"]),
        "--num_views", str(m["num_views"]),
        "--image_size", str(m["image_size"]),
        "--hidden_size", str(m["hidden_size"]),
        "--depth", str(m["depth"]),
        "--num_heads", str(m["num_heads"]),
        "--lora_rank", str(m["lora_rank"]),
        "--lora_alpha", str(m["lora_alpha"]),
        "--lora_dropout", str(m["lora_dropout"]),
    ]
    if num_processes > 1:
        cmd.insert(2, "--multi_gpu")
    if t["use_cosine_decay"]:
        cmd.append("--use_cosine_decay")
    if t["gradient_checkpointing"]:
        cmd.append("--gradient_checkpointing")
    if resume_ckpt is not None:
        cmd += ["--models", str(resume_ckpt), "--resume"]
    return cmd


def compute_norm_cmd(data_dir: Path, out_json: Path, max_samples: int) -> list[str]:
    """cf_data.compute_norm_stats over meta/cf_balanced.json in `data_dir`."""
    return [
        sys.executable, "-m", "cf_data.compute_norm_stats",
        "--data-dir", str(data_dir),
        "--output", str(out_json),
        "--max-samples", str(max_samples),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path, nargs="?", default=Path("dagger_config.yaml"))
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run, boot, roll, offm = cfg["run"], cfg["bootstrap"], cfg["rollout"], cfg["offmanifold"]
    norm_max = cfg["norm"]["max_samples"]
    child_env = training_environment(cfg["launch"])

    out_root, data_root = Path(run["out_root"]), Path(run["data_root"])
    norm_dir = Path(cfg["paths"]["norm_stats"])
    if run["overwrite"]:
        for d in (out_root, data_root):
            if d.exists():
                shutil.rmtree(d)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_dir = data_root / "bootstrap"
    accum_dir = data_root / "accum"
    dagger_dirs: list[Path] = []
    base_ckpt = Path(run["base_ckpt"]) if run["base_ckpt"] else None
    prev_ckpt: Path | None = base_ckpt
    prev_norm: Path | None = None  # norm the prev ckpt was trained with -> next iter's inference norm

    for it in range(run["iterations"]):
        run_dir = out_root / f"iter{it:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}\n[dagger] iteration {it}/{run['iterations'] - 1}  ->  {run_dir}\n{'='*70}")

        if it == 0:
            # iter 0: fresh expert rollout (on-manifold) via collect.py + build.py (normal, CF)
            collect_cmd = [
                sys.executable, "-m", "cf_data.collect",
                "--out", str(bootstrap_dir),
                "--scenes", str(boot["scenes"]),
                "--seed", str(boot["seed"]),
                "--width", str(boot["width"]),
                "--height", str(boot["height"]),
                "--robot-noise", str(boot["robot_noise"]),
                "--overwrite",
            ]
            run_cmd(collect_cmd, None, "iter 0: expert rollout (collect.py)", child_env)

            bcmd = [
                sys.executable, "-m", "cf_data.build",
                "--in", str(bootstrap_dir),
                "--horizon", str(boot["build"]["horizon"]),
                "--anchor-stride", str(boot["build"]["anchor_stride"]),
                "--seed", str(boot["build"]["seed"]),
                "--overwrite",
            ]
            if boot["build"]["max_anchors"] is not None:
                bcmd += ["--max-anchors", str(boot["build"]["max_anchors"])]
            run_cmd(bcmd, None, "iter 0: build CF anchors (build.py normal)", child_env)
            train_meta = bootstrap_dir / "meta" / "cf_balanced.json"
        else:
            # 1. rollout model closed-loop + locate off-manifold threshold t*
            dagger_data = data_root / f"dagger_iter{it:02d}"
            roll_seed = roll["seed"]  # same scenes every iter (DAgger: revisit trained distribution)
            collect_cmd = [
                sys.executable, "dagger_collect.py",
                "--checkpoint", str(prev_ckpt),
                "--norm-stats", str(prev_norm),
                "--out", str(dagger_data),
                "--scenes", str(roll["scenes"]),
                "--seed", str(roll_seed),
                "--execute-steps", str(roll["execute_steps"]),
                "--max-replans", str(roll["max_replans"]),
                "--width", str(roll["width"]),
                "--height", str(roll["height"]),
                "--policy-seed", str(roll["policy_seed"] + it),
                "--track-mae-threshold", str(offm["track_mae_threshold_deg"]),
                "--persist-steps", str(offm["persist_steps"]),
                "--min-offmanifold-frames", str(offm["min_offmanifold_frames"]),
                "--tf-stride", str(offm["tf_stride"]),
                "--overwrite",
            ]
            # Live terminal (log_path=None) so the tqdm replan bar shows; the
            # per-objective t*/off-manifold prints are the persisted diagnostics.
            run_cmd(collect_cmd, None, f"iter {it}: rollout + off-manifold threshold", child_env)

            # 2. fresh-oracle supervision on the off-manifold anchors
            build_cmd = [
                sys.executable, "-m", "cf_data.build",
                "--in", str(dagger_data),
                "--dagger",
                "--horizon", str(cfg["dagger_build"]["horizon"]),
                "--overwrite",
            ]
            run_cmd(build_cmd, run_dir / "build.log", f"iter {it}: build fresh-oracle dagger anchors", child_env)
            dagger_dirs.append(dagger_data)

            # 3. accumulate bootstrap + all dagger dirs
            train_meta = merge_datasets([bootstrap_dir] + dagger_dirs, accum_dir, cfg["dagger_build"]["horizon"])

        # 4. recompute norm stats from the actual built pool, then train E epochs.
        #    iter 0 uses base learning_rate; iter>=1 (DAgger finetune, warm-start) uses finetune_learning_rate.
        data_dir = train_meta.parent.parent
        norm_path = norm_dir / f"iter{it:02d}_norm.json"
        run_cmd(compute_norm_cmd(data_dir, norm_path, norm_max), run_dir / "norm.log",
                f"iter {it}: compute norm stats from {data_dir}", child_env)

        train_cmd = build_train_cmd(cfg, train_meta, run_dir, prev_ckpt, norm_path, finetune=(it >= 1))
        run_cmd(train_cmd, None, f"iter {it}: train {cfg['train']['epochs']} epochs on {train_meta} (lr={'finetune' if it>=1 else 'base'})", child_env)

        prev_ckpt = latest_ckpt(run_dir)
        prev_norm = norm_path
        print(f"[dagger] iter {it} done; latest checkpoint -> {prev_ckpt}")

    print(f"\n[dagger] pipeline complete. final checkpoint: {prev_ckpt}")


if __name__ == "__main__":
    main()
