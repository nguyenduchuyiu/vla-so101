# vla-so101 — DAgger pipeline

## Run

```bash
python run_dagger.py dagger_config.yaml
```

Runs the full DAgger loop (expert rollout → train → closed-loop rollout → off-manifold collect → re-supervise → merge → train) for `run.iterations` iterations. Edit `dagger_config.yaml` to change settings; do not pass CLI flags.

`launch.num_processes: auto` enables DDP over every free GPU; an integer caps the
world size. The pipeline selects devices by VRAM use and utilization, unless `CUDA_VISIBLE_DEVICES` or
`launch.gpu_ids` pins them explicitly. Counterfactual anchors are sharded across
ranks without splitting an anchor's branches.

Checkpoints land in `runs/dagger/iterNN/`, norm stats in `norm_stats/iterNN_norm.json`.

## Test (eval a checkpoint, writes an mp4)

```bash
python evaluate_so101.py \
  --checkpoint runs/dagger/iter02/ckpt-203 \
  --norm_stats norm_stats/iter02_norm.json \
  --objective_id 0 \
  --target_id 1 \
  --output outputs/dagger_iter02_ck203_obj0_tgt1.mp4
```

`--objective_id`: 0=red, 1=blue, 2=green, 3=yellow, 4=purple. `--target_id`: 0=white, 1=black, 2=orange. Prints `success / is_grasped / obj_to_target_dist`.

View the video: `open outputs/dagger_iter02_ck203_obj0.mp4`
