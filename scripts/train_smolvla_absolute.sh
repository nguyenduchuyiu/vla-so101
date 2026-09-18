#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Activate the server's LeRobot environment before running this script.
# 83 frames at 50 Hz cover 1.66 s; 17 executed frames cover 0.34 s.
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/so101_nominal_absolute_50hz \
  --dataset.root=data/lerobot_so101_absolute_50hz \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.use_amp=true \
  --policy.chunk_size=83 \
  --policy.n_action_steps=17 \
  --batch_size=4 \
  --steps=20000 \
  --num_workers=4 \
  --save_freq=2000 \
  --env_eval_freq=0 \
  --output_dir=runs/smolvla_absolute_20k \
  --job_name=smolvla_so101_absolute_50hz
