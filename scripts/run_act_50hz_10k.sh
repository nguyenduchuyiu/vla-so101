#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source .venv-smolvla/bin/activate

mkdir -p logs
exec > >(tee -a logs/act_cf_50hz_10k.log) 2>&1

echo "[$(date --iso-8601=seconds)] exporting raw 50 Hz LeRobot dataset"
python -m smolvla_cf.export \
  --source data/cf_nominal \
  --output data/lerobot_cf_50hz \
  --repo-id local/so101_cf_50hz \
  --chunk-size 83 \
  --fps 50 \
  --image-storage video \
  --workers 4 \
  --episodes-per-shard 24 \
  --encoder-threads 2

echo "[$(date --iso-8601=seconds)] training ACT for 10k optimizer steps"
python -m smolvla_cf.train_act \
  --data data/lerobot_cf_50hz \
  --split all \
  --device cuda \
  --batch-size 16 \
  --num-workers 4 \
  --chunk-size 83 \
  --steps 10000 \
  --lr 1e-5 \
  --backbone-lr 1e-5 \
  --imagenet-backbone \
  --execute-steps 1 \
  --temporal-ensemble-coeff 0.01 \
  --save-every 1000 \
  --output runs/act_cf_50hz_10k

echo "[$(date --iso-8601=seconds)] pipeline complete"
