#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv-smolvla/bin/activate
mkdir -p logs
device="${1:-mps}"
case "$device" in
  mps) batch_size=1; steps=50000 ;;
  cuda) batch_size=8; steps=10000 ;;
  *) echo "Usage: bash scripts/train_smolvla_nominal.sh [mps|cuda]" >&2; exit 2 ;;
esac
python -m smolvla_cf.train \
  --data data/lerobot_nominal_smolvla_50hz --split train \
  --pretrained lerobot/smolvla_base \
  --device "$device" --batch-size "$batch_size" \
  --chunk-size 50 --execute-steps 8 \
  --steps "$steps" --lr 1e-4 \
  --warmup-steps 1000 --decay-steps 30000 --decay-lr 2.5e-6 \
  --save-every 1000 --output "runs/smolvla_nominal_spatial_$device" \
  2>&1 | tee "logs/smolvla_nominal_spatial_$device.log"
