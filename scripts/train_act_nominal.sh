#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv-smolvla/bin/activate
mkdir -p logs
python -m smolvla_cf.train_act \
  --data data/lerobot_nominal_spatial --split train \
  --device cuda --batch-size 16 --num-workers 4 \
  --chunk-size 83 --steps 50000 \
  --lr 1e-5 --backbone-lr 1e-5 --imagenet-backbone \
  --execute-steps 1 --temporal-ensemble-coeff 0.01 \
  --save-every 5000 --output runs/act_nominal_spatial \
  2>&1 | tee logs/act_nominal_spatial.log
