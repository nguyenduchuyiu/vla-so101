#!/bin/bash

set -e

BATCH_SIZE=${1:-1}
OUTPUT_DIR=${2:-./runs/simvla_so101_small}
RESUME_CKPT=${3:-""}
ITERS=${ITERS:-20000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
SAMPLES_PER_EPISODE=${SAMPLES_PER_EPISODE:-32}
ACTION_MODE=${ACTION_MODE:-so101_joint}
NORM_STATS_PATH=${NORM_STATS_PATH:-./norm_stats/so101_observable_norm.json}

export CUDA_VISIBLE_DEVICES=0

SO101_DATA_DIR="../data/so101_counterfactual_observable"
TRAIN_METAS_PATH=${TRAIN_METAS_PATH:-./datasets/metas/so101_observable_train.json}
SMOLVLM_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"

if [ ! -f "$TRAIN_METAS_PATH" ] || [ ! -f "$NORM_STATS_PATH" ]; then
    python prepare_so101_data.py \
        --data_dir "$SO101_DATA_DIR" \
        --meta_output "$TRAIN_METAS_PATH" \
        --stats_output "$NORM_STATS_PATH"
fi

ARGS="--output_dir ${OUTPUT_DIR} \
    --train_metas_path ${TRAIN_METAS_PATH} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode ${ACTION_MODE} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --learning_coef 0.0 \
    --num_actions 10 \
    --num_views 2 \
    --samples_per_episode ${SAMPLES_PER_EPISODE} \
    --freeze_vlm \
    --iters ${ITERS} \
    --warmup_steps 0 \
    --freeze_steps 0 \
    --hidden_size 768 \
    --depth 12 \
    --num_heads 12 \
    --num_workers 0 \
    --save_interval ${SAVE_INTERVAL} \
    --log_interval 20 \
    --image_size 384 \
    --norm_stats_path ${NORM_STATS_PATH} \
    --max_grad_norm 1.0"

if [ -n "$RESUME_CKPT" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
fi

PYTORCH_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=1 \
    --main_process_port 29504 \
    --mixed_precision bf16 \
    train_smolvlm.py ${ARGS}
