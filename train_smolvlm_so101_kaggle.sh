#!/bin/bash

set -e

# Per-GPU batch size. Two Kaggle T4 GPUs give a global batch size of 4.
BATCH_SIZE=${1:-2}
OUTPUT_DIR=${2:-./runs/simvla_so101_kaggle}
RESUME_CKPT=${3:-""}

ITERS=${ITERS:-50000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
LEARNING_COEF=${LEARNING_COEF:-1.0}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
SMOLVLM_MODEL_PATH=${SMOLVLM_MODEL_PATH:-HuggingFaceTB/SmolVLM-500M-Instruct}
SO101_DATA_DIR=${SO101_DATA_DIR:-../data/so101_counterfactual_observable}
TRAIN_METAS_PATH=${TRAIN_METAS_PATH:-./simvla_datasets/metas/so101_observable_train.json}
NORM_STATS_PATH=${NORM_STATS_PATH:-./norm_stats/so101_observable_norm.json}

export CUDA_VISIBLE_DEVICES=0,1

if [ "${REBUILD_DATA:-0}" = "1" ] || [ ! -f "$TRAIN_METAS_PATH" ] || [ ! -f "$NORM_STATS_PATH" ]; then
    EPISODE_ARGS=()
    if [ -n "${OVERFIT_EPISODE:-}" ]; then
        EPISODE_ARGS=(--episode_index "$OVERFIT_EPISODE")
    fi
    python prepare_so101_data.py \
        --data_dir "$SO101_DATA_DIR" \
        --meta_output "$TRAIN_METAS_PATH" \
        --stats_output "$NORM_STATS_PATH" \
        "${EPISODE_ARGS[@]}"
fi

ARGS="--output_dir ${OUTPUT_DIR} \
    --train_metas_path ${TRAIN_METAS_PATH} \
    --smolvlm_model_path ${SMOLVLM_MODEL_PATH} \
    --action_mode so101_joint \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --learning_coef ${LEARNING_COEF} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --num_actions 10 \
    --num_views 2 \
    --iters ${ITERS} \
    --warmup_steps 0 \
    --freeze_steps 0 \
    --hidden_size 768 \
    --depth 12 \
    --num_heads 12 \
    --num_workers 2 \
    --save_interval ${SAVE_INTERVAL} \
    --log_interval 20 \
    --image_size 384 \
    --norm_stats_path ${NORM_STATS_PATH} \
    --max_grad_norm 1.0 \
    --gradient_checkpointing"

if [ -n "$RESUME_CKPT" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --main_process_port 29504 \
    --mixed_precision fp16 \
    train_smolvlm.py ${ARGS}
