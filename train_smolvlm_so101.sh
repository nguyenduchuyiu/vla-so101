#!/bin/bash

set -e

# Balanced counterfactual SO101-delta training (1 GPU, bf16).
# batch_size 256 is the design point (64 source-decision + 64 target-decision
# + 128 execution = 25/25/50). Pass 8 as the first arg to reproduce the live
# overfit run. BATCH_SIZE must be divisible by 8 (enforced in dataset_smolvlm.py).
BATCH_SIZE=${1:-256}
OUTPUT_DIR=${2:-./runs/so101_balanced_delta_seed0_scratch}
RESUME_CKPT=${3:-""}

SO101_DATA_DIR=${SO101_DATA_DIR:-./data/branch_source}
TRAIN_METAS_PATH=${TRAIN_METAS_PATH:-./outputs/so101_balanced_counterfactual_seed0.json}
NORM_STATS_PATH=${NORM_STATS_PATH:-./norm_stats/so101_full_episode_delta_seed0_norm.json}
NORM_META_PATH=${NORM_META_PATH:-./outputs/so101_full_episode_overfit_seed0.json}
SMOLVLM_MODEL=${SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}

ITERS=${ITERS:-20000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}

export CUDA_VISIBLE_DEVICES=0

# Rebuild the balanced meta every run: build() is deterministic, and
# sampler_batch_size must equal the training batch_size (enforced in
# dataset_smolvlm.py).
python vla_data/balanced_counterfactual_dataset.py \
    --data-dir "$SO101_DATA_DIR" \
    --output "$TRAIN_METAS_PATH" \
    --scene-seed 0 \
    --num-actions 10 \
    --batch-size "$BATCH_SIZE"

# Delta norm stats are batch-independent; rebuild only if missing.
if [ ! -f "$NORM_STATS_PATH" ]; then
    python vla_data/full_episode_overfit.py \
        --data-dir "$SO101_DATA_DIR" \
        --scene-seed 0 \
        --output "$NORM_META_PATH"
    python prepare_so101_delta_stats.py \
        --meta "$NORM_META_PATH" \
        --output "$NORM_STATS_PATH" \
        --horizon 10
fi

ARGS="--output_dir ${OUTPUT_DIR} \
    --train_metas_path ${TRAIN_METAS_PATH} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode so101_delta \
    --batch_size ${BATCH_SIZE} \
    --num_workers 0 \
    --learning_rate 1e-4 \
    --learning_coef 1.0 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_actions 10 \
    --num_views 2 \
    --image_size 384 \
    --hidden_size 768 \
    --depth 12 \
    --num_heads 12 \
    --max_grad_norm 1.0 \
    --warmup_steps 0 \
    --freeze_steps 0 \
    --save_interval ${SAVE_INTERVAL} \
    --log_interval 25 \
    --iters ${ITERS} \
    --norm_stats_path ${NORM_STATS_PATH}"

if [ "$GRADIENT_CHECKPOINTING" = "1" ]; then
    ARGS="${ARGS} --gradient_checkpointing"
fi

if [ -n "$RESUME_CKPT" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
fi

PYTORCH_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=1 \
    --main_process_port 29504 \
    --mixed_precision bf16 \
    train_smolvlm.py ${ARGS}