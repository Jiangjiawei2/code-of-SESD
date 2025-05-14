#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# === Automatically create save directory ===
DATE_STR=$(date +"%Y-%m-%d-%H%M")
SAVE_DIR="/root/data1/jjw/code/jjw_mpgd_pytorch-main/linear_inv/logs/train-$DATE_STR"
mkdir -p "$SAVE_DIR"
export OPENAI_LOGDIR=$SAVE_DIR
echo "Saving logs and checkpoints to: $SAVE_DIR"

# === Model parameters ===
MODEL_FLAGS="\
--image_size 256 \
--num_channels 128 \
--num_res_blocks 1 \
--learn_sigma True \
--class_cond False \
--use_checkpoint False \
--attention_resolutions 16 \
--num_heads 4 \
--num_head_channels 64 \
--num_heads_upsample -1 \
--use_scale_shift_norm True \
--dropout 0.0 \
--resblock_updown True \
--use_fp16 False \
--use_new_attention_order False"

# === Diffusion process parameters ===
DIFFUSION_FLAGS="\
--diffusion_steps 1000 \
--noise_schedule linear"

# === Training parameters ===
TRAIN_FLAGS="\
--lr 1e-4 \
--batch_size 16 \
--log_interval 10 \
--save_interval 500 \
--lr_anneal_steps 100000"  # <<< Total training steps: 100,000 (note: 100 thousand steps)

# === Pretrained model ===
RESUME_FLAGS="--resume_checkpoint /root/data1/jjw/code/jjw_mpgd_pytorch-main/linear_inv/models/celebahq_p2.pt"

# === Start training ===
python ./image_train.py \
    --data_dir /root/data1/jjw/dataset/m4raw/train \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $TRAIN_FLAGS \
    $RESUME_FLAGS