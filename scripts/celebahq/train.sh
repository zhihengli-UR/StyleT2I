#!/bin/bash

{

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH python train.py \
    --batch 40 \
    --num_workers 14 \
    --seed 0 \
    --dataset celebahq \
    --ckpt_clip_for_train exp/ft_clip_text/ft_clip_ViT-B_32_celebahq_train/ckpt.pth \
    --ckpt exp/ckpt/stylegan2_celebahq_size_256_split_train/300000.pt \
    --lr 1e-4 \
    --latent_space wp \
    --direction_norm_penalty_threshold 8.0 \
    --lambda_direction_norm_penalty 1.0

    exit
}