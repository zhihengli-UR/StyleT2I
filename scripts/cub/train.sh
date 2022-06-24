#!/bin/bash

{

PYTHONPATH=.:$PYTHONPATH python train.py \
    --iter 80001 \
    --batch 40 \
    --num_workers 26 \
    --stylegan_size 256 \
    --dataset cub \
    --truncation 0.5 \
    --ckpt_clip_for_train exp/ft_clip_text/ft_clip_ViT-B_32_cub_train/ckpt.pth \
    --ckpt exp/ckpt/stylegan2_cub_size_256_split_train/last.pt \
    --lr 1e-4 \
    --latent_space wp \
    --seed 0 \
    --lambda_direction_norm_penalty 1.0 \
    --direction_norm_penalty_threshold 8.0

    exit
}