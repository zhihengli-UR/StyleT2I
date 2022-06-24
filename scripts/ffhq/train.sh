#!/bin/bash

{

PYTHONPATH=.:$PYTHONPATH python train.py \
    --batch 40 \
    --num_workers 26 \
    --seed 0 \
    --stylegan_size 256 \
    --dataset ffhq \
    --truncation 0.5 \
    --ckpt_clip_for_train exp/ft_clip_text/ft_clip_ViT-B_32_celebahq_train/ckpt.pth \
    --ckpt exp/pretrained_stylegan2/stylegan2_ffhq.pt \
    --lr 1e-4 \
    --latent_space wp \
    --direction_norm_penalty_threshold 10.0 \
    --lambda_direction_norm_penalty 1.0

    exit
}