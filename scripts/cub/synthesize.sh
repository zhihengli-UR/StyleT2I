#!/bin/bash

{

PYTHONPATH=.:$PYTHONPATH python synthesize.py \
    --seed 0 \
    --num_workers 20 \
    --eval_batch 20 \
    --stylegan_size 256 \
    --dataset cub \
    --truncation 0.5 \
    --ckpt exp/ckpt/stylegan2_cub_size_256_split_train/last.pt \
    --latent_space wp \
    --sentence2latent_ckpt exp/stylet2i/stylet2i_cub/ckpt/last.pt

    exit
}