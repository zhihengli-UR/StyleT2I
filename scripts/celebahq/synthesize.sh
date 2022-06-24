#!/bin/bash

{

PYTHONPATH=.:$PYTHONPATH python synthesize.py \
    --eval_batch 20 \
    --num_workers 20 \
    --seed 0 \
    --stylegan_size 256 \
    --dataset celebahq \
    --ckpt exp/ckpt/stylegan2_celebahq_size_256_split_train/300000.pt \
    --sentence2latent_ckpt exp/stylet2i/stylet2i_celebahq/ckpt/last.pt

    exit
}