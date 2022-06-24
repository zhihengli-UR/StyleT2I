#!/bin/bash

{

PYTHONPATH=.:$PYTHONPATH python synthesize.py \
    --eval_batch 20 \
    --num_workers 40 \
    --seed 0 \
    --stylegan_size 256 \
    --dataset celebahq \
    --truncation 0.5 \
    --ckpt exp/pretrained_stylegan2/stylegan2_ffhq.pt \
    --latent_space wp \
    --sentence2latent_ckpt exp/stylet2i/stylet2i_ffhq/ckpt/last.pt

    exit
}