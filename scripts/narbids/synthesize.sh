#!/bin/bash

{

PYTHONPATH=.:$PYTHONPATH python synthesize.py \
    --seed 0 \
    --num_workers 20 \
    --eval_batch 20 \
    --stylegan_size 256 \
    --dataset nabirds \
    --truncation 0.5 \
    --ckpt exp/pretrained_stylegan2/stylegan2_nabirds.pt \
    --latent_space wp \
    --sentence2latent_ckpt exp/stylet2i/stylet2i_nabirds/ckpt/last.pt

    exit
}