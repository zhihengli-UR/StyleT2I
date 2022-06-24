{

    PYTHONPATH=.:$PYTHONPATH python ft_clip_text.py \
        --dataset celebahq \
        --batch 192 \
        --num_workers 14 \
        --epoch 1000 \
        --train_split train

}
