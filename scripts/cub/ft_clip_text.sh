{

    PYTHONPATH=.:$PYTHONPATH python ft_clip_text.py \
        --dataset cub \
        --train_split train \
        --batch 256 \
        --num_workers 26 \
        --epoch 100 \

exit
}