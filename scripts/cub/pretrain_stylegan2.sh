python -m torch.distributed.launch --nproc_per_node=4 --master_port=6889 pretrain_stylegan2.py \
    --batch 4 \
    --num_workers 4 \
    --dataset cub \
    --size 256
