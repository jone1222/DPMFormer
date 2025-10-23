#!/bin/sh
CONFIG=configs/tqdm/tqdm_clip_vit-l_1e-5_20k-g2c-512.py
GPUS=$1
PORT=${PORT:-29501}


python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    train.py $CONFIG --launcher pytorch ${@:3}  \
    --gpus $GPUS --seed 2023 --deterministic \
