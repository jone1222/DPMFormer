#!/bin/sh
CONFIG=configs/dpmformer/tqdm_clip_vit-l_1e-5_20k-g2c-512_new_cons_bnmetanet_contra_t0.5.py
GPUS=$1
PORT=${PORT:-29505}

python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    train.py $CONFIG --launcher pytorch ${@:3}  \
    --gpus $GPUS --seed 2023 --deterministic \
