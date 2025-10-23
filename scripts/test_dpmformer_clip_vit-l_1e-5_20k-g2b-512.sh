#!/bin/sh
CONFIG=configs/dpmformer/test_dpmformer_clip_vit-l_1e-5_20k-g2b-512.py
GPUS=$1
CHECKPOINT=$2
PORT=${PORT:-29512}

python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval mIoU