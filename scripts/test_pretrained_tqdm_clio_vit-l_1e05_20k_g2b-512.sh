#!/bin/sh
CONFIG=/home/jone9312/tqdm-main/configs/tqdm/tqdm_clip_vit-l_1e-5_20k-g2b-512.py
CHECKPOINT=/home/jone9312/tqdm-main/pretrained/pretrained_tqdm/tqdm_clip_gta.pth
GPUS=$1

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=29502 \
	test.py $CONFIG $CHECKPOINT --launcher pytorch --eval mIoU
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# 	python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
# 	    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
		

# /home/jone9312/tqdm-main/