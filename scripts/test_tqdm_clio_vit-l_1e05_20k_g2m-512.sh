#!/bin/sh
CONFIG=/home/jone9312/tqdm-main/configs/tqdm/tqdm_clip_vit-l_1e-5_20k-g2m-512.py
CHECKPOINT=work_dirs_d/tqdm_clip_vit-l_1e-5_20k-g2b-512/latest.pth
GPUS=$1

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=29503 \
	test.py $CONFIG $CHECKPOINT --launcher pytorch --eval mIoU
