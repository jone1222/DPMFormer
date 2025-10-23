#!/bin/sh
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
# OUT_DIR=$3

python test.py $CONFIG $CHECKPOINT --eval mIoU --launcher none ${@:4}
# python test.py $CONFIG $CHECKPOINT --eval mIoU --launcher none --out $OUT_DIR ${@:4}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# 	python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
# 	    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
		
