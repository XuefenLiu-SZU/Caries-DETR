#!/bin/bash
# Distributed training launcher
# Usage: ./tools/dist_train.sh <config> <num_gpus> [extra args...]

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --launcher pytorch \
    "${@:3}"
