#!/usr/bin/env python
# Copyright (c) 2025 Caries-DETR Authors. All rights reserved.
"""Training script for Caries-DETR.

Usage (single GPU):
    python tools/train.py configs/caries_detr_alphadent.py

Usage (multi-GPU, 4 GPUs):
    ./tools/dist_train.sh configs/caries_detr_alphadent.py 4
"""
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train Caries-DETR')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='Override work_dir in config')
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from the latest checkpoint in work_dir',
    )
    parser.add_argument('--amp', action='store_true', help='Enable AMP training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options, e.g. --cfg-options model.backbone.depth=50',
    )
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    # Update local_rank for torch.distributed
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    cfg.launcher = args.launcher
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.resume:
        cfg.resume = True
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
