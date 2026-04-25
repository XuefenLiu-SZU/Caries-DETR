#!/usr/bin/env python
# Copyright (c) 2025 Caries-DETR Authors. All rights reserved.
"""Evaluation script for Caries-DETR.

Usage:
    python tools/test.py configs/caries_detr_alphadent.py checkpoints/model.pth
"""
import argparse

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Test Caries-DETR')
    parser.add_argument('config', help='Test config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', help='Dump results to a pickle file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options',
    )
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
