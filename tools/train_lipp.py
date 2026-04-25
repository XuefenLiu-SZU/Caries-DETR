#!/usr/bin/env python
# Copyright (c) 2025 Caries-DETR Authors. All rights reserved.
"""
Large-scale Intraoral Photograph Pre-training (LIPP) for the Structure
Perception Branch (SPB).

This script implements the self-supervised pre-training strategy described in:

    "Tooth Structure-aware Prior and Lesion-aware Dynamic Loss Refinement
     for DETR Based Caries Detection"

The SPB is pre-trained on a large corpus of unlabelled intraoral photographs
to learn high-frequency structural priors (tooth boundaries, enamel edges,
fissure patterns) *without* any bounding-box annotations.

**Pretext task.**  A GradientStructureGenerator computes a pseudo ground-truth
heatmap from the input image using Scharr gradient operators followed by a
log-transform that enhances subtle, low-contrast structures (e.g., early
caries texture).  The SPB is trained with an L1 loss to regress this heatmap
from the FPN P3 feature map.

**Training procedure.**
1. A DINO-compatible backbone + neck are loaded from an MMDetection config
   and frozen.
2. Only the SPB parameters are updated (L1 loss, AdamW optimizer).
3. The resulting checkpoint can be loaded into Caries-DETR via ``init_cfg``.

Usage:
    python tools/train_lipp.py \\
        --config configs/caries_detr_alphadent.py \\
        --data-root /path/to/intraoral_photos/ \\
        --save-dir work_dirs/lipp_pretrain \\
        --epochs 50 --batch-size 16 --lr 1e-4

The saved checkpoint (e.g., ``spb_pretrain_best.pth``) can then be used:

    bbox_head=dict(
        type='CariesDETRHead',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='work_dirs/lipp_pretrain/spb_pretrain_best.pth',
            prefix='structure_perception_branch.',
        ),
        ...
    )
"""

import argparse
import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from mmdet.apis import init_detector

# Allow loading of truncated images without raising errors
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ======================================================================
# Pseudo Ground-Truth Generator
# ======================================================================

class GradientStructureGenerator(nn.Module):
    """Generate structural pseudo ground-truth heatmaps via gradient analysis.

    A Scharr operator extracts horizontal and vertical gradients from the
    grayscale input.  The gradient magnitude is log-transformed to enhance
    low-contrast regions (e.g., early-stage caries texture and fissure
    patterns), then instance-normalised to [0, 1].

    The resulting heatmap highlights tooth boundaries, enamel-dentin
    junctions, and subtle lesion margins — serving as a self-supervised
    training signal for the Structure Perception Branch (SPB).
    """

    def __init__(self):
        super().__init__()
        # Scharr kernels are more sensitive to fine structures than Sobel
        scharr_x = torch.tensor(
            [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 32.0
        scharr_y = torch.tensor(
            [[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 32.0
        self.register_buffer('scharr_x', scharr_x)
        self.register_buffer('scharr_y', scharr_y)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute the structural heatmap from an RGB image batch.

        Args:
            images (Tensor): ``(B, 3, H, W)`` normalised input images.

        Returns:
            Tensor: ``(B, 1, H, W)`` heatmap with values in [0, 1].
        """
        # Convert to grayscale
        gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]

        # Scharr gradient magnitude
        grad_x = F.conv2d(gray, self.scharr_x, padding=1)
        grad_y = F.conv2d(gray, self.scharr_y, padding=1)
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # Log-transform to enhance low-contrast regions
        magnitude = torch.log1p(magnitude)

        # Instance-level normalisation (robust to illumination variation)
        B = magnitude.shape[0]
        flat = magnitude.view(B, -1)
        min_v = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_v = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        norm_mag = (magnitude - min_v) / (max_v - min_v + 1e-6)

        return norm_mag


# ======================================================================
# Dataset
# ======================================================================

class IntraoralDataset(Dataset):
    """Simple dataset that recursively loads intraoral photographs.

    No annotation files are required — every image found under
    ``img_folder`` (including sub-directories) is used.

    Args:
        img_folder (str): Root directory containing intraoral images.
        img_size (tuple[int, int]): Target ``(H, W)`` after resizing.
            Default: ``(800, 800)``.
    """

    _EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')

    def __init__(self, img_folder: str, img_size=(800, 800)):
        self.files = []
        for root, _dirs, files in os.walk(img_folder):
            for f in files:
                if f.lower().endswith(self._EXTENSIONS):
                    path = os.path.join(root, f)
                    if os.path.isfile(path):
                        self.files.append(path)
        print(f'[LIPP] Found {len(self.files)} intraoral images.')

        self.transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            with Image.open(self.files[idx]) as img:
                img = img.convert('RGB')
                return self.transform(img)
        except Exception as e:
            print(f'[LIPP] Error loading {self.files[idx]}: {e}')
            return None


# ======================================================================
# Pretext Model Wrapper
# ======================================================================

class LIPPModel(nn.Module):
    """Wrapper that pairs a frozen backbone+neck with a trainable SPB.

    The backbone and neck are initialised from an MMDetection config to
    guarantee architectural consistency with the downstream Caries-DETR
    model.  Only the SPB parameters are updated during pre-training.

    Args:
        config_path (str): Path to an MMDetection config file that
            defines the backbone and neck (e.g., Caries-DETR config).
        device (torch.device): Target device.
    """

    def __init__(self, config_path: str, device: torch.device):
        super().__init__()

        # Import StructurePerceptionBranch from our package
        from caries_detr.models.tsqi_module import StructurePerceptionBranch

        print(f'[LIPP] Building backbone/neck from config: {config_path}')
        full_model = init_detector(config_path, device=device)

        self.backbone = full_model.backbone
        self.neck = full_model.neck

        # Freeze backbone and neck — only SPB is trained
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.neck.parameters():
            param.requires_grad = False

        # Trainable SPB
        self.structure_perception_branch = StructurePerceptionBranch(
            in_channels=256, mid_channels=128,
        )

        # Pseudo GT generator (parameter-free)
        self.gt_generator = GradientStructureGenerator()

    def forward(self, img: torch.Tensor):
        """Run forward pass.

        Args:
            img (Tensor): ``(B, 3, H, W)`` normalised input batch.

        Returns:
            tuple[Tensor, Tensor]:
                - pred: SPB prediction ``(B, 1, h, w)``.
                - gt: Pseudo ground-truth ``(B, 1, h, w)``.
        """
        # Extract frozen features
        with torch.no_grad():
            x = self.backbone(img)
            feats = self.neck(x)
            p3_feat = feats[0]  # P3 (stride 8)

        # SPB prediction
        pred_structure = self.structure_perception_branch(p3_feat)

        # Generate and downsample pseudo GT
        with torch.no_grad():
            gt_highres = self.gt_generator(img)
            gt = F.interpolate(
                gt_highres, size=pred_structure.shape[-2:],
                mode='bilinear', align_corners=False,
            )

        return pred_structure, gt


def safe_collate(batch):
    """Collate function that silently drops ``None`` entries."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)


# ======================================================================
# Main
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='LIPP: Large-scale Intraoral Photograph Pre-training '
                    'for the Structure Perception Branch (SPB)',
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='MMDetection config defining backbone + neck architecture',
    )
    parser.add_argument(
        '--data-root', type=str, required=True,
        help='Root directory of intraoral photographs (no annotations needed)',
    )
    parser.add_argument(
        '--save-dir', type=str, default='work_dirs/lipp_pretrain',
        help='Directory to save SPB checkpoints',
    )
    parser.add_argument('--img-size', type=int, nargs=2, default=[800, 800],
                        help='Input image size (H W)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU ids (e.g., "0,1,2,3")')
    parser.add_argument('--log-interval', type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'[LIPP] Available GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

    # Dataset and dataloader
    dataset = IntraoralDataset(args.data_root, img_size=tuple(args.img_size))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=safe_collate,
    )

    # Build model
    model = LIPPModel(args.config, device).to(device)
    if torch.cuda.device_count() > 1:
        print(f'[LIPP] Using DataParallel across {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)

    # Auto-detect P3 channel count
    dummy = torch.randn(1, 3, args.img_size[0], args.img_size[1]).to(device)
    with torch.no_grad():
        core = model.module if isinstance(model, torch.nn.DataParallel) else model
        feats = core.neck(core.backbone(dummy))
        actual_ch = feats[0].shape[1]
        if actual_ch != 256:
            from caries_detr.models.tsqi_module import StructurePerceptionBranch
            print(f'[LIPP] Auto-detected P3 channels: {actual_ch}')
            core.structure_perception_branch = StructurePerceptionBranch(
                in_channels=actual_ch, mid_channels=128,
            ).to(device)

    # Optimizer (only SPB parameters)
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    optimizer = torch.optim.AdamW(
        core.structure_perception_branch.parameters(), lr=args.lr,
    )
    loss_fn = nn.L1Loss()

    # Training loop
    loss_window = deque(maxlen=5)
    best_loss = float('inf')

    print('[LIPP] Starting pre-training ...')
    for epoch in range(args.epochs):
        core.structure_perception_branch.train()
        total_loss = 0.0

        for i, imgs in enumerate(dataloader):
            if imgs is None:
                continue
            imgs = imgs.to(device)

            pred, gt = model(imgs)
            loss = loss_fn(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % args.log_interval == 0:
                print(f'  Epoch [{epoch}/{args.epochs}][{i}/{len(dataloader)}] '
                      f'Loss: {loss.item():.5f}')

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f'  Epoch {epoch} done.  Avg loss: {avg_loss:.5f}')

        # Save per-epoch checkpoint
        state = core.structure_perception_branch.state_dict()
        torch.save(state, os.path.join(args.save_dir, f'spb_pretrain_epoch_{epoch}.pth'))

        # Track best (smoothed) loss
        loss_window.append(avg_loss)
        smoothed = sum(loss_window) / len(loss_window)
        if smoothed < best_loss:
            best_loss = smoothed
            torch.save(state, os.path.join(args.save_dir, 'spb_pretrain_best.pth'))
            print(f'  Best model updated!  Smoothed loss: {best_loss:.5f}')

    # Save final checkpoint
    torch.save(
        core.structure_perception_branch.state_dict(),
        os.path.join(args.save_dir, 'spb_pretrain_final.pth'),
    )
    print('[LIPP] Pre-training complete.')


if __name__ == '__main__':
    main()
