# Copyright (c) 2025 Caries-DETR Authors. All rights reserved.
"""
Tooth Structure-aware Query Initialization (TSQI) module for Caries-DETR.

This module implements the Tooth Structure-aware Query Initialization (TSQI)
strategy described in:

    "Tooth Structure-aware Prior and Lesion-aware Dynamic Loss Refinement
     for DETR Based Caries Detection"

TSQI consists of two sub-modules:

1. **Structure Perception Branch (SPB)** — a lightweight CNN that produces a
   single-channel heatmap over the backbone feature map, encoding high-frequency
   structural priors (e.g., tooth boundaries, enamel edges, and lesion margins)
   learned from large-scale intraoral photograph pre-training.

2. **TSQIQueryGenerator** — selects the top-k spatially salient positions from
   the feature map, weighted by the SPB heatmap, and projects them into
   DINO-compatible content queries and positional encodings.  This guides the
   decoder to attend to anatomically significant regions from the very first
   decoding step.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructurePerceptionBranch(nn.Module):
    """Structure Perception Branch (SPB).

    A lightweight two-layer convolutional encoder followed by a 1x1 projection
    that produces a spatial heatmap of tooth structural priors.  The heatmap
    values lie in [0, 1], where higher values indicate regions with strong
    anatomical significance such as tooth boundaries, enamel-dentin junctions,
    and potential lesion margins.

    The SPB is pre-trained on large-scale intraoral photographs to capture
    domain-specific high-frequency structural features.

    Args:
        in_channels (int): Number of input feature channels (typically 256
            from the FPN neck).
        mid_channels (int): Number of intermediate hidden channels.
            Default: 128.
    """

    def __init__(self, in_channels: int, mid_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.head = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the structural prior heatmap.

        Args:
            x (Tensor): Feature map of shape ``(B, C, H, W)``.

        Returns:
            Tensor: Heatmap of shape ``(B, 1, H, W)`` with values in [0, 1].
        """
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        heat = torch.sigmoid(self.head(y))
        return heat


class TSQIQueryGenerator(nn.Module):
    """Tooth Structure-aware Query Initialization (TSQI) Generator.

    Selects the top-k spatially salient positions from the backbone feature
    map, guided by the structural prior heatmap produced by the SPB.  The
    selected positions are projected into content queries and positional
    encodings that serve as the initial input to the DINO decoder.

    By biasing initial queries toward anatomically relevant locations, TSQI
    enables the detector to focus on clinically significant regions (e.g.,
    lesion boundaries) from the very first decoder layer, substantially
    reducing the number of iterations needed to converge on subtle caries.

    Args:
        in_channels (int): Number of input feature channels.
        query_dim (int): Output dimensionality of the content queries.
            Default: 256.
        num_queries (int): Total number of DINO queries. Default: 300.
        topk (int): Number of structure-guided positions to select.
            Default: 300.
    """

    def __init__(
        self,
        in_channels: int,
        query_dim: int = 256,
        num_queries: int = 300,
        topk: int = 300,
    ):
        super().__init__()
        self.score_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.proj = nn.Linear(in_channels, query_dim)
        self.pos_proj = nn.Linear(4, query_dim)
        self.num_queries = num_queries
        self.topk = topk

    def _get_grid_coords(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return normalized grid coordinates of shape ``(H*W, 2)``."""
        ys = torch.linspace(-1, 1, steps=H, device=device)
        xs = torch.linspace(-1, 1, steps=W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        return grid

    def forward(
        self,
        feat: torch.Tensor,
        structure_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate structure-aware initial queries.

        Args:
            feat (Tensor): Backbone feature map ``(B, C, H, W)``.
            structure_prior (Tensor): SPB heatmap ``(B, 1, H', W')``.

        Returns:
            tuple:
                - queries (Tensor): Content queries ``(B, topk, query_dim)``.
                - query_pos (Tensor): Positional encodings
                  ``(B, topk, query_dim)``.
        """
        B, C, H, W = feat.shape
        device = feat.device

        # Score each spatial position
        score = self.score_conv(feat).view(B, -1)  # (B, H*W)

        # Up-sample structural prior to feature map resolution and fuse
        sp = F.interpolate(structure_prior, size=(H, W), mode='bilinear', align_corners=False)
        sp_flat = sp.view(B, -1)
        weighted_score = score * (1.0 + sp_flat)

        # Select top-k structure-guided positions
        topk = min(self.topk, H * W)
        vals, idxs = torch.topk(weighted_score, k=topk, dim=1)

        feat_flat = feat.view(B, C, -1)  # (B, C, H*W)
        queries = torch.zeros(B, topk, self.proj.out_features, device=device)
        query_pos = torch.zeros(B, topk, self.pos_proj.out_features, device=device)
        grid_coords = self._get_grid_coords(H, W, device)  # (H*W, 2)

        for b in range(B):
            idb = idxs[b]
            fsel = feat_flat[b:b + 1, :, idb].squeeze(0).permute(1, 0)  # (topk, C)
            queries[b] = self.proj(fsel)

            coords = grid_coords[idb]  # (topk, 2)
            local_activation = vals[b].unsqueeze(1)  # (topk, 1)
            pos_input = torch.cat([coords, local_activation], dim=1)  # (topk, 3)
            pad = torch.zeros(topk, 1, device=device)
            pos_in = torch.cat([pos_input, pad], dim=1)  # (topk, 4)
            query_pos[b] = self.pos_proj(pos_in)

        return queries, query_pos
