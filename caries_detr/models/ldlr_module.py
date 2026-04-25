# Copyright (c) 2025 Caries-DETR Authors. All rights reserved.
"""
Lesion-aware Dynamic Loss Refinement (LDLR) module for Caries-DETR.

This module implements the Lesion-aware Dynamic Loss Refinement (LDLR)
strategy described in:

    "Tooth Structure-aware Prior and Lesion-aware Dynamic Loss Refinement
     for DETR Based Caries Detection"

LDLR performs quality-driven hard mining by computing three independent,
per-prediction adaptive weights:

  w_iou   = 1 + gamma_iou  * (1 - IoU)          # localization quality
  w_cls   = 1 + alpha_cls  * (1 - cls_prob)      # classification quality
  w_bbox  = 1 + beta_bbox  * (1 - GIoU_quality)  # bounding-box quality

Predictions that already match the ground truth well receive weights close
to 1, while hard examples (low IoU, low confidence, poor localization) are
up-weighted.  This focuses training on challenging, subtle lesions that are
otherwise overwhelmed by easy negatives or well-detected instances.

The three sensitivity coefficients (gamma_iou, alpha_cls, beta_bbox) provide
fine-grained control over the relative emphasis on localization difficulty,
classification uncertainty, and anatomical-relevance-based box quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LesionDynamicLossRefiner(nn.Module):
    """Lesion-aware Dynamic Loss Refiner (LDLR).

    A parameter-free module that computes per-prediction quality-aware
    weights for the classification, bounding-box regression, and IoU loss
    terms.  By adaptively re-weighting the loss based on lesion size,
    anatomical relevance, and prediction quality, LDLR implements an
    implicit hard-mining strategy that optimizes detection for subtle,
    low-contrast caries lesions.

    Args:
        gamma_iou (float): Sensitivity coefficient for the IoU-based
            weight.  Larger values amplify the penalty for poorly
            localized predictions.  Default: 1.0.
        alpha_cls (float): Sensitivity coefficient for the classification-
            based weight.  Larger values amplify the penalty for low-
            confidence predictions.  Default: 1.0.
        beta_bbox (float): Sensitivity coefficient for the bounding-box
            quality weight (derived from GIoU).  Default: 1.0.
    """

    def __init__(
        self,
        gamma_iou: float = 1.0,
        alpha_cls: float = 1.0,
        beta_bbox: float = 1.0,
    ):
        super().__init__()
        self.gamma_iou = float(gamma_iou)
        self.alpha_cls = float(alpha_cls)
        self.beta_bbox = float(beta_bbox)

    # ------------------------------------------------------------------
    # Geometry utilities
    # ------------------------------------------------------------------

    @staticmethod
    def box_area(boxes: torch.Tensor) -> torch.Tensor:
        """Compute areas of a set of bounding boxes.

        Args:
            boxes (Tensor): Shape ``(N, 4)`` in xyxy format.

        Returns:
            Tensor: Areas of shape ``(N,)``.
        """
        if boxes.numel() == 0:
            return boxes.new_zeros((0,))
        x1, y1, x2, y2 = boxes.unbind(-1)
        return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    @staticmethod
    def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU matrix.

        Args:
            boxes1 (Tensor): Shape ``(N, 4)`` in xyxy format.
            boxes2 (Tensor): Shape ``(M, 4)`` in xyxy format.

        Returns:
            Tensor: IoU matrix of shape ``(N, M)``.
        """
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros(
                (boxes1.shape[0], boxes2.shape[0]), device=boxes1.device
            )
        area1 = LesionDynamicLossRefiner.box_area(boxes1)
        area2 = LesionDynamicLossRefiner.box_area(boxes2)
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2 - inter
        return inter / (union + 1e-6)

    @staticmethod
    def giou(pred_boxes: torch.Tensor, tgt_boxes: torch.Tensor) -> torch.Tensor:
        """Compute per-pair Generalized IoU for aligned box pairs.

        Args:
            pred_boxes (Tensor): ``(N, 4)`` in xyxy format.
            tgt_boxes (Tensor): ``(N, 4)`` in xyxy format.

        Returns:
            Tensor: GIoU values of shape ``(N,)`` in [-1, 1].
        """
        if pred_boxes.numel() == 0 or tgt_boxes.numel() == 0:
            return torch.zeros((pred_boxes.shape[0],), device=pred_boxes.device)

        inter_lt = torch.max(pred_boxes[:, :2], tgt_boxes[:, :2])
        inter_rb = torch.min(pred_boxes[:, 2:], tgt_boxes[:, 2:])
        inter_wh = (inter_rb - inter_lt).clamp(min=0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]

        area_pred = LesionDynamicLossRefiner.box_area(pred_boxes)
        area_tgt = LesionDynamicLossRefiner.box_area(tgt_boxes)
        union = area_pred + area_tgt - inter
        iou = inter / (union + 1e-6)

        enclose_lt = torch.min(pred_boxes[:, :2], tgt_boxes[:, :2])
        enclose_rb = torch.max(pred_boxes[:, 2:], tgt_boxes[:, 2:])
        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + 1e-6

        return iou - (enclose_area - union) / enclose_area

    # ------------------------------------------------------------------
    # Quality scores
    # ------------------------------------------------------------------

    def calculate_bbox_quality(
        self, pred_boxes: torch.Tensor, tgt_boxes: torch.Tensor
    ) -> torch.Tensor:
        """GIoU-based bounding-box quality, re-mapped to [0, 1].

        A value of 1.0 indicates a perfect match; lower values indicate
        increasing localization difficulty.

        Args:
            pred_boxes (Tensor): ``(N, 4)`` predicted boxes (xyxy).
            tgt_boxes (Tensor): ``(N, 4)`` matched ground-truth boxes (xyxy).

        Returns:
            Tensor: Quality scores of shape ``(N,)`` in [0, 1].
        """
        if pred_boxes.numel() == 0 or tgt_boxes.numel() == 0:
            return torch.ones((pred_boxes.shape[0],), device=pred_boxes.device)
        giou = self.giou(pred_boxes, tgt_boxes)
        return ((giou + 1.0) / 2.0).clamp(0.0, 1.0)

    def calculate_cls_quality(
        self, pred_logits: torch.Tensor, tgt_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute the softmax probability for the target class.

        Args:
            pred_logits (Tensor): ``(N, C)`` raw class logits.
            tgt_labels (Tensor): ``(N,)`` ground-truth class indices.

        Returns:
            Tensor | None: Per-prediction target-class probability ``(N,)``
                or ``None`` if inputs are empty.
        """
        if pred_logits is None or pred_logits.numel() == 0:
            return None
        pred_probs = F.softmax(pred_logits, dim=-1)
        cls_quality = pred_probs[
            torch.arange(pred_probs.shape[0], device=pred_probs.device), tgt_labels
        ]
        return cls_quality

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred_boxes: torch.Tensor,
        tgt_boxes: torch.Tensor,
        pred_logits: torch.Tensor = None,
        tgt_labels: torch.Tensor = None,
    ) -> dict:
        """Compute per-prediction adaptive loss weights.

        Supports single-image inputs (2-D tensors) or batched inputs
        (3-D tensors).

        Args:
            pred_boxes (Tensor): Predicted boxes ``(N, 4)`` or
                ``(B, N, 4)`` in xyxy format.
            tgt_boxes (Tensor): Ground-truth boxes ``(M, 4)`` or
                ``(B, M, 4)`` in xyxy format.
            pred_logits (Tensor, optional): Class logits ``(N, C)`` or
                ``(B, N, C)``.
            tgt_labels (Tensor, optional): GT class indices ``(M,)`` or
                ``(B, M)``.

        Returns:
            dict: Dictionary with the following keys:

                - ``w_iou`` — IoU-based adaptive weights.
                - ``w_cls`` — Classification-based adaptive weights.
                - ``w_bbox`` — BBox-quality-based adaptive weights.
                - ``ious`` — Best-match IoU per prediction.
                - ``cls_scores`` — Target-class probability per prediction.
                - ``bbox_scores`` — GIoU-based quality per prediction.
        """
        if pred_boxes.dim() == 3:
            B = pred_boxes.shape[0]
            results = [
                self.forward(
                    pred_boxes[b],
                    tgt_boxes[b],
                    pred_logits[b] if pred_logits is not None else None,
                    tgt_labels[b] if tgt_labels is not None else None,
                )
                for b in range(B)
            ]
            return {k: torch.stack([r[k] for r in results], dim=0) for k in results[0]}

        # Single-image path
        N = pred_boxes.shape[0]
        if N == 0 or tgt_boxes.numel() == 0:
            ones = torch.ones((N,), device=pred_boxes.device)
            return {
                'w_iou': ones,
                'w_cls': ones,
                'w_bbox': ones,
                'ious': torch.zeros((N,), device=pred_boxes.device),
                'cls_scores': ones,
                'bbox_scores': ones,
            }

        ious_mat = self.box_iou(pred_boxes, tgt_boxes)       # (N, M)
        best_iou, best_tidx = ious_mat.max(dim=1)

        matched_tgts = tgt_boxes[best_tidx]
        bbox_scores = self.calculate_bbox_quality(pred_boxes, matched_tgts)

        if pred_logits is not None and tgt_labels is not None:
            matched_labels = tgt_labels[best_tidx]
            cls_scores = self.calculate_cls_quality(pred_logits, matched_labels)
            if cls_scores is None:
                cls_scores = torch.ones_like(best_iou)
        else:
            cls_scores = torch.ones_like(best_iou)

        w_iou = 1.0 + self.gamma_iou * (1.0 - best_iou)
        w_bbox = 1.0 + self.beta_bbox * (1.0 - bbox_scores)
        w_cls = 1.0 + self.alpha_cls * (1.0 - cls_scores)

        return {
            'w_iou': w_iou,
            'w_cls': w_cls,
            'w_bbox': w_bbox,
            'ious': best_iou,
            'cls_scores': cls_scores,
            'bbox_scores': bbox_scores,
        }
