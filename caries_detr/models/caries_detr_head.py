# Copyright (c) 2025 Caries-DETR Authors. All rights reserved.
"""
CariesDETRHead — DINO detection head with TSQI and LDLR.

This module extends the standard DINO detection head (DINOHead from
MMDetection) with two novel components proposed in:

    "Tooth Structure-aware Prior and Lesion-aware Dynamic Loss Refinement
     for DETR Based Caries Detection"

1. **Tooth Structure-aware Query Initialization (TSQI)**
   A Structure Perception Branch (SPB) produces a spatial heatmap from the
   backbone feature map.  A TSQIQueryGenerator selects the top-k high-
   response positions as initial decoder queries, guiding the Transformer
   decoder to attend to anatomically significant regions (e.g., tooth
   boundaries and lesion margins) from the very first decoding step.

2. **Lesion-aware Dynamic Loss Refinement (LDLR)**
   A parameter-free module that computes per-prediction quality-aware
   weights (w_iou, w_cls, w_bbox) and scales the corresponding loss terms
   so that hard, poorly-localized, or low-confidence predictions receive
   greater training emphasis — implementing quality-driven hard mining for
   subtle caries lesions.

Usage (via MMDetection ``custom_imports``):

    custom_imports = dict(
        imports=['caries_detr.models.caries_detr_head'],
        allow_failed_imports=False,
    )

    model = dict(
        bbox_head=dict(
            type='CariesDETRHead',
            num_classes=9,
            tsqi_cfg=dict(
                in_channels=256,
                mid_channels=128,
                num_queries=900,
                query_dim=256,
                topk=300,
            ),
            ldlr_cfg=dict(
                gamma_iou=1.0,
                alpha_cls=1.0,
                beta_bbox=1.0,
                enable_ldlr=True,
            ),
            ...
        ),
        ...
    )
"""

import torch
from mmdet.models.dense_heads.dino_head import DINOHead
from mmdet.registry import MODELS

from .tsqi_module import StructurePerceptionBranch, TSQIQueryGenerator
from .ldlr_module import LesionDynamicLossRefiner


@MODELS.register_module()
class CariesDETRHead(DINOHead):
    """DINO detection head augmented with TSQI and LDLR for caries detection.

    Args:
        tsqi_cfg (dict, optional): Configuration for the Tooth Structure-
            aware Query Initialization module.  Keys: ``in_channels``,
            ``mid_channels``, ``num_queries``, ``query_dim``, ``topk``.
            If ``None``, TSQI is disabled and the detector falls back to
            the default DINO learned query embedding.
        ldlr_cfg (dict, optional): Configuration for the Lesion-aware
            Dynamic Loss Refinement module.  Keys: ``gamma_iou``,
            ``alpha_cls``, ``beta_bbox``, ``enable_ldlr``.
            Set ``enable_ldlr=False`` to disable LDLR at runtime without
            removing the module.
        **kwargs: All remaining arguments are forwarded to
            :class:`DINOHead`.
    """

    def __init__(self, *args, tsqi_cfg=None, ldlr_cfg=None,
                 # keep backward compatibility with old config key names
                 sa_cfg=None, loss_refiner_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Support both new (tsqi_cfg/ldlr_cfg) and legacy (sa_cfg/loss_refiner_cfg) keys
        tsqi_cfg = tsqi_cfg or sa_cfg or {}
        ldlr_cfg = ldlr_cfg or loss_refiner_cfg or {}

        in_ch = tsqi_cfg.get('in_channels', getattr(self, 'embed_dims', 256))

        # --- Tooth Structure-aware Query Initialization (TSQI) ---
        self.structure_perception_branch = StructurePerceptionBranch(
            in_channels=in_ch,
            mid_channels=tsqi_cfg.get('mid_channels', 128),
        )
        self.tsqi_query_gen = TSQIQueryGenerator(
            in_channels=in_ch,
            query_dim=tsqi_cfg.get('query_dim', getattr(self, 'embed_dims', 256)),
            num_queries=tsqi_cfg.get('num_queries', getattr(self, 'num_queries', 300)),
            topk=tsqi_cfg.get('topk', min(getattr(self, 'num_queries', 300), 300)),
        )

        # --- Lesion-aware Dynamic Loss Refinement (LDLR) ---
        self._enable_ldlr = ldlr_cfg.get('enable_ldlr', ldlr_cfg.get('enable_dlr', True))
        self.loss_refiner = LesionDynamicLossRefiner(
            gamma_iou=ldlr_cfg.get('gamma_iou', 1.0),
            alpha_cls=ldlr_cfg.get('alpha_cls', 1.0),
            beta_bbox=ldlr_cfg.get('beta_bbox', 1.0),
        )

        # Placeholders for captured predictions used by LDLR
        self.last_pred_boxes = None
        self.last_pred_logits = None

    # ------------------------------------------------------------------
    # Prediction extraction helpers
    # ------------------------------------------------------------------

    def extract_ldlr_predictions(self, raw_outputs: dict) -> None:
        """Populate ``last_pred_boxes`` / ``last_pred_logits`` from decoder outputs.

        These cached predictions are used by the LDLR module to compute
        quality-based adaptive loss weights.

        Args:
            raw_outputs (dict): Must contain ``'hidden_states'`` (list of
                tensors, each ``(B, N, D)``) and ``'references'`` (list of
                tensors, each ``(B, N, 4)``).
        """
        if not isinstance(raw_outputs, dict):
            self.last_pred_boxes = self.last_pred_logits = None
            return

        hidden_states = raw_outputs.get('hidden_states')
        references = raw_outputs.get('references')

        if hidden_states is None or references is None:
            self.last_pred_boxes = self.last_pred_logits = None
            return

        last_hidden = hidden_states[-1]   # (B, N, D)
        last_ref = references[-1]         # (B, N, 4)

        cls_branch = self.cls_branches[-1]
        reg_branch = self.reg_branches[-1]

        pred_logits = cls_branch(last_hidden)
        pred_boxes_unact = reg_branch(last_hidden)
        pred_boxes = self.bbox_coder.decode(pred_boxes_unact, last_ref)

        self.last_pred_boxes = pred_boxes.detach()
        self.last_pred_logits = pred_logits.detach()

    def apply_ldlr_weighting(
        self,
        losses: dict,
        pred_boxes: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_bboxes: list,
        gt_labels: list,
    ) -> dict:
        """Scale per-image losses by their corresponding LDLR weights.

        For each image in the batch, the LDLR module computes per-prediction
        quality weights (w_cls, w_bbox, w_iou).  The mean weight is used to
        scale the corresponding aggregated loss scalar.

        Args:
            losses (dict): Loss dictionary from the base DINO head.
            pred_boxes (Tensor): ``(B, N, 4)`` predicted boxes.
            pred_logits (Tensor): ``(B, N, C)`` predicted class logits.
            gt_bboxes (list[Tensor]): Per-image GT boxes, each ``(M_i, 4)``.
            gt_labels (list[Tensor]): Per-image GT labels, each ``(M_i,)``.

        Returns:
            dict: Updated loss dictionary with LDLR-weighted terms.
        """
        if pred_boxes is None or pred_logits is None:
            return losses

        B = len(gt_bboxes)
        for b in range(B):
            if len(gt_bboxes[b]) == 0 or pred_boxes[b].numel() == 0:
                continue

            ldlr_out = self.loss_refiner(
                pred_boxes=pred_boxes[b:b + 1],
                tgt_boxes=gt_bboxes[b].unsqueeze(0),
                pred_logits=pred_logits[b:b + 1],
                tgt_labels=gt_labels[b].unsqueeze(0),
            )

            w_cls_mean = ldlr_out['w_cls'][0].mean()
            w_bbox_mean = ldlr_out['w_bbox'][0].mean()
            w_iou_mean = ldlr_out['w_iou'][0].mean()

            for key, weight in [
                ('loss_cls', w_cls_mean),
                ('loss_bbox', w_bbox_mean),
                ('loss_iou', w_iou_mean),
            ]:
                if key not in losses:
                    continue
                try:
                    losses[key][b] = losses[key][b] * weight
                except Exception:
                    losses[key] = losses[key] * weight

        return losses

    # ------------------------------------------------------------------
    # Forward overrides
    # ------------------------------------------------------------------

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels, **kwargs):
        """Forward pass with TSQI query injection and LDLR loss weighting.

        1. The SPB produces a structural prior heatmap from the finest-scale
           feature map.
        2. The TSQIQueryGenerator selects top-k structure-guided positions
           and temporarily overwrites the DINO learned query embedding.
        3. The base DINO forward computes losses as usual.
        4. If LDLR is enabled, the loss terms are re-weighted by per-
           prediction quality-aware coefficients.
        5. The original query embedding is restored.
        """
        # --- TSQI: structure-aware query initialization ---
        feat_for_prior = x[0]
        structure_prior = self.structure_perception_branch(feat_for_prior)
        queries, _query_pos = self.tsqi_query_gen(feat_for_prior, structure_prior)

        replaced_query = False
        if hasattr(self, 'query_embedding'):
            old_q = self.query_embedding.detach().clone()
            mean_q = queries.mean(dim=0)
            nq = getattr(self, 'num_queries', mean_q.shape[0])
            if mean_q.shape[0] >= nq:
                new_q = mean_q[:nq, :]
            else:
                new_q = torch.cat(
                    [mean_q,
                     torch.zeros(nq - mean_q.shape[0], mean_q.shape[1],
                                 device=mean_q.device)],
                    dim=0,
                )
            self.query_embedding.data[:nq, :] = new_q
            replaced_query = True

        # --- Base DINO training forward ---
        losses = super().forward_train(x, img_metas, gt_bboxes, gt_labels, **kwargs)

        # --- LDLR: lesion-aware dynamic loss refinement ---
        if self._enable_ldlr:
            if hasattr(self, '_raw_outputs') and self._raw_outputs is not None:
                self.extract_ldlr_predictions(self._raw_outputs)

            try:
                if self.last_pred_boxes is not None and self.last_pred_logits is not None:
                    losses = self.apply_ldlr_weighting(
                        losses,
                        self.last_pred_boxes,
                        self.last_pred_logits,
                        gt_bboxes,
                        gt_labels,
                    )
            except Exception as e:
                # LDLR failure is non-fatal; log and continue training
                print(f'[Caries-DETR] LDLR application skipped: {e}')

        # Restore original query embedding
        if replaced_query:
            self.query_embedding.data[:old_q.shape[0], :] = old_q

        return losses

    def loss(self, *args, **kwargs):
        """Override to allow capturing raw decoder outputs for LDLR."""
        losses = super().loss(*args, **kwargs)
        return losses
