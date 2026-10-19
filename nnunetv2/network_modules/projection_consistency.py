import math
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_rotation_matrix_3d(angle_rad: float, device: torch.device,
                             dtype: torch.dtype) -> torch.Tensor:
    """Build a 3x3 rotation matrix around the z-axis (volume depth axis)."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    R = torch.tensor([
        [cos_a,  sin_a, 0],
        [-sin_a, cos_a, 0],
        [0,      0,     1],
    ], device=device, dtype=dtype)
    return R


def _build_affine_cache(angles_deg: List[float], device: torch.device,
                        dtype: torch.dtype) -> Dict[float, torch.Tensor]:
    """Precompute (1, 3, 4) affine matrices for each projection angle."""
    cache = {}
    for angle_deg in angles_deg:
        theta = math.radians(angle_deg)
        cos_a, sin_a = math.cos(theta), math.sin(theta)
        aff = torch.zeros(1, 3, 4, device=device, dtype=dtype)
        aff[0, 0, 0] = cos_a
        aff[0, 0, 1] = sin_a
        aff[0, 1, 0] = -sin_a
        aff[0, 1, 1] = cos_a
        aff[0, 2, 2] = 1.0
        cache[angle_deg] = aff
    return cache


def differentiable_forward_projection(
    volume: torch.Tensor,
    angles_deg: Union[List[float], np.ndarray],
    align_corners: bool = False,
    affine_cache: Optional[Dict[float, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Differentiable parallel-beam forward projection via rotation + integration.

    For each projection angle, rotates the 3D volume around the z-axis and
    averages along z to produce a 2D projection image.  The rotation uses
    ``F.affine_grid`` + ``F.grid_sample`` and is fully differentiable.

    Args:
        volume: (B, C, D, H, W) float tensor.
        angles_deg: List of projection angles in degrees.
        align_corners: Passed to grid_sample / affine_grid.
        affine_cache: Optional precomputed dict angle_deg -> (1, 3, 4) affine.

    Returns:
        (B, N_views, C, H, W) — 2D projections stacked along the view axis.
    """
    B, C, D, H, W = volume.shape
    N = len(angles_deg)
    device = volume.device
    dtype = volume.dtype

    if affine_cache is None:
        affine_cache = _build_affine_cache(list(angles_deg), device, dtype)
    else:
        # Rebuild cache if device/dtype mismatch (e.g. model moved across GPUs)
        sample_aff = next(iter(affine_cache.values()))
        if sample_aff.device != device or sample_aff.dtype != dtype:
            affine_cache = _build_affine_cache(list(angles_deg), device, dtype)

    size = torch.Size([B, C, D, H, W])
    projections = []

    for angle_deg in angles_deg:
        theta_rad = math.radians(angle_deg)

        # 0-degree projection is a degenerate case — just average along z
        if abs(theta_rad) < 1e-8:
            proj = volume.mean(dim=2)
            projections.append(proj)
            continue

        affine = affine_cache[angle_deg].expand(B, -1, -1)
        grid = F.affine_grid(affine, size, align_corners=align_corners)
        rotated = F.grid_sample(volume, grid, mode='bilinear',
                                align_corners=align_corners, padding_mode='zeros')

        proj = rotated.mean(dim=2)  # average along z -> (B, C, H, W)
        projections.append(proj)

    return torch.stack(projections, dim=1)  # (B, N, C, H, W)


class ProjectionConsistencyLoss(nn.Module):
    """
    MSE between forward-projected prediction and forward-projected ground truth.

    This enforces 3D geometric consistency: the 3D segmentation shape must be
    consistent with all projection views.  Prediction logits are softmax-ed
    internally, then both prediction and target are forward-projected using the
    same geometry, and the mean squared error between the resulting projection
    pairs is minimized.

    Args:
        angles_deg: List of projection angles in degrees.
        align_corners: Passed to grid_sample (default False).
    """

    def __init__(self, angles_deg: Union[List[float], np.ndarray],
                 align_corners: bool = False):
        super().__init__()
        self.angles_deg = list(angles_deg)
        self.align_corners = align_corners
        self._affine_cache: Optional[Dict[float, torch.Tensor]] = None

    def _get_affine_cache(self, device: torch.device,
                          dtype: torch.dtype) -> Dict[float, torch.Tensor]:
        if self._affine_cache is None:
            self._affine_cache = _build_affine_cache(
                self.angles_deg, device, dtype)
        else:
            sample = next(iter(self._affine_cache.values()))
            if sample.device != device or sample.dtype != dtype:
                self._affine_cache = _build_affine_cache(
                    self.angles_deg, device, dtype)
        return self._affine_cache

    def forward(self, pred_logits: torch.Tensor,
                target_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: (B, num_classes, D, H, W) — network output (pre-softmax).
            target_onehot: (B, num_classes, D, H, W) — one-hot ground truth.

        Returns:
            Scalar MSE loss.
        """
        pred_probs = torch.softmax(pred_logits, dim=1)
        cache = self._get_affine_cache(pred_probs.device, pred_probs.dtype)

        pred_proj = differentiable_forward_projection(
            pred_probs, self.angles_deg, self.align_corners,
            affine_cache=cache)
        target_proj = differentiable_forward_projection(
            target_onehot.float(), self.angles_deg, self.align_corners,
            affine_cache=cache)

        return F.mse_loss(pred_proj, target_proj)


class CombinedSegProjLoss(nn.Module):
    """
    Combines a segmentation loss (DC+CE / DC+BCE, optionally wrapped in
    DeepSupervisionWrapper) with a ProjectionConsistencyLoss into a single
    ``nn.Module`` whose ``forward(output, target)`` signature is compatible
    with ``nnUNetTrainer.loss``.

    This makes the combined loss usable with external training loops such as
    ``run_training_dual_containment.py`` which call ``trainer.loss(output, target)``
    directly.

    Args:
        seg_loss: DC+CE or DC+BCE loss, optionally wrapped in DeepSupervisionWrapper.
        proj_loss: ProjectionConsistencyLoss instance.
        proj_weight: Weight for the projection consistency term.
        label_manager: nnUNet LabelManager (for target → one-hot conversion).
    """

    def __init__(self, seg_loss: nn.Module, proj_loss: ProjectionConsistencyLoss,
                 proj_weight: float, label_manager):
        super().__init__()
        from nnunetv2.training.loss.composite_loss import CompositeSegmentationLoss
        from nnunetv2.training.pipeline_config import EnhancementConfig

        self.seg_loss = seg_loss
        self.proj_loss = proj_loss
        self.proj_weight = proj_weight
        self.label_manager = label_manager
        angles = tuple(getattr(proj_loss, "angles_deg", ()))
        config = EnhancementConfig(
            projection_angles_deg=angles,
            projection_consistency_weight=proj_weight,
        )
        self._composite = CompositeSegmentationLoss(
            seg_loss, config, label_manager, projection_loss=proj_loss
        )

    def forward(self, output, target) -> torch.Tensor:
        """
        Args:
            output: Tensor (B,C,D,H,W) or list of Tensors (deep supervision).
            target: Tensor or list of Tensors.

        Returns:
            Scalar combined loss.
        """
        return self._composite(output, target).total
