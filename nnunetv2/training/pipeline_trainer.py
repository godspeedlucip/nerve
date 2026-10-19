"""Shared assembly and compatibility adapters for legacy trainers."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor, nn

from nnunetv2.network_modules.enhancement_pipeline import EnhancementPipeline
from nnunetv2.network_modules.projection_consistency import ProjectionConsistencyLoss
from nnunetv2.training.auxiliary_targets import AuxiliaryTargets, TensorTree
from nnunetv2.training.loss.composite_loss import CompositeSegmentationLoss, LossBreakdown
from nnunetv2.training.loss.dtm_losses import GenSurfLoss
from nnunetv2.training.pipeline_config import EnhancementConfig
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


def legacy_config(name: str) -> EnhancementConfig:
    return EnhancementConfig.from_legacy_preset(name)


def _preset_name_for_class(name: str) -> str:
    return name.removesuffix("_A").removesuffix("_B")


class LegacyTrainerAssemblyMixin:
    """Small class-level compatibility surface shared by legacy trainers."""

    enhancement_preset: str = "nnUNetTrainer"

    @classmethod
    def enhancement_config(cls) -> EnhancementConfig:
        return legacy_config(_preset_name_for_class(cls.enhancement_preset))

    @classmethod
    def legacy_preset_name(cls) -> str:
        return cls.enhancement_preset


def build_network(
    architecture_class_name: str,
    arch_init_kwargs: dict,
    arch_init_kwargs_req_import,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool,
    config: EnhancementConfig,
) -> nn.Module:
    backbone = get_network_from_plans(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        allow_init=True,
        deep_supervision=enable_deep_supervision,
    )
    if not (config.enable_anisotropic_attention or config.enable_tubular_enhancement):
        return backbone
    return EnhancementPipeline(backbone, num_input_channels, num_output_channels, config)


class TrainerLossAdapter(nn.Module):
    """Scalar-compatible trainer loss retaining the named composite breakdown."""

    def __init__(self, composite: CompositeSegmentationLoss):
        super().__init__()
        self.composite = composite
        self.seg_loss = composite.segmentation_loss
        self.proj_loss = composite.projection_loss
        self.dtm_loss = composite.dtm_loss
        self.last_breakdown: Optional[LossBreakdown] = None

    def breakdown(self, output, target, auxiliary=None, epoch=None) -> LossBreakdown:
        self.last_breakdown = self.composite(output, target, auxiliary, epoch)
        return self.last_breakdown

    def _target_to_onehot(self, target: Tensor, num_classes: int) -> Tensor:
        indices = target[:, 0].long()
        onehot = torch.nn.functional.one_hot(indices, num_classes=num_classes)
        return onehot.permute(0, 4, 1, 2, 3).float()

    def forward(self, output, target, auxiliary=None, epoch=None) -> Tensor:
        return self.breakdown(output, target, auxiliary, epoch).total


def build_composite_loss(
    config: EnhancementConfig,
    segmentation_loss: nn.Module,
    label_manager: Any,
    projection_loss: Optional[nn.Module] = None,
    dtm_loss: Optional[nn.Module] = None,
) -> TrainerLossAdapter:
    if config.projection_consistency_weight and projection_loss is None:
        projection_loss = ProjectionConsistencyLoss(config.projection_angles_deg, align_corners=False)
    if config.dtm_mode != "disabled" and dtm_loss is None:
        dtm_loss = GenSurfLoss(class_weights=None, ignored_label=-1)
    return TrainerLossAdapter(
        CompositeSegmentationLoss(
            segmentation_loss,
            config,
            label_manager,
            projection_loss=projection_loss,
            dtm_loss=dtm_loss,
        )
    )


def build_legacy_loss(preset_name: str, segmentation_loss: nn.Module, label_manager: Any) -> TrainerLossAdapter:
    return build_composite_loss(legacy_config(preset_name), segmentation_loss, label_manager)


def _normalize_legacy_mask(dtm: TensorTree, valid_mask: TensorTree) -> TensorTree:
    if isinstance(dtm, torch.Tensor) and isinstance(valid_mask, torch.Tensor):
        if dtm.ndim == 6 and valid_mask.ndim == 5:
            expanded = valid_mask.unsqueeze(-1)
            return expanded.expand(dtm.shape)
        return valid_mask
    if isinstance(dtm, (list, tuple)) and isinstance(valid_mask, (list, tuple)):
        if len(dtm) != len(valid_mask):
            raise ValueError(
                f"DTM and valid_mask list lengths must match: {len(dtm)} != {len(valid_mask)}"
            )
        values = [_normalize_legacy_mask(d, m) for d, m in zip(dtm, valid_mask)]
        return type(valid_mask)(values) if isinstance(valid_mask, tuple) else values
    return valid_mask


def move_auxiliary_batch(batch: dict, device: torch.device | str) -> AuxiliaryTargets:
    dtm = batch.get("dtm")
    valid_mask = batch.get("valid_mask")
    if dtm is not None and valid_mask is not None:
        valid_mask = _normalize_legacy_mask(dtm, valid_mask)
    return AuxiliaryTargets(dtm=dtm, valid_mask=valid_mask).to(device)


def calculate_loss(loss: TrainerLossAdapter, output, target, auxiliary=None, epoch=None) -> LossBreakdown:
    return loss.breakdown(output, target, auxiliary, epoch)


def breakdown_log_values(breakdown: LossBreakdown) -> dict[str, Tensor]:
    return {
        "loss": breakdown.total,
        "seg_loss": breakdown.segmentation,
        "proj_loss": breakdown.projection,
        "gsl_loss": breakdown.dtm,
    }
