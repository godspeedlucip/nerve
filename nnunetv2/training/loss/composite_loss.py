"""Composable segmentation, projection-consistency, and DTM losses."""

from dataclasses import dataclass
import warnings
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nnunetv2.network_modules.projection_consistency import ProjectionConsistencyLoss
from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
from nnunetv2.training.pipeline_config import EnhancementConfig


@dataclass(frozen=True)
class LossBreakdown:
    """Scalar component losses produced by :class:`CompositeSegmentationLoss`."""

    total: Tensor
    segmentation: Tensor
    projection: Tensor
    dtm: Tensor

    def as_scalar(self) -> Tensor:
        return self.total


@dataclass(frozen=True)
class _NormalizedTargets:
    output: Tensor
    target: Tensor


def _first_resolution(value):
    return value[0] if isinstance(value, (list, tuple)) else value


class CompositeSegmentationLoss(nn.Module):
    """Compose the existing segmentation, projection, and DTM loss modules."""

    def __init__(
        self,
        segmentation_loss: nn.Module,
        config: EnhancementConfig,
        label_manager,
        projection_loss: Optional[nn.Module] = None,
        dtm_loss: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.segmentation_loss = segmentation_loss
        self.config = config
        self.label_manager = label_manager
        self.projection_loss = projection_loss
        self.dtm_loss = dtm_loss
        self._missing_auxiliary_warned = False

        if config.projection_consistency_weight and projection_loss is None:
            raise ValueError(
                "projection_loss is required when projection consistency is enabled"
            )

    @staticmethod
    def _as_scalar(value: Tensor | float | int, reference: Tensor) -> Tensor:
        if isinstance(value, Tensor):
            return value if value.ndim == 0 else value.mean()
        return reference.new_tensor(value)

    def _normalize_final_targets(self, output: Tensor, target: Tensor) -> _NormalizedTargets:
        if self.label_manager.has_regions:
            num_classes = self.label_manager.num_segmentation_heads
            target_onehot = target[:, :num_classes].float()
            valid = None
            if self.label_manager.has_ignore_label:
                valid = 1.0 - target[:, -1:].float()
        else:
            num_classes = self.label_manager.num_segmentation_heads
            valid = None
            target_indices = target[:, 0].long()
            if self.label_manager.has_ignore_label:
                valid = (target_indices != self.label_manager.ignore_label).unsqueeze(1).float()
                target_indices = target_indices.masked_fill(valid[:, 0] == 0, 0)
            target_onehot = F.one_hot(target_indices, num_classes=num_classes)
            target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()

        if valid is not None:
            output = output * valid
            target_onehot = target_onehot * valid
        return _NormalizedTargets(output=output, target=target_onehot)

    @staticmethod
    def _validate_final_shapes(component: str, output: Tensor, target: Tensor) -> None:
        if output.ndim != 5 or target.ndim != 5:
            raise ValueError(
                f"{component} loss shape mismatch: output={tuple(output.shape)}, "
                f"target={tuple(target.shape)}; expected 5D tensors"
            )
        if output.shape[0] != target.shape[0] or output.shape[2:] != target.shape[2:]:
            raise ValueError(
                f"{component} loss shape mismatch: output={tuple(output.shape)}, "
                f"target={tuple(target.shape)}"
            )

    @staticmethod
    def _validate_dtm_shapes(output: Tensor, dtm: Tensor, valid_mask: Tensor) -> None:
        if dtm.ndim == 5:
            dtm_batch = dtm.shape[0]
            dtm_spatial = dtm.shape[2:]
        elif dtm.ndim == 6:
            dtm_batch = dtm.shape[0]
            dtm_spatial = dtm.shape[2:5]
        else:
            raise ValueError(
                f"DTM loss shape mismatch: output={tuple(output.shape)}, "
                f"dtm={tuple(dtm.shape)}, valid_mask={tuple(valid_mask.shape)}; "
                "expected DTM to be 5D or 6D"
            )
        if output.shape[0] != dtm_batch or output.shape[2:] != dtm_spatial:
            raise ValueError(
                f"DTM loss shape mismatch: output={tuple(output.shape)}, "
                f"dtm={tuple(dtm.shape)}, valid_mask={tuple(valid_mask.shape)}"
            )

    def _dtm_component(
        self, output: Tensor, target: Tensor, auxiliary: Optional[AuxiliaryTargets], epoch: Optional[int]
    ) -> Tensor:
        zero = output.sum() * 0
        if self.config.dtm_mode == "disabled":
            return zero
        if auxiliary is None or self.dtm_loss is None:
            if not self._missing_auxiliary_warned:
                warnings.warn(
                    "DTM loss is enabled but AuxiliaryTargets are missing or incomplete; "
                    "DTM component is zero.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._missing_auxiliary_warned = True
            return zero

        dtm, valid_mask = auxiliary.final_resolution()
        if dtm is None or valid_mask is None:
            if not self._missing_auxiliary_warned:
                warnings.warn(
                    "DTM loss is enabled but AuxiliaryTargets are missing or incomplete; "
                    "DTM component is zero.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._missing_auxiliary_warned = True
            return zero
        self._validate_dtm_shapes(output, dtm, valid_mask)
        return self._as_scalar(self.dtm_loss(target, output, dtm, valid_mask), output)

    def forward(
        self,
        output,
        target,
        auxiliary: Optional[AuxiliaryTargets] = None,
        epoch: Optional[int] = None,
    ) -> LossBreakdown:
        segmentation = self._as_scalar(self.segmentation_loss(output, target), output[0] if isinstance(output, (list, tuple)) else output)
        output_final = _first_resolution(output)
        target_final = _first_resolution(target)

        if self.config.projection_consistency_weight:
            self._validate_final_shapes("Projection", output_final, target_final)
        if self.config.dtm_mode != "disabled":
            self._validate_final_shapes("DTM", output_final, target_final)

        projection = output_final.sum() * 0
        normalized = None
        if self.config.projection_consistency_weight or self.config.dtm_mode != "disabled":
            normalized = self._normalize_final_targets(output_final, target_final)

        if self.config.projection_consistency_weight and self.projection_loss is not None:
            projection_input, target_onehot = normalized.output, normalized.target
            projection = self._as_scalar(self.projection_loss(projection_input, target_onehot), output_final)

        normalized_output = normalized.output if normalized is not None else output_final
        normalized_target = normalized.target if normalized is not None else target_final
        dtm = self._dtm_component(normalized_output, normalized_target, auxiliary, epoch)
        total = segmentation + self.config.projection_consistency_weight * projection
        if self.config.dtm_mode == "active" and auxiliary is not None and self.dtm_loss is not None:
            if epoch is not None and epoch >= self.config.dtm_activate_after_epoch:
                total = total + self.config.dtm_weight * self.config.dtm_active_weight * dtm
        return LossBreakdown(total=total, segmentation=segmentation, projection=projection, dtm=dtm)
