from types import SimpleNamespace

import torch
from torch import nn

from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
from nnunetv2.training.pipeline_config import EnhancementConfig
from nnunetv2.training.pipeline_trainer import build_composite_loss
from nnunetv2.network_modules.enhancement_pipeline import EnhancementPipeline


class TinyBackbone(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.decoder = nn.Module()
        self.projection = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.projection(x)


class TinySegmentationLoss(nn.Module):
    def forward(self, output, target):
        return nn.functional.cross_entropy(output, target[:, 0].long())


def test_all_enabled_cpu_pipeline_loss_backward_is_finite():
    torch.manual_seed(0)
    config = EnhancementConfig(
        enable_anisotropic_attention=True,
        enable_tubular_enhancement=True,
        projection_angles_deg=(0.0,),
        projection_consistency_weight=0.25,
        dtm_mode="active",
        dtm_weight=0.5,
        dtm_activate_after_epoch=0,
        dtm_active_weight=1.0,
    )
    pipeline = EnhancementPipeline(TinyBackbone(), 1, 2, config).cpu()
    label_manager = SimpleNamespace(
        has_regions=False,
        has_ignore_label=False,
        ignore_label=None,
        num_segmentation_heads=2,
    )
    loss = build_composite_loss(config, TinySegmentationLoss(), label_manager).cpu()

    image = torch.randn(1, 1, 8, 8, 8)
    target = torch.randint(0, 2, (1, 1, 8, 8, 8))
    dtm = torch.rand(1, 1, 8, 8, 8, 2)
    valid_mask = torch.ones_like(dtm)

    output = pipeline(image)
    breakdown = loss.breakdown(
        output,
        target,
        AuxiliaryTargets(dtm=dtm, valid_mask=valid_mask),
        epoch=0,
    )
    breakdown.total.backward()

    for component in (
        breakdown.total,
        breakdown.segmentation,
        breakdown.projection,
        breakdown.dtm,
    ):
        assert torch.isfinite(component)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in pipeline.parameters()
        if parameter.requires_grad
    )
