from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nnunetv2.network_modules.projection_consistency import (
    CombinedSegProjLoss,
)
from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
from nnunetv2.training.loss.composite_loss import (
    CompositeSegmentationLoss,
    LossBreakdown,
)
from nnunetv2.training.pipeline_config import EnhancementConfig


class RecordingSegmentationLoss(nn.Module):
    def __init__(self, value=2.0):
        super().__init__()
        self.value = value
        self.targets = []

    def forward(self, output, target):
        self.targets.append(target)
        tensor = output[0] if isinstance(output, list) else output
        return tensor.mean() * 0 + self.value


class RecordingProjectionLoss(nn.Module):
    def __init__(self, value=3.0):
        super().__init__()
        self.value = value
        self.calls = []

    def forward(self, output, target_onehot):
        self.calls.append((output, target_onehot))
        return output.mean() * 0 + self.value


class RecordingDTMLoss(nn.Module):
    def __init__(self, value=5.0):
        super().__init__()
        self.value = value
        self.calls = []

    def forward(self, target, output, dtm, valid_mask):
        self.calls.append((target, output, dtm, valid_mask))
        return output.mean() * 0 + self.value


def label_manager(*, regions=False, ignore=None, classes=2):
    return SimpleNamespace(
        has_regions=regions,
        has_ignore_label=ignore is not None,
        ignore_label=ignore,
        num_segmentation_heads=classes,
    )


def base_output():
    return torch.zeros((1, 2, 2, 2, 2), requires_grad=True)


def test_segmentation_only_returns_named_breakdown_and_scalar_gradients():
    seg = RecordingSegmentationLoss()
    loss = CompositeSegmentationLoss(
        seg, EnhancementConfig(), label_manager()
    )

    breakdown = loss(base_output(), torch.zeros((1, 1, 2, 2, 2), dtype=torch.long))

    assert isinstance(breakdown, LossBreakdown)
    assert breakdown.total.item() == pytest.approx(2.0)
    assert breakdown.segmentation.item() == pytest.approx(2.0)
    assert breakdown.projection.item() == pytest.approx(0.0)
    assert breakdown.dtm.item() == pytest.approx(0.0)
    breakdown.total.backward()
    assert torch.isfinite(breakdown.total)


def test_projection_is_weighted_and_uses_final_resolution_one_hot_target():
    proj = RecordingProjectionLoss()
    loss = CompositeSegmentationLoss(
        RecordingSegmentationLoss(),
        EnhancementConfig(projection_angles_deg=(0.0,), projection_consistency_weight=2.0),
        label_manager(classes=2),
        projection_loss=proj,
    )
    output = [base_output(), torch.zeros((1, 2, 1, 1, 1), requires_grad=True)]
    target = [torch.tensor([[[[[1, 0], [0, 1]], [[0, 1], [1, 0]]]]]),
              torch.zeros((1, 1, 1, 1, 1), dtype=torch.long)]

    breakdown = loss(output, target)

    assert breakdown.total.item() == pytest.approx(8.0)
    assert len(proj.calls) == 1
    assert proj.calls[0][0] is output[0]
    assert proj.calls[0][1].shape == (1, 2, 2, 2, 2)
    assert torch.equal(proj.calls[0][1][:, 1], target[0][:, 0].float())


def test_dtm_monitor_is_reported_but_not_added_to_total():
    dtm = RecordingDTMLoss()
    loss = CompositeSegmentationLoss(
        RecordingSegmentationLoss(),
        EnhancementConfig(dtm_mode="monitor", dtm_weight=9.0),
        label_manager(),
        dtm_loss=dtm,
    )
    auxiliary = AuxiliaryTargets(
        dtm=torch.ones((1, 1, 2, 2, 2)),
        valid_mask=torch.ones((1, 1, 2, 2, 2)),
    )

    breakdown = loss(base_output(), torch.zeros((1, 1, 2, 2, 2), dtype=torch.long), auxiliary, epoch=0)

    assert breakdown.dtm.item() == pytest.approx(5.0)
    assert breakdown.total.item() == pytest.approx(2.0)
    assert len(dtm.calls) == 1


def test_dtm_active_weight_starts_after_configured_epoch():
    dtm = RecordingDTMLoss()
    loss = CompositeSegmentationLoss(
        RecordingSegmentationLoss(),
        EnhancementConfig(dtm_mode="active", dtm_weight=2.0, dtm_activate_after_epoch=3,
                          dtm_active_weight=0.5),
        label_manager(),
        dtm_loss=dtm,
    )
    auxiliary = AuxiliaryTargets(torch.ones((1, 1, 2, 2, 2)), torch.ones((1, 1, 2, 2, 2)))
    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)

    before = loss(base_output(), target, auxiliary, epoch=2)
    after = loss(base_output(), target, auxiliary, epoch=3)

    assert before.total.item() == pytest.approx(2.0)
    assert after.total.item() == pytest.approx(2.0 + 2.0 * 0.5 * 5.0)


def test_missing_auxiliary_data_keeps_dtm_zero_without_calling_loss():
    dtm = RecordingDTMLoss()
    loss = CompositeSegmentationLoss(
        RecordingSegmentationLoss(), EnhancementConfig(dtm_mode="active", dtm_weight=1.0),
        label_manager(), dtm_loss=dtm,
    )

    with pytest.warns(RuntimeWarning, match="AuxiliaryTargets are missing"):
        breakdown = loss(base_output(), torch.zeros((1, 1, 2, 2, 2), dtype=torch.long), epoch=10)

    assert breakdown.dtm.item() == 0.0
    assert len(dtm.calls) == 0


def test_empty_auxiliary_targets_warn_once_and_keep_dtm_zero():
    dtm = RecordingDTMLoss()
    loss = CompositeSegmentationLoss(
        RecordingSegmentationLoss(), EnhancementConfig(dtm_mode="active"),
        label_manager(), dtm_loss=dtm,
    )
    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)

    with pytest.warns(RuntimeWarning, match="AuxiliaryTargets are missing") as captured:
        first = loss(base_output(), target, AuxiliaryTargets.empty(), epoch=10)
        second = loss(base_output(), target, AuxiliaryTargets.empty(), epoch=10)

    assert len(captured) == 1
    assert first.dtm.item() == 0.0
    assert second.dtm.item() == 0.0
    assert len(dtm.calls) == 0


def test_partially_populated_auxiliary_targets_warn_once_and_keep_dtm_zero():
    dtm = RecordingDTMLoss()
    loss = CompositeSegmentationLoss(
        RecordingSegmentationLoss(), EnhancementConfig(dtm_mode="active"),
        label_manager(), dtm_loss=dtm,
    )
    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)
    partial = AuxiliaryTargets(dtm=torch.ones((1, 1, 2, 2, 2)))

    with pytest.warns(RuntimeWarning, match="AuxiliaryTargets are missing") as captured:
        first = loss(base_output(), target, partial, epoch=10)
        second = loss(base_output(), target, partial, epoch=10)

    assert len(captured) == 1
    assert first.dtm.item() == 0.0
    assert second.dtm.item() == 0.0
    assert len(dtm.calls) == 0


@pytest.mark.parametrize(
    "target_shape",
    [(2, 1, 2, 2, 2), (1, 1, 3, 2, 2)],
)
def test_projection_rejects_final_batch_or_spatial_shape_mismatch(target_shape):
    projection = RecordingProjectionLoss()
    loss = CompositeSegmentationLoss(
        RecordingSegmentationLoss(),
        EnhancementConfig(projection_angles_deg=(0.0,), projection_consistency_weight=1.0),
        label_manager(), projection_loss=projection,
    )
    output = base_output()
    target = torch.zeros(target_shape, dtype=torch.long)

    with pytest.raises(
        ValueError,
        match=r"Projection loss shape mismatch: output=\(1, 2, 2, 2, 2\), target=.*",
    ) as error:
        loss(output, target)

    assert str(target_shape) in str(error.value)
    assert len(projection.calls) == 0


def test_dtm_rejects_final_output_target_shape_mismatch():
    dtm = RecordingDTMLoss()
    loss = CompositeSegmentationLoss(
        RecordingSegmentationLoss(), EnhancementConfig(dtm_mode="active"),
        label_manager(), dtm_loss=dtm,
    )
    output = base_output()
    target = torch.zeros((2, 1, 2, 2, 2), dtype=torch.long)
    auxiliary = AuxiliaryTargets(
        dtm=torch.ones((2, 1, 2, 2, 2)),
        valid_mask=torch.ones((2, 1, 2, 2, 2)),
    )

    with pytest.raises(
        ValueError,
        match=r"DTM loss shape mismatch: output=.*target=.*",
    ) as error:
        loss(output, target, auxiliary, epoch=10)

    assert "output=(1, 2, 2, 2, 2)" in str(error.value)
    assert "target=(2, 1, 2, 2, 2)" in str(error.value)
    assert len(dtm.calls) == 0


@pytest.mark.parametrize(
    "dtm_shape",
    [(2, 1, 2, 2, 2, 2), (1, 1, 3, 2, 2, 2)],
)
def test_dtm_rejects_batch_or_spatial_shape_mismatch(dtm_shape):
    dtm = RecordingDTMLoss()
    loss = CompositeSegmentationLoss(
        RecordingSegmentationLoss(), EnhancementConfig(dtm_mode="active"),
        label_manager(), dtm_loss=dtm,
    )
    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)
    auxiliary = AuxiliaryTargets(
        dtm=torch.ones(dtm_shape), valid_mask=torch.ones(dtm_shape)
    )

    with pytest.raises(
        ValueError,
        match=r"DTM loss shape mismatch: output=.*dtm=.*",
    ) as error:
        loss(base_output(), target, auxiliary, epoch=10)

    assert str(dtm_shape) in str(error.value)
    assert len(dtm.calls) == 0


def test_regions_and_ignore_label_are_normalized_for_projection_mask():
    proj = RecordingProjectionLoss()
    loss = CompositeSegmentationLoss(
        RecordingSegmentationLoss(),
        EnhancementConfig(projection_angles_deg=(0.0,), projection_consistency_weight=1.0),
        label_manager(regions=True, ignore=99, classes=2), projection_loss=proj,
    )
    target = torch.zeros((1, 3, 2, 2, 2), dtype=torch.float32)
    target[0, 0, 0] = torch.tensor([[1, 0], [0, 1]])
    target[0, 1, 0] = torch.tensor([[0, 1], [1, 0]])
    target[0, 2, 0, 0, 0] = 1

    loss(base_output(), target)

    pred, target_onehot = proj.calls[0]
    assert target_onehot.shape[1] == 2
    assert target_onehot[0, :, 0, 0, 0].tolist() == [0.0, 0.0]
    assert pred[0, :, 0, 0, 0].tolist() == [0.0, 0.0]


def test_dtm_receives_projection_normalized_final_target_while_segmentation_keeps_target_tree():
    seg = RecordingSegmentationLoss()
    proj = RecordingProjectionLoss()
    dtm = RecordingDTMLoss()
    loss = CompositeSegmentationLoss(
        seg,
        EnhancementConfig(
            projection_angles_deg=(0.0,),
            projection_consistency_weight=1.0,
            dtm_mode="active",
            dtm_weight=1.0,
            dtm_active_weight=1.0,
        ),
        label_manager(regions=True, ignore=99, classes=2),
        projection_loss=proj,
        dtm_loss=dtm,
    )
    final_target = torch.zeros((1, 3, 2, 2, 2), dtype=torch.float32)
    final_target[0, 0, 0] = torch.tensor([[1, 0], [0, 1]])
    final_target[0, 1, 0] = torch.tensor([[0, 1], [1, 0]])
    final_target[0, 2, 0, 0, 0] = 1
    target_tree = [final_target, torch.zeros((1, 3, 1, 1, 1))]
    auxiliary = AuxiliaryTargets(
        torch.ones((1, 1, 2, 2, 2)), torch.ones((1, 1, 2, 2, 2))
    )

    loss([base_output(), base_output()], target_tree, auxiliary, epoch=0)

    assert seg.targets[-1] is target_tree
    assert torch.equal(dtm.calls[0][0], proj.calls[0][1])
    assert dtm.calls[0][0].shape == (1, 2, 2, 2, 2)
    assert dtm.calls[0][0][0, :, 0, 0, 0].tolist() == [0.0, 0.0]


def test_deep_supervision_segmentation_loss_receives_original_tree_and_combined_compatibility_is_scalar():
    seg = RecordingSegmentationLoss()
    composite = CompositeSegmentationLoss(seg, EnhancementConfig(), label_manager())
    output = [base_output(), base_output()]
    target = [torch.zeros((1, 1, 2, 2, 2), dtype=torch.long), torch.zeros((1, 1, 1, 1, 1), dtype=torch.long)]

    result = composite(output, target)

    assert seg.targets[-1] is target
    assert result.total.ndim == 0
    combined = CombinedSegProjLoss(seg, nn.Identity(), 0.0, label_manager())
    scalar = combined(base_output(), torch.zeros((1, 1, 2, 2, 2), dtype=torch.long))
    assert isinstance(scalar, torch.Tensor) and scalar.ndim == 0
