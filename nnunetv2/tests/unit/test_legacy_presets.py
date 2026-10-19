from types import SimpleNamespace
import sys
import types

import pytest
import torch
from torch import nn


if "fft_conv_pytorch" not in sys.modules:
    sys.modules["fft_conv_pytorch"] = types.SimpleNamespace(fft_conv=None)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_AnisoStent import (
    nnUNetTrainerAnisoStent,
    nnUNetTrainerAnisoStent_A,
    nnUNetTrainerAnisoStent_B,
    nnUNetTrainerAnisoStent_ProjCons,
    nnUNetTrainerAnisoStent_ProjCons_A,
    nnUNetTrainerAnisoStent_ProjCons_B,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerDTM import (
    nnUNetTrainerDTM,
    nnUNetTrainerDTM_GSL,
)
from nnunetv2.training.nnUNetTrainer.dual_network_trainers import (
    nnUNetTrainerDTM_A,
    nnUNetTrainerDTM_B,
    nnUNetTrainerDTM_GSL_A,
    nnUNetTrainerDTM_GSL_B,
)
from nnunetv2.training.pipeline_config import EnhancementConfig
from nnunetv2.training.pipeline_trainer import (
    build_legacy_loss,
    move_auxiliary_batch,
)


LEGACY = {
    nnUNetTrainerAnisoStent: "nnUNetTrainerAnisoStent",
    nnUNetTrainerAnisoStent_A: "nnUNetTrainerAnisoStent_A",
    nnUNetTrainerAnisoStent_B: "nnUNetTrainerAnisoStent_B",
    nnUNetTrainerAnisoStent_ProjCons: "nnUNetTrainerAnisoStent_ProjCons",
    nnUNetTrainerAnisoStent_ProjCons_A: "nnUNetTrainerAnisoStent_ProjCons_A",
    nnUNetTrainerAnisoStent_ProjCons_B: "nnUNetTrainerAnisoStent_ProjCons_B",
    nnUNetTrainerDTM: "nnUNetTrainerDTM",
    nnUNetTrainerDTM_GSL: "nnUNetTrainerDTM_GSL",
    nnUNetTrainerDTM_A: "nnUNetTrainerDTM_A",
    nnUNetTrainerDTM_B: "nnUNetTrainerDTM_B",
    nnUNetTrainerDTM_GSL_A: "nnUNetTrainerDTM_GSL_A",
    nnUNetTrainerDTM_GSL_B: "nnUNetTrainerDTM_GSL_B",
}


@pytest.mark.parametrize("trainer_class,preset_name", LEGACY.items())
def test_every_legacy_trainer_exposes_its_compatibility_preset(trainer_class, preset_name):
    assert trainer_class.enhancement_config() == EnhancementConfig.from_legacy_preset(preset_name)


@pytest.mark.parametrize(
    ("trainer_a", "trainer_b"),
    [
        (nnUNetTrainerAnisoStent_A, nnUNetTrainerAnisoStent_B),
        (nnUNetTrainerAnisoStent_ProjCons_A, nnUNetTrainerAnisoStent_ProjCons_B),
        (nnUNetTrainerDTM_A, nnUNetTrainerDTM_B),
        (nnUNetTrainerDTM_GSL_A, nnUNetTrainerDTM_GSL_B),
    ],
)
def test_a_b_legacy_pairs_have_equal_configs(trainer_a, trainer_b):
    assert trainer_a.enhancement_config() == trainer_b.enhancement_config()


class _ScalarSegmentationLoss(nn.Module):
    def forward(self, output, target):
        tensor = output[0] if isinstance(output, (list, tuple)) else output
        return tensor.mean() * 0 + 1.0


def _label_manager():
    return SimpleNamespace(
        has_regions=False,
        has_ignore_label=False,
        ignore_label=None,
        num_segmentation_heads=2,
    )


def test_legacy_loss_adapter_keeps_trainer_loss_output_target_callable():
    loss = build_legacy_loss(
        "nnUNetTrainerAnisoStent",
        _ScalarSegmentationLoss(),
        _label_manager(),
    )
    output = torch.zeros((1, 2, 2, 2, 2), requires_grad=True)
    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)

    value = loss(output, target)

    assert isinstance(value, torch.Tensor)
    assert value.ndim == 0
    value.backward()
    assert output.grad is not None


def test_legacy_dtm_adapter_normalizes_6d_map_and_5d_validity_mask():
    dtm = torch.zeros((2, 1, 4, 4, 4, 2))
    valid_mask = torch.ones((2, 1, 4, 4, 4), dtype=torch.int16)

    auxiliary = move_auxiliary_batch({"dtm": dtm, "valid_mask": valid_mask}, "cpu")

    assert auxiliary.dtm.shape == dtm.shape
    assert auxiliary.valid_mask.shape == dtm.shape
    assert torch.equal(auxiliary.valid_mask[..., 0], valid_mask)
    assert torch.equal(auxiliary.valid_mask[..., 1], valid_mask)
