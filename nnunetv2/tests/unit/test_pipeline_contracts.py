import pytest
import torch

from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
from nnunetv2.training.pipeline_config import EnhancementConfig


def test_enhancement_config_defaults_are_disabled_and_immutable():
    config = EnhancementConfig()

    assert config.enable_anisotropic_attention is False
    assert config.enable_tubular_enhancement is False
    assert config.projection_angles_deg == ()
    assert config.projection_consistency_weight == 0.0
    assert config.dtm_mode == "disabled"
    assert config.dtm_weight == 0.0
    assert config.dtm_activate_after_epoch == 0
    assert config.dtm_active_weight == 0.5
    with pytest.raises((AttributeError, TypeError)):
        config.dtm_mode = "active"


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        (
            "nnUNetTrainerAnisoStent",
            dict(
                enable_anisotropic_attention=True,
                enable_tubular_enhancement=True,
                projection_angles_deg=(),
                projection_consistency_weight=0.0,
                dtm_mode="disabled",
            ),
        ),
        (
            "nnUNetTrainerAnisoStent_ProjCons",
            dict(
                enable_anisotropic_attention=True,
                enable_tubular_enhancement=True,
                projection_angles_deg=(0.0, 30.0, 60.0),
                projection_consistency_weight=0.1,
                dtm_mode="disabled",
            ),
        ),
        (
            "nnUNetTrainerDTM",
            dict(
                dtm_mode="monitor",
                dtm_weight=0.0,
                dtm_activate_after_epoch=250,
                dtm_active_weight=0.5,
            ),
        ),
        (
            "nnUNetTrainerDTM_GSL",
            dict(
                dtm_mode="active",
                dtm_weight=1.0,
                dtm_activate_after_epoch=250,
                dtm_active_weight=0.5,
            ),
        ),
    ],
)
def test_legacy_presets_preserve_existing_trainer_values(preset, expected):
    config = EnhancementConfig.from_legacy_preset(preset)

    for field, value in expected.items():
        assert getattr(config, field) == value


def test_legacy_preset_accepts_short_names_and_rejects_unknown_names():
    assert EnhancementConfig.from_legacy_preset("aniso_stent").enable_tubular_enhancement
    assert EnhancementConfig.from_legacy_preset("dtm_gsl").dtm_mode == "active"
    with pytest.raises(ValueError, match="Unknown legacy preset"):
        EnhancementConfig.from_legacy_preset("not-a-trainer")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"projection_consistency_weight": -0.1},
        {"dtm_weight": -0.1},
        {"dtm_active_weight": -0.1},
        {"dtm_activate_after_epoch": -1},
        {"dtm_mode": "sometimes"},
        {"projection_consistency_weight": 0.1, "projection_angles_deg": ()},
    ],
)
def test_enhancement_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        EnhancementConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"projection_angles_deg": (float("nan"),)},
        {"projection_angles_deg": (float("inf"),)},
        {"projection_consistency_weight": float("nan")},
        {"dtm_weight": float("inf")},
        {"dtm_active_weight": float("nan")},
    ],
)
def test_enhancement_config_rejects_non_finite_numeric_values(kwargs):
    with pytest.raises(ValueError, match="finite"):
        EnhancementConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtm_activate_after_epoch": 1.5},
        {"dtm_activate_after_epoch": True},
        {"enable_anisotropic_attention": 1},
        {"enable_tubular_enhancement": "yes"},
    ],
)
def test_enhancement_config_rejects_non_integer_epochs_and_non_boolean_flags(kwargs):
    with pytest.raises(TypeError):
        EnhancementConfig(**kwargs)


def test_auxiliary_targets_to_moves_nested_tensor_lists_recursively():
    targets = AuxiliaryTargets(
        dtm=[torch.ones(2, 1, 8, 8), [torch.ones(2, 1, 4, 4)]],
        valid_mask=[torch.ones(2, 1, 8, 8), [torch.ones(2, 1, 4, 4)]],
    )

    moved = targets.to(torch.device("meta"))

    assert moved.dtm[0].device.type == "meta"
    assert moved.dtm[1][0].device.type == "meta"
    assert moved.valid_mask[1][0].shape == (2, 1, 4, 4)
    assert targets.dtm[0].device.type == "cpu"


def test_empty_auxiliary_targets_have_no_final_resolution_data():
    targets = AuxiliaryTargets.empty()

    assert targets.dtm is None
    assert targets.valid_mask is None
    assert targets.final_resolution() == (None, None)
    assert targets.to("cpu") == targets


def test_auxiliary_targets_select_first_deep_supervision_resolution():
    dtm = [torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 4, 4)]
    valid_mask = [torch.ones(2, 1, 8, 8), torch.ones(2, 1, 4, 4)]

    final_dtm, final_mask = AuxiliaryTargets(dtm, valid_mask).final_resolution()

    assert final_dtm.shape == (2, 1, 8, 8)
    assert final_mask.shape == (2, 1, 8, 8)


@pytest.mark.parametrize(
    ("dtm", "valid_mask", "expected_shapes"),
    [
        (
            torch.zeros(2, 1, 8, 8),
            torch.ones(3, 1, 8, 8),
            ((2, 1, 8, 8), (3, 1, 8, 8)),
        ),
        (
            [torch.zeros(2, 1, 8, 8)],
            [torch.ones(2, 1, 4, 4)],
            ((2, 1, 8, 8), (2, 1, 4, 4)),
        ),
    ],
)
def test_auxiliary_targets_reject_mismatched_batch_or_spatial_shapes(
    dtm, valid_mask, expected_shapes
):
    with pytest.raises(ValueError) as error:
        AuxiliaryTargets(dtm, valid_mask)

    assert str(expected_shapes[0]) in str(error.value)
    assert str(expected_shapes[1]) in str(error.value)


def test_auxiliary_targets_list_length_error_includes_all_per_level_shapes():
    dtm = [torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 4, 4)]
    valid_mask = [torch.ones(2, 1, 8, 8)]

    with pytest.raises(ValueError) as error:
        AuxiliaryTargets(dtm, valid_mask)

    message = str(error.value)
    assert "dtm_shapes=[(2, 1, 8, 8), (2, 1, 4, 4)]" in message
    assert "valid_mask_shapes=[(2, 1, 8, 8)]" in message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dtm", [torch.zeros(2, 1, 8, 8), "not-a-tensor"]),
        ("valid_mask", [torch.ones(2, 1, 8, 8), {"bad": "tree"}]),
        ("dtm", []),
        ("valid_mask", ()),
    ],
)
def test_auxiliary_targets_validate_each_non_none_tree_at_construction(field, value):
    kwargs = {"dtm": None, "valid_mask": None}
    kwargs[field] = value

    with pytest.raises((TypeError, ValueError)):
        AuxiliaryTargets(**kwargs)
