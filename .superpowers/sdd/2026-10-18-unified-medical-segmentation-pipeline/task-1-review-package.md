COMMITS
ce1606b (HEAD -> master) feat: add unified pipeline contracts
STAT
 nnunetv2/tests/unit/test_pipeline_contracts.py | 141 +++++++++++++++++++++++++
 nnunetv2/training/__init__.py                  |   4 +
 nnunetv2/training/auxiliary_targets.py         |  87 +++++++++++++++
 nnunetv2/training/pipeline_config.py           |  82 ++++++++++++++
 4 files changed, 314 insertions(+)
DIFF
diff --git a/nnunetv2/tests/unit/test_pipeline_contracts.py b/nnunetv2/tests/unit/test_pipeline_contracts.py
new file mode 100644
index 0000000..3e3746a
--- /dev/null
+++ b/nnunetv2/tests/unit/test_pipeline_contracts.py
@@ -0,0 +1,141 @@
+import pytest
+import torch
+
+from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
+from nnunetv2.training.pipeline_config import EnhancementConfig
+
+
+def test_enhancement_config_defaults_are_disabled_and_immutable():
+    config = EnhancementConfig()
+
+    assert config.enable_anisotropic_attention is False
+    assert config.enable_tubular_enhancement is False
+    assert config.projection_angles_deg == ()
+    assert config.projection_consistency_weight == 0.0
+    assert config.dtm_mode == "disabled"
+    assert config.dtm_weight == 0.0
+    assert config.dtm_activate_after_epoch == 0
+    assert config.dtm_active_weight == 0.5
+    with pytest.raises((AttributeError, TypeError)):
+        config.dtm_mode = "active"
+
+
+@pytest.mark.parametrize(
+    ("preset", "expected"),
+    [
+        (
+            "nnUNetTrainerAnisoStent",
+            dict(
+                enable_anisotropic_attention=True,
+                enable_tubular_enhancement=True,
+                projection_angles_deg=(),
+                projection_consistency_weight=0.0,
+                dtm_mode="disabled",
+            ),
+        ),
+        (
+            "nnUNetTrainerAnisoStent_ProjCons",
+            dict(
+                enable_anisotropic_attention=True,
+                enable_tubular_enhancement=True,
+                projection_angles_deg=(0.0, 30.0, 60.0),
+                projection_consistency_weight=0.1,
+                dtm_mode="disabled",
+            ),
+        ),
+        (
+            "nnUNetTrainerDTM",
+            dict(dtm_mode="monitor", dtm_activate_after_epoch=250, dtm_active_weight=0.5),
+        ),
+        (
+            "nnUNetTrainerDTM_GSL",
+            dict(dtm_mode="active", dtm_activate_after_epoch=250, dtm_active_weight=0.5),
+        ),
+    ],
+)
+def test_legacy_presets_preserve_existing_trainer_values(preset, expected):
+    config = EnhancementConfig.from_legacy_preset(preset)
+
+    for field, value in expected.items():
+        assert getattr(config, field) == value
+
+
+def test_legacy_preset_accepts_short_names_and_rejects_unknown_names():
+    assert EnhancementConfig.from_legacy_preset("aniso_stent").enable_tubular_enhancement
+    assert EnhancementConfig.from_legacy_preset("dtm_gsl").dtm_mode == "active"
+    with pytest.raises(ValueError, match="Unknown legacy preset"):
+        EnhancementConfig.from_legacy_preset("not-a-trainer")
+
+
+@pytest.mark.parametrize(
+    "kwargs",
+    [
+        {"projection_consistency_weight": -0.1},
+        {"dtm_weight": -0.1},
+        {"dtm_active_weight": -0.1},
+        {"dtm_activate_after_epoch": -1},
+        {"dtm_mode": "sometimes"},
+        {"projection_consistency_weight": 0.1, "projection_angles_deg": ()},
+    ],
+)
+def test_enhancement_config_rejects_invalid_values(kwargs):
+    with pytest.raises(ValueError):
+        EnhancementConfig(**kwargs)
+
+
+def test_auxiliary_targets_to_moves_nested_tensor_lists_recursively():
+    targets = AuxiliaryTargets(
+        dtm=[torch.ones(2, 1, 8, 8), [torch.ones(2, 1, 4, 4)]],
+        valid_mask=[torch.ones(2, 1, 8, 8), [torch.ones(2, 1, 4, 4)]],
+    )
+
+    moved = targets.to(torch.device("meta"))
+
+    assert moved.dtm[0].device.type == "meta"
+    assert moved.dtm[1][0].device.type == "meta"
+    assert moved.valid_mask[1][0].shape == (2, 1, 4, 4)
+    assert targets.dtm[0].device.type == "cpu"
+
+
+def test_empty_auxiliary_targets_have_no_final_resolution_data():
+    targets = AuxiliaryTargets.empty()
+
+    assert targets.dtm is None
+    assert targets.valid_mask is None
+    assert targets.final_resolution() == (None, None)
+    assert targets.to("cpu") == targets
+
+
+def test_auxiliary_targets_select_first_deep_supervision_resolution():
+    dtm = [torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 4, 4)]
+    valid_mask = [torch.ones(2, 1, 8, 8), torch.ones(2, 1, 4, 4)]
+
+    final_dtm, final_mask = AuxiliaryTargets(dtm, valid_mask).final_resolution()
+
+    assert final_dtm.shape == (2, 1, 8, 8)
+    assert final_mask.shape == (2, 1, 8, 8)
+
+
+@pytest.mark.parametrize(
+    ("dtm", "valid_mask", "expected_shapes"),
+    [
+        (
+            torch.zeros(2, 1, 8, 8),
+            torch.ones(3, 1, 8, 8),
+            ((2, 1, 8, 8), (3, 1, 8, 8)),
+        ),
+        (
+            [torch.zeros(2, 1, 8, 8)],
+            [torch.ones(2, 1, 4, 4)],
+            ((2, 1, 8, 8), (2, 1, 4, 4)),
+        ),
+    ],
+)
+def test_auxiliary_targets_reject_mismatched_batch_or_spatial_shapes(
+    dtm, valid_mask, expected_shapes
+):
+    with pytest.raises(ValueError) as error:
+        AuxiliaryTargets(dtm, valid_mask)
+
+    assert str(expected_shapes[0]) in str(error.value)
+    assert str(expected_shapes[1]) in str(error.value)
diff --git a/nnunetv2/training/__init__.py b/nnunetv2/training/__init__.py
index e69de29..0853314 100644
--- a/nnunetv2/training/__init__.py
+++ b/nnunetv2/training/__init__.py
@@ -0,0 +1,4 @@
+from .auxiliary_targets import AuxiliaryTargets
+from .pipeline_config import EnhancementConfig
+
+__all__ = ["AuxiliaryTargets", "EnhancementConfig"]
diff --git a/nnunetv2/training/auxiliary_targets.py b/nnunetv2/training/auxiliary_targets.py
new file mode 100644
index 0000000..6951220
--- /dev/null
+++ b/nnunetv2/training/auxiliary_targets.py
@@ -0,0 +1,87 @@
+"""Contracts for optional DTM and validity-mask training targets."""
+
+from dataclasses import dataclass
+from typing import Optional, TypeAlias
+
+import torch
+from torch import Tensor
+
+
+TensorTree: TypeAlias = Tensor | list["TensorTree"] | tuple["TensorTree", ...]
+
+
+def _shape(value: TensorTree) -> tuple[int, ...]:
+    if isinstance(value, torch.Tensor):
+        return tuple(value.shape)
+    raise TypeError(f"Auxiliary targets must contain tensors, got {type(value).__name__}")
+
+
+def _validate_matching_shapes(dtm: TensorTree, valid_mask: TensorTree) -> None:
+    if isinstance(dtm, torch.Tensor) and isinstance(valid_mask, torch.Tensor):
+        if tuple(dtm.shape) != tuple(valid_mask.shape):
+            raise ValueError(
+                "DTM and valid_mask shapes must match: "
+                f"dtm={tuple(dtm.shape)}, valid_mask={tuple(valid_mask.shape)}"
+            )
+        return
+
+    if isinstance(dtm, (list, tuple)) and isinstance(valid_mask, (list, tuple)):
+        if len(dtm) != len(valid_mask):
+            raise ValueError(
+                "DTM and valid_mask list lengths must match: "
+                f"dtm={len(dtm)}, valid_mask={len(valid_mask)}"
+            )
+        for dtm_item, mask_item in zip(dtm, valid_mask):
+            _validate_matching_shapes(dtm_item, mask_item)
+        return
+
+    dtm_shape = _shape(dtm) if isinstance(dtm, torch.Tensor) else dtm
+    mask_shape = _shape(valid_mask) if isinstance(valid_mask, torch.Tensor) else valid_mask
+    raise ValueError(
+        "DTM and valid_mask must have matching tensor/list structure: "
+        f"dtm={dtm_shape}, valid_mask={mask_shape}"
+    )
+
+
+def _move_to(value: Optional[TensorTree], device: torch.device | str) -> Optional[TensorTree]:
+    if value is None:
+        return None
+    if isinstance(value, torch.Tensor):
+        return value.to(device)
+    if isinstance(value, list):
+        return [_move_to(item, device) for item in value]  # type: ignore[list-item]
+    if isinstance(value, tuple):
+        return tuple(_move_to(item, device) for item in value)  # type: ignore[return-value]
+    raise TypeError(f"Auxiliary targets must contain tensors, got {type(value).__name__}")
+
+
+def _first_resolution(value: Optional[TensorTree]) -> Optional[Tensor]:
+    if value is None:
+        return None
+    if isinstance(value, torch.Tensor):
+        return value
+    if not value:
+        raise ValueError("Auxiliary target resolution lists cannot be empty")
+    return _first_resolution(value[0])
+
+
+@dataclass(frozen=True)
+class AuxiliaryTargets:
+    """Optional DTM and validity-mask tensors, including deep-supervision lists."""
+
+    dtm: Optional[TensorTree] = None
+    valid_mask: Optional[TensorTree] = None
+
+    def __post_init__(self) -> None:
+        if self.dtm is not None and self.valid_mask is not None:
+            _validate_matching_shapes(self.dtm, self.valid_mask)
+
+    @classmethod
+    def empty(cls) -> "AuxiliaryTargets":
+        return cls()
+
+    def to(self, device: torch.device | str) -> "AuxiliaryTargets":
+        return type(self)(_move_to(self.dtm, device), _move_to(self.valid_mask, device))
+
+    def final_resolution(self) -> tuple[Optional[Tensor], Optional[Tensor]]:
+        return _first_resolution(self.dtm), _first_resolution(self.valid_mask)
diff --git a/nnunetv2/training/pipeline_config.py b/nnunetv2/training/pipeline_config.py
new file mode 100644
index 0000000..6592bb2
--- /dev/null
+++ b/nnunetv2/training/pipeline_config.py
@@ -0,0 +1,82 @@
+"""Configuration contracts for the unified enhancement pipeline."""
+
+from dataclasses import dataclass
+from typing import Literal
+
+
+DTMMode = Literal["disabled", "monitor", "active"]
+
+
+@dataclass(frozen=True)
+class EnhancementConfig:
+    """Immutable, validated configuration shared by pipeline components."""
+
+    enable_anisotropic_attention: bool = False
+    enable_tubular_enhancement: bool = False
+    projection_angles_deg: tuple[float, ...] = ()
+    projection_consistency_weight: float = 0.0
+    dtm_mode: DTMMode = "disabled"
+    dtm_weight: float = 0.0
+    dtm_activate_after_epoch: int = 0
+    dtm_active_weight: float = 0.5
+
+    def __post_init__(self) -> None:
+        angles = tuple(float(angle) for angle in self.projection_angles_deg)
+        object.__setattr__(self, "projection_angles_deg", angles)
+
+        if self.dtm_mode not in ("disabled", "monitor", "active"):
+            raise ValueError(
+                f"dtm_mode must be 'disabled', 'monitor', or 'active', got {self.dtm_mode!r}"
+            )
+        if self.projection_consistency_weight < 0:
+            raise ValueError("projection_consistency_weight must be non-negative")
+        if self.dtm_weight < 0:
+            raise ValueError("dtm_weight must be non-negative")
+        if self.dtm_active_weight < 0:
+            raise ValueError("dtm_active_weight must be non-negative")
+        if self.dtm_activate_after_epoch < 0:
+            raise ValueError("dtm_activate_after_epoch must be non-negative")
+        if not angles and self.projection_consistency_weight != 0:
+            raise ValueError(
+                "projection_angles_deg cannot be empty when projection_consistency_weight is non-zero"
+            )
+
+    @classmethod
+    def from_legacy_preset(cls, name: str) -> "EnhancementConfig":
+        """Return the configuration represented by an existing trainer name."""
+
+        normalized = name.removesuffix("_A").removesuffix("_B")
+        aliases = {
+            "aniso_stent": "nnUNetTrainerAnisoStent",
+            "aniso_stent_projcons": "nnUNetTrainerAnisoStent_ProjCons",
+            "dtm": "nnUNetTrainerDTM",
+            "dtm_gsl": "nnUNetTrainerDTM_GSL",
+        }
+        normalized = aliases.get(normalized, normalized)
+
+        if normalized == "nnUNetTrainerAnisoStent":
+            return cls(
+                enable_anisotropic_attention=True,
+                enable_tubular_enhancement=True,
+            )
+        if normalized == "nnUNetTrainerAnisoStent_ProjCons":
+            return cls(
+                enable_anisotropic_attention=True,
+                enable_tubular_enhancement=True,
+                projection_angles_deg=(0.0, 30.0, 60.0),
+                projection_consistency_weight=0.1,
+            )
+        if normalized == "nnUNetTrainerDTM":
+            return cls(
+                dtm_mode="monitor",
+                dtm_activate_after_epoch=250,
+                dtm_active_weight=0.5,
+            )
+        if normalized == "nnUNetTrainerDTM_GSL":
+            return cls(
+                dtm_mode="active",
+                dtm_weight=1.0,
+                dtm_activate_after_epoch=250,
+                dtm_active_weight=0.5,
+            )
+        raise ValueError(f"Unknown legacy preset: {name!r}")

