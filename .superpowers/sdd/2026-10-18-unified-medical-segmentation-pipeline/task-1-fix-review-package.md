COMMITS
9e92e71 (HEAD -> master) fix: harden unified pipeline contracts
STAT
 nnunetv2/tests/unit/test_pipeline_contracts.py | 72 +++++++++++++++++++++++++-
 nnunetv2/training/auxiliary_targets.py         | 31 ++++++++---
 nnunetv2/training/pipeline_config.py           | 36 +++++++++++--
 3 files changed, 127 insertions(+), 12 deletions(-)
DIFF
diff --git a/nnunetv2/tests/unit/test_pipeline_contracts.py b/nnunetv2/tests/unit/test_pipeline_contracts.py
index 3e3746a..243e342 100644
--- a/nnunetv2/tests/unit/test_pipeline_contracts.py
+++ b/nnunetv2/tests/unit/test_pipeline_contracts.py
@@ -38,25 +38,35 @@ def test_enhancement_config_defaults_are_disabled_and_immutable():
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
-            dict(dtm_mode="monitor", dtm_activate_after_epoch=250, dtm_active_weight=0.5),
+            dict(
+                dtm_mode="monitor",
+                dtm_weight=0.0,
+                dtm_activate_after_epoch=250,
+                dtm_active_weight=0.5,
+            ),
         ),
         (
             "nnUNetTrainerDTM_GSL",
-            dict(dtm_mode="active", dtm_activate_after_epoch=250, dtm_active_weight=0.5),
+            dict(
+                dtm_mode="active",
+                dtm_weight=1.0,
+                dtm_activate_after_epoch=250,
+                dtm_active_weight=0.5,
+            ),
         ),
     ],
 )
 def test_legacy_presets_preserve_existing_trainer_values(preset, expected):
     config = EnhancementConfig.from_legacy_preset(preset)
 
     for field, value in expected.items():
         assert getattr(config, field) == value
 
 
@@ -76,20 +86,49 @@ def test_legacy_preset_accepts_short_names_and_rejects_unknown_names():
         {"dtm_activate_after_epoch": -1},
         {"dtm_mode": "sometimes"},
         {"projection_consistency_weight": 0.1, "projection_angles_deg": ()},
     ],
 )
 def test_enhancement_config_rejects_invalid_values(kwargs):
     with pytest.raises(ValueError):
         EnhancementConfig(**kwargs)
 
 
+@pytest.mark.parametrize(
+    "kwargs",
+    [
+        {"projection_angles_deg": (float("nan"),)},
+        {"projection_angles_deg": (float("inf"),)},
+        {"projection_consistency_weight": float("nan")},
+        {"dtm_weight": float("inf")},
+        {"dtm_active_weight": float("nan")},
+    ],
+)
+def test_enhancement_config_rejects_non_finite_numeric_values(kwargs):
+    with pytest.raises(ValueError, match="finite"):
+        EnhancementConfig(**kwargs)
+
+
+@pytest.mark.parametrize(
+    "kwargs",
+    [
+        {"dtm_activate_after_epoch": 1.5},
+        {"dtm_activate_after_epoch": True},
+        {"enable_anisotropic_attention": 1},
+        {"enable_tubular_enhancement": "yes"},
+    ],
+)
+def test_enhancement_config_rejects_non_integer_epochs_and_non_boolean_flags(kwargs):
+    with pytest.raises(TypeError):
+        EnhancementConfig(**kwargs)
+
+
 def test_auxiliary_targets_to_moves_nested_tensor_lists_recursively():
     targets = AuxiliaryTargets(
         dtm=[torch.ones(2, 1, 8, 8), [torch.ones(2, 1, 4, 4)]],
         valid_mask=[torch.ones(2, 1, 8, 8), [torch.ones(2, 1, 4, 4)]],
     )
 
     moved = targets.to(torch.device("meta"))
 
     assert moved.dtm[0].device.type == "meta"
     assert moved.dtm[1][0].device.type == "meta"
@@ -132,10 +171,39 @@ def test_auxiliary_targets_select_first_deep_supervision_resolution():
     ],
 )
 def test_auxiliary_targets_reject_mismatched_batch_or_spatial_shapes(
     dtm, valid_mask, expected_shapes
 ):
     with pytest.raises(ValueError) as error:
         AuxiliaryTargets(dtm, valid_mask)
 
     assert str(expected_shapes[0]) in str(error.value)
     assert str(expected_shapes[1]) in str(error.value)
+
+
+def test_auxiliary_targets_list_length_error_includes_all_per_level_shapes():
+    dtm = [torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 4, 4)]
+    valid_mask = [torch.ones(2, 1, 8, 8)]
+
+    with pytest.raises(ValueError) as error:
+        AuxiliaryTargets(dtm, valid_mask)
+
+    message = str(error.value)
+    assert "dtm_shapes=[(2, 1, 8, 8), (2, 1, 4, 4)]" in message
+    assert "valid_mask_shapes=[(2, 1, 8, 8)]" in message
+
+
+@pytest.mark.parametrize(
+    ("field", "value"),
+    [
+        ("dtm", [torch.zeros(2, 1, 8, 8), "not-a-tensor"]),
+        ("valid_mask", [torch.ones(2, 1, 8, 8), {"bad": "tree"}]),
+        ("dtm", []),
+        ("valid_mask", ()),
+    ],
+)
+def test_auxiliary_targets_validate_each_non_none_tree_at_construction(field, value):
+    kwargs = {"dtm": None, "valid_mask": None}
+    kwargs[field] = value
+
+    with pytest.raises((TypeError, ValueError)):
+        AuxiliaryTargets(**kwargs)
diff --git a/nnunetv2/training/auxiliary_targets.py b/nnunetv2/training/auxiliary_targets.py
index 6951220..959ee4a 100644
--- a/nnunetv2/training/auxiliary_targets.py
+++ b/nnunetv2/training/auxiliary_targets.py
@@ -3,50 +3,65 @@
 from dataclasses import dataclass
 from typing import Optional, TypeAlias
 
 import torch
 from torch import Tensor
 
 
 TensorTree: TypeAlias = Tensor | list["TensorTree"] | tuple["TensorTree", ...]
 
 
-def _shape(value: TensorTree) -> tuple[int, ...]:
+def _shape_summary(value: TensorTree) -> tuple[int, ...] | list[object]:
     if isinstance(value, torch.Tensor):
         return tuple(value.shape)
-    raise TypeError(f"Auxiliary targets must contain tensors, got {type(value).__name__}")
+    return [_shape_summary(item) for item in value]
+
+
+def _validate_tree(value: TensorTree, field_name: str, path: str = "root") -> None:
+    if isinstance(value, torch.Tensor):
+        return
+    if isinstance(value, (list, tuple)):
+        if not value:
+            raise ValueError(f"{field_name} tree at {path} cannot be empty")
+        for index, item in enumerate(value):
+            _validate_tree(item, field_name, f"{path}[{index}]")
+        return
+    raise TypeError(
+        f"{field_name} tree at {path} must contain only tensors and lists/tuples, "
+        f"got {type(value).__name__}"
+    )
 
 
 def _validate_matching_shapes(dtm: TensorTree, valid_mask: TensorTree) -> None:
     if isinstance(dtm, torch.Tensor) and isinstance(valid_mask, torch.Tensor):
         if tuple(dtm.shape) != tuple(valid_mask.shape):
             raise ValueError(
                 "DTM and valid_mask shapes must match: "
                 f"dtm={tuple(dtm.shape)}, valid_mask={tuple(valid_mask.shape)}"
             )
         return
 
     if isinstance(dtm, (list, tuple)) and isinstance(valid_mask, (list, tuple)):
         if len(dtm) != len(valid_mask):
             raise ValueError(
                 "DTM and valid_mask list lengths must match: "
-                f"dtm={len(dtm)}, valid_mask={len(valid_mask)}"
+                f"dtm={len(dtm)}, valid_mask={len(valid_mask)}, "
+                f"dtm_shapes={_shape_summary(dtm)}, "
+                f"valid_mask_shapes={_shape_summary(valid_mask)}"
             )
         for dtm_item, mask_item in zip(dtm, valid_mask):
             _validate_matching_shapes(dtm_item, mask_item)
         return
 
-    dtm_shape = _shape(dtm) if isinstance(dtm, torch.Tensor) else dtm
-    mask_shape = _shape(valid_mask) if isinstance(valid_mask, torch.Tensor) else valid_mask
     raise ValueError(
         "DTM and valid_mask must have matching tensor/list structure: "
-        f"dtm={dtm_shape}, valid_mask={mask_shape}"
+        f"dtm_shapes={_shape_summary(dtm)}, valid_mask_shapes={_shape_summary(valid_mask)}"
     )
 
 
 def _move_to(value: Optional[TensorTree], device: torch.device | str) -> Optional[TensorTree]:
     if value is None:
         return None
     if isinstance(value, torch.Tensor):
         return value.to(device)
     if isinstance(value, list):
         return [_move_to(item, device) for item in value]  # type: ignore[list-item]
@@ -66,20 +81,24 @@ def _first_resolution(value: Optional[TensorTree]) -> Optional[Tensor]:
 
 
 @dataclass(frozen=True)
 class AuxiliaryTargets:
     """Optional DTM and validity-mask tensors, including deep-supervision lists."""
 
     dtm: Optional[TensorTree] = None
     valid_mask: Optional[TensorTree] = None
 
     def __post_init__(self) -> None:
+        if self.dtm is not None:
+            _validate_tree(self.dtm, "dtm")
+        if self.valid_mask is not None:
+            _validate_tree(self.valid_mask, "valid_mask")
         if self.dtm is not None and self.valid_mask is not None:
             _validate_matching_shapes(self.dtm, self.valid_mask)
 
     @classmethod
     def empty(cls) -> "AuxiliaryTargets":
         return cls()
 
     def to(self, device: torch.device | str) -> "AuxiliaryTargets":
         return type(self)(_move_to(self.dtm, device), _move_to(self.valid_mask, device))
 
diff --git a/nnunetv2/training/pipeline_config.py b/nnunetv2/training/pipeline_config.py
index 6592bb2..7a3606f 100644
--- a/nnunetv2/training/pipeline_config.py
+++ b/nnunetv2/training/pipeline_config.py
@@ -1,45 +1,73 @@
 """Configuration contracts for the unified enhancement pipeline."""
 
 from dataclasses import dataclass
+import math
+from numbers import Real
 from typing import Literal
 
 
 DTMMode = Literal["disabled", "monitor", "active"]
 
 
+def _finite_float(name: str, value: object) -> float:
+    if isinstance(value, bool) or not isinstance(value, Real):
+        raise TypeError(f"{name} must be a real number")
+    result = float(value)
+    if not math.isfinite(result):
+        raise ValueError(f"{name} must be finite")
+    return result
+
+
 @dataclass(frozen=True)
 class EnhancementConfig:
     """Immutable, validated configuration shared by pipeline components."""
 
     enable_anisotropic_attention: bool = False
     enable_tubular_enhancement: bool = False
     projection_angles_deg: tuple[float, ...] = ()
     projection_consistency_weight: float = 0.0
     dtm_mode: DTMMode = "disabled"
     dtm_weight: float = 0.0
     dtm_activate_after_epoch: int = 0
     dtm_active_weight: float = 0.5
 
     def __post_init__(self) -> None:
-        angles = tuple(float(angle) for angle in self.projection_angles_deg)
+        if not isinstance(self.enable_anisotropic_attention, bool):
+            raise TypeError("enable_anisotropic_attention must be a bool")
+        if not isinstance(self.enable_tubular_enhancement, bool):
+            raise TypeError("enable_tubular_enhancement must be a bool")
+        if not isinstance(self.dtm_activate_after_epoch, int) or isinstance(
+            self.dtm_activate_after_epoch, bool
+        ):
+            raise TypeError("dtm_activate_after_epoch must be an integer")
+
+        angles = tuple(_finite_float("projection_angles_deg", angle) for angle in self.projection_angles_deg)
         object.__setattr__(self, "projection_angles_deg", angles)
+        projection_weight = _finite_float(
+            "projection_consistency_weight", self.projection_consistency_weight
+        )
+        dtm_weight = _finite_float("dtm_weight", self.dtm_weight)
+        active_weight = _finite_float("dtm_active_weight", self.dtm_active_weight)
+        object.__setattr__(self, "projection_consistency_weight", projection_weight)
+        object.__setattr__(self, "dtm_weight", dtm_weight)
+        object.__setattr__(self, "dtm_active_weight", active_weight)
 
         if self.dtm_mode not in ("disabled", "monitor", "active"):
             raise ValueError(
                 f"dtm_mode must be 'disabled', 'monitor', or 'active', got {self.dtm_mode!r}"
             )
-        if self.projection_consistency_weight < 0:
+        if projection_weight < 0:
             raise ValueError("projection_consistency_weight must be non-negative")
-        if self.dtm_weight < 0:
+        if dtm_weight < 0:
             raise ValueError("dtm_weight must be non-negative")
-        if self.dtm_active_weight < 0:
+        if active_weight < 0:
             raise ValueError("dtm_active_weight must be non-negative")
         if self.dtm_activate_after_epoch < 0:
             raise ValueError("dtm_activate_after_epoch must be non-negative")
         if not angles and self.projection_consistency_weight != 0:
             raise ValueError(
                 "projection_angles_deg cannot be empty when projection_consistency_weight is non-zero"
             )
 
     @classmethod
     def from_legacy_preset(cls, name: str) -> "EnhancementConfig":

