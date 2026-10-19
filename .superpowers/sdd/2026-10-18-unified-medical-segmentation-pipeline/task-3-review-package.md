COMMITS
4207073 (HEAD -> master) refactor: compose segmentation auxiliary losses
STAT
 .../task-3-report.md                               |  34 ++++
 nnunetv2/network_modules/projection_consistency.py |  44 ++---
 nnunetv2/tests/unit/test_composite_loss.py         | 188 +++++++++++++++++++++
 nnunetv2/training/loss/__init__.py                 |   3 +
 nnunetv2/training/loss/composite_loss.py           | 127 ++++++++++++++
 nnunetv2/training/loss/dtm_losses.py               |  18 +-
 6 files changed, 377 insertions(+), 37 deletions(-)
DIFF
diff --git a/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-3-report.md b/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-3-report.md
new file mode 100644
index 0000000..020df80
--- /dev/null
+++ b/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-3-report.md
@@ -0,0 +1,34 @@
+# Task 3 report: composite segmentation, projection, and DTM loss
+
+## Result
+
+Implemented the Task 3 composite loss pipeline and preserved the scalar compatibility path for `CombinedSegProjLoss.forward(output, target)`.
+
+## TDD evidence
+
+- RED: `python -m pytest nnunetv2/tests/unit/test_composite_loss.py -q` failed during collection with `ModuleNotFoundError: nnunetv2.training.loss.composite_loss`, before the production module existed.
+- GREEN: the focused suite passed with `7 passed`.
+- Full unit suite: `python -m pytest nnunetv2/tests/unit -q` passed with `45 passed`.
+- `git diff --check`: passed.
+
+The direct `pytest` executable was unavailable in this environment, so the equivalent module invocation was used. The full suite reports one unrelated pytest-asyncio deprecation warning under Python 3.14.
+
+## Changed files
+
+- `nnunetv2/training/loss/composite_loss.py`: added `LossBreakdown` and `CompositeSegmentationLoss`; centralized final-resolution selection, target normalization, ignore masking, projection calculation, DTM monitor/active scheduling, and missing-auxiliary handling.
+- `nnunetv2/network_modules/projection_consistency.py`: adapted `CombinedSegProjLoss` to the composite implementation while retaining its scalar forward signature and import path.
+- `nnunetv2/training/loss/dtm_losses.py`: preserved the existing foreground DTM mean and added support for 5D foreground maps alongside the existing 6D stored-DTM layout.
+- `nnunetv2/training/loss/__init__.py`: exported the composite loss and breakdown.
+- `nnunetv2/tests/unit/test_composite_loss.py`: added focused coverage for all Task 3 behaviors.
+
+## Commit
+
+Commit: `refactor: compose segmentation auxiliary losses` (final hash is recorded in the task handoff)
+
+Planned message: `refactor: compose segmentation auxiliary losses`
+
+## Concerns
+
+- The existing `AuxiliaryTargets` contract requires DTM and validity-mask tensors to have exactly matching shapes, while the legacy DTM loader naturally stores class channels in a 6D DTM tensor and validity masks as 5D tensors. This task keeps the committed Task 1 contract unchanged; a later trainer adapter may need to normalize that legacy shape before constructing `AuxiliaryTargets`.
+- The environment does not expose a standalone `pytest` executable; verification used `python -m pytest`.
+- User edits in `.gitignore` and `TRAINING_INFERENCE_GUIDE.md` were left untouched.
diff --git a/nnunetv2/network_modules/projection_consistency.py b/nnunetv2/network_modules/projection_consistency.py
index af4dbc3..52c4e05 100644
--- a/nnunetv2/network_modules/projection_consistency.py
+++ b/nnunetv2/network_modules/projection_consistency.py
@@ -166,56 +166,36 @@ class CombinedSegProjLoss(nn.Module):
     Args:
         seg_loss: DC+CE or DC+BCE loss, optionally wrapped in DeepSupervisionWrapper.
         proj_loss: ProjectionConsistencyLoss instance.
         proj_weight: Weight for the projection consistency term.
         label_manager: nnUNet LabelManager (for target → one-hot conversion).
     """
 
     def __init__(self, seg_loss: nn.Module, proj_loss: ProjectionConsistencyLoss,
                  proj_weight: float, label_manager):
         super().__init__()
+        from nnunetv2.training.loss.composite_loss import CompositeSegmentationLoss
+        from nnunetv2.training.pipeline_config import EnhancementConfig
+
         self.seg_loss = seg_loss
         self.proj_loss = proj_loss
         self.proj_weight = proj_weight
         self.label_manager = label_manager
-
-    def _target_to_onehot(self, target: torch.Tensor,
-                          num_classes: int) -> torch.Tensor:
-        """(B, 1, D, H, W) class indices → (B, C, D, H, W) one-hot."""
-        t = target.squeeze(1).long()
-        oh = F.one_hot(t, num_classes=num_classes)
-        return oh.permute(0, 4, 1, 2, 3).float()
+        angles = tuple(getattr(proj_loss, "angles_deg", ()))
+        config = EnhancementConfig(
+            projection_angles_deg=angles,
+            projection_consistency_weight=proj_weight,
+        )
+        self._composite = CompositeSegmentationLoss(
+            seg_loss, config, label_manager, projection_loss=proj_loss
+        )
 
     def forward(self, output, target) -> torch.Tensor:
         """
         Args:
             output: Tensor (B,C,D,H,W) or list of Tensors (deep supervision).
             target: Tensor or list of Tensors.
 
         Returns:
             Scalar combined loss.
         """
-        l_seg = self.seg_loss(output, target)
-
-        output_final = output[0] if isinstance(output, list) else output
-        target_final = target[0] if isinstance(target, list) else target
-
-        if self.label_manager.has_regions:
-            target_oh = target_final[:, :-1].float()
-            num_classes = target_oh.shape[1]
-            pred_vol = output_final
-            if self.label_manager.has_ignore_label:
-                mask = 1 - target_final[:, -1:]
-                pred_vol = pred_vol * mask
-        else:
-            num_classes = self.label_manager.num_segmentation_heads
-            target_oh = self._target_to_onehot(target_final, num_classes)
-            pred_vol = output_final
-            if self.label_manager.has_ignore_label:
-                mask = (target_final != self.label_manager.ignore_label).float()
-                pred_vol = pred_vol * mask
-                target_oh = target_oh * mask
-
-        # proj_loss applies softmax internally, so pass raw logits
-        l_proj = self.proj_loss(pred_vol, target_oh)
-
-        return l_seg + self.proj_weight * l_proj
+        return self._composite(output, target).total
diff --git a/nnunetv2/tests/unit/test_composite_loss.py b/nnunetv2/tests/unit/test_composite_loss.py
new file mode 100644
index 0000000..b427774
--- /dev/null
+++ b/nnunetv2/tests/unit/test_composite_loss.py
@@ -0,0 +1,188 @@
+from types import SimpleNamespace
+
+import pytest
+import torch
+from torch import nn
+
+from nnunetv2.network_modules.projection_consistency import (
+    CombinedSegProjLoss,
+)
+from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
+from nnunetv2.training.loss.composite_loss import (
+    CompositeSegmentationLoss,
+    LossBreakdown,
+)
+from nnunetv2.training.pipeline_config import EnhancementConfig
+
+
+class RecordingSegmentationLoss(nn.Module):
+    def __init__(self, value=2.0):
+        super().__init__()
+        self.value = value
+        self.targets = []
+
+    def forward(self, output, target):
+        self.targets.append(target)
+        tensor = output[0] if isinstance(output, list) else output
+        return tensor.mean() * 0 + self.value
+
+
+class RecordingProjectionLoss(nn.Module):
+    def __init__(self, value=3.0):
+        super().__init__()
+        self.value = value
+        self.calls = []
+
+    def forward(self, output, target_onehot):
+        self.calls.append((output, target_onehot))
+        return output.mean() * 0 + self.value
+
+
+class RecordingDTMLoss(nn.Module):
+    def __init__(self, value=5.0):
+        super().__init__()
+        self.value = value
+        self.calls = []
+
+    def forward(self, target, output, dtm, valid_mask):
+        self.calls.append((target, output, dtm, valid_mask))
+        return output.mean() * 0 + self.value
+
+
+def label_manager(*, regions=False, ignore=None, classes=2):
+    return SimpleNamespace(
+        has_regions=regions,
+        has_ignore_label=ignore is not None,
+        ignore_label=ignore,
+        num_segmentation_heads=classes,
+    )
+
+
+def base_output():
+    return torch.zeros((1, 2, 2, 2, 2), requires_grad=True)
+
+
+def test_segmentation_only_returns_named_breakdown_and_scalar_gradients():
+    seg = RecordingSegmentationLoss()
+    loss = CompositeSegmentationLoss(
+        seg, EnhancementConfig(), label_manager()
+    )
+
+    breakdown = loss(base_output(), torch.zeros((1, 1, 2, 2, 2), dtype=torch.long))
+
+    assert isinstance(breakdown, LossBreakdown)
+    assert breakdown.total.item() == pytest.approx(2.0)
+    assert breakdown.segmentation.item() == pytest.approx(2.0)
+    assert breakdown.projection.item() == pytest.approx(0.0)
+    assert breakdown.dtm.item() == pytest.approx(0.0)
+    breakdown.total.backward()
+    assert torch.isfinite(breakdown.total)
+
+
+def test_projection_is_weighted_and_uses_final_resolution_one_hot_target():
+    proj = RecordingProjectionLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(),
+        EnhancementConfig(projection_angles_deg=(0.0,), projection_consistency_weight=2.0),
+        label_manager(classes=2),
+        projection_loss=proj,
+    )
+    output = [base_output(), torch.zeros((1, 2, 1, 1, 1), requires_grad=True)]
+    target = [torch.tensor([[[[[1, 0], [0, 1]], [[0, 1], [1, 0]]]]]),
+              torch.zeros((1, 1, 1, 1, 1), dtype=torch.long)]
+
+    breakdown = loss(output, target)
+
+    assert breakdown.total.item() == pytest.approx(8.0)
+    assert len(proj.calls) == 1
+    assert proj.calls[0][0] is output[0]
+    assert proj.calls[0][1].shape == (1, 2, 2, 2, 2)
+    assert torch.equal(proj.calls[0][1][:, 1], target[0][:, 0].float())
+
+
+def test_dtm_monitor_is_reported_but_not_added_to_total():
+    dtm = RecordingDTMLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(),
+        EnhancementConfig(dtm_mode="monitor", dtm_weight=9.0),
+        label_manager(),
+        dtm_loss=dtm,
+    )
+    auxiliary = AuxiliaryTargets(
+        dtm=torch.ones((1, 1, 2, 2, 2)),
+        valid_mask=torch.ones((1, 1, 2, 2, 2)),
+    )
+
+    breakdown = loss(base_output(), torch.zeros((1, 1, 2, 2, 2), dtype=torch.long), auxiliary, epoch=0)
+
+    assert breakdown.dtm.item() == pytest.approx(5.0)
+    assert breakdown.total.item() == pytest.approx(2.0)
+    assert len(dtm.calls) == 1
+
+
+def test_dtm_active_weight_starts_after_configured_epoch():
+    dtm = RecordingDTMLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(),
+        EnhancementConfig(dtm_mode="active", dtm_weight=2.0, dtm_activate_after_epoch=3,
+                          dtm_active_weight=0.5),
+        label_manager(),
+        dtm_loss=dtm,
+    )
+    auxiliary = AuxiliaryTargets(torch.ones((1, 1, 2, 2, 2)), torch.ones((1, 1, 2, 2, 2)))
+    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)
+
+    before = loss(base_output(), target, auxiliary, epoch=2)
+    after = loss(base_output(), target, auxiliary, epoch=3)
+
+    assert before.total.item() == pytest.approx(2.0)
+    assert after.total.item() == pytest.approx(2.0 + 2.0 * 0.5 * 5.0)
+
+
+def test_missing_auxiliary_data_keeps_dtm_zero_without_calling_loss():
+    dtm = RecordingDTMLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(), EnhancementConfig(dtm_mode="active", dtm_weight=1.0),
+        label_manager(), dtm_loss=dtm,
+    )
+
+    with pytest.warns(RuntimeWarning, match="AuxiliaryTargets are missing"):
+        breakdown = loss(base_output(), torch.zeros((1, 1, 2, 2, 2), dtype=torch.long), epoch=10)
+
+    assert breakdown.dtm.item() == 0.0
+    assert len(dtm.calls) == 0
+
+
+def test_regions_and_ignore_label_are_normalized_for_projection_mask():
+    proj = RecordingProjectionLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(),
+        EnhancementConfig(projection_angles_deg=(0.0,), projection_consistency_weight=1.0),
+        label_manager(regions=True, ignore=99, classes=2), projection_loss=proj,
+    )
+    target = torch.zeros((1, 3, 2, 2, 2), dtype=torch.float32)
+    target[0, 0, 0] = torch.tensor([[1, 0], [0, 1]])
+    target[0, 1, 0] = torch.tensor([[0, 1], [1, 0]])
+    target[0, 2, 0, 0, 0] = 1
+
+    loss(base_output(), target)
+
+    pred, target_onehot = proj.calls[0]
+    assert target_onehot.shape[1] == 2
+    assert target_onehot[0, :, 0, 0, 0].tolist() == [0.0, 0.0]
+    assert pred[0, :, 0, 0, 0].tolist() == [0.0, 0.0]
+
+
+def test_deep_supervision_segmentation_loss_receives_original_tree_and_combined_compatibility_is_scalar():
+    seg = RecordingSegmentationLoss()
+    composite = CompositeSegmentationLoss(seg, EnhancementConfig(), label_manager())
+    output = [base_output(), base_output()]
+    target = [torch.zeros((1, 1, 2, 2, 2), dtype=torch.long), torch.zeros((1, 1, 1, 1, 1), dtype=torch.long)]
+
+    result = composite(output, target)
+
+    assert seg.targets[-1] is target
+    assert result.total.ndim == 0
+    combined = CombinedSegProjLoss(seg, nn.Identity(), 0.0, label_manager())
+    scalar = combined(base_output(), torch.zeros((1, 1, 2, 2, 2), dtype=torch.long))
+    assert isinstance(scalar, torch.Tensor) and scalar.ndim == 0
diff --git a/nnunetv2/training/loss/__init__.py b/nnunetv2/training/loss/__init__.py
index e69de29..1445ed6 100644
--- a/nnunetv2/training/loss/__init__.py
+++ b/nnunetv2/training/loss/__init__.py
@@ -0,0 +1,3 @@
+from .composite_loss import CompositeSegmentationLoss, LossBreakdown
+
+__all__ = ["CompositeSegmentationLoss", "LossBreakdown"]
diff --git a/nnunetv2/training/loss/composite_loss.py b/nnunetv2/training/loss/composite_loss.py
new file mode 100644
index 0000000..91ca48b
--- /dev/null
+++ b/nnunetv2/training/loss/composite_loss.py
@@ -0,0 +1,127 @@
+"""Composable segmentation, projection-consistency, and DTM losses."""
+
+from dataclasses import dataclass
+import warnings
+from typing import Optional
+
+import torch
+from torch import Tensor, nn
+from torch.nn import functional as F
+
+from nnunetv2.network_modules.projection_consistency import ProjectionConsistencyLoss
+from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
+from nnunetv2.training.pipeline_config import EnhancementConfig
+
+
+@dataclass(frozen=True)
+class LossBreakdown:
+    """Scalar component losses produced by :class:`CompositeSegmentationLoss`."""
+
+    total: Tensor
+    segmentation: Tensor
+    projection: Tensor
+    dtm: Tensor
+
+    def as_scalar(self) -> Tensor:
+        return self.total
+
+
+def _first_resolution(value):
+    return value[0] if isinstance(value, (list, tuple)) else value
+
+
+class CompositeSegmentationLoss(nn.Module):
+    """Compose the existing segmentation, projection, and DTM loss modules."""
+
+    def __init__(
+        self,
+        segmentation_loss: nn.Module,
+        config: EnhancementConfig,
+        label_manager,
+        projection_loss: Optional[nn.Module] = None,
+        dtm_loss: Optional[nn.Module] = None,
+    ):
+        super().__init__()
+        self.segmentation_loss = segmentation_loss
+        self.config = config
+        self.label_manager = label_manager
+        self.projection_loss = projection_loss
+        self.dtm_loss = dtm_loss
+        self._missing_auxiliary_warned = False
+
+        if config.projection_consistency_weight and projection_loss is None:
+            raise ValueError(
+                "projection_loss is required when projection consistency is enabled"
+            )
+
+    @staticmethod
+    def _as_scalar(value: Tensor | float | int, reference: Tensor) -> Tensor:
+        if isinstance(value, Tensor):
+            return value if value.ndim == 0 else value.mean()
+        return reference.new_tensor(value)
+
+    def _projection_targets(self, output: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
+        if self.label_manager.has_regions:
+            num_classes = self.label_manager.num_segmentation_heads
+            target_onehot = target[:, :num_classes].float()
+            valid = None
+            if self.label_manager.has_ignore_label:
+                valid = 1.0 - target[:, -1:].float()
+        else:
+            num_classes = self.label_manager.num_segmentation_heads
+            valid = None
+            target_indices = target[:, 0].long()
+            if self.label_manager.has_ignore_label:
+                valid = (target_indices != self.label_manager.ignore_label).unsqueeze(1).float()
+                target_indices = target_indices.masked_fill(valid[:, 0] == 0, 0)
+            target_onehot = F.one_hot(target_indices, num_classes=num_classes)
+            target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()
+
+        if valid is not None:
+            output = output * valid
+            target_onehot = target_onehot * valid
+        return output, target_onehot
+
+    def _dtm_component(
+        self, output: Tensor, target: Tensor, auxiliary: Optional[AuxiliaryTargets], epoch: Optional[int]
+    ) -> Tensor:
+        zero = output.sum() * 0
+        if self.config.dtm_mode == "disabled":
+            return zero
+        if auxiliary is None or self.dtm_loss is None:
+            if not self._missing_auxiliary_warned:
+                warnings.warn(
+                    "DTM loss is enabled but AuxiliaryTargets are missing; DTM component is zero.",
+                    RuntimeWarning,
+                    stacklevel=2,
+                )
+                self._missing_auxiliary_warned = True
+            return zero
+
+        dtm, valid_mask = auxiliary.final_resolution()
+        if dtm is None or valid_mask is None:
+            return zero
+        return self._as_scalar(self.dtm_loss(target, output, dtm, valid_mask), output)
+
+    def forward(
+        self,
+        output,
+        target,
+        auxiliary: Optional[AuxiliaryTargets] = None,
+        epoch: Optional[int] = None,
+    ) -> LossBreakdown:
+        segmentation = self._as_scalar(self.segmentation_loss(output, target), output[0] if isinstance(output, (list, tuple)) else output)
+        output_final = _first_resolution(output)
+        target_final = _first_resolution(target)
+
+        projection = output_final.sum() * 0
+        if self.config.projection_consistency_weight and self.projection_loss is not None:
+            projection_input, target_onehot = self._projection_targets(output_final, target_final)
+            projection = self._as_scalar(self.projection_loss(projection_input, target_onehot), output_final)
+
+        dtm = self._dtm_component(output_final, target_final, auxiliary, epoch)
+        total = segmentation + self.config.projection_consistency_weight * projection
+        if self.config.dtm_mode == "active" and auxiliary is not None and self.dtm_loss is not None:
+            if epoch is not None and epoch >= self.config.dtm_activate_after_epoch:
+                total = total + self.config.dtm_weight * self.config.dtm_active_weight * dtm
+        return LossBreakdown(total=total, segmentation=segmentation, projection=projection, dtm=dtm)
diff --git a/nnunetv2/training/loss/dtm_losses.py b/nnunetv2/training/loss/dtm_losses.py
index 6df0844..a80ef5a 100644
--- a/nnunetv2/training/loss/dtm_losses.py
+++ b/nnunetv2/training/loss/dtm_losses.py
@@ -18,27 +18,35 @@ class GenSurfLoss(nn.Module):
         self.smooth = 1e-6
         self.axes = (2, 3, 4)
 
     def forward(self, y_true, y_pred, dtm, valid_mask):
         """
         y_true: (B, 1, D, H, W)
         y_pred: (B, 2, D, H, W)
         valid_mask: (B, 1, D, H, W)
         dtm: (B, channel, D, H, W, class)
         """
-        # Reshape dtm from (B, Channel, D, H, W, Class) -> (B, Class, D, H, W)
-        dtm = torch.transpose(dtm, dim0=5, dim1=1)
-        dtm = torch.squeeze(dtm, dim=5)
+        # Reshape the stored DTM from (B, Channel, D, H, W, Class) to
+        # (B, Class, D, H, W). AuxiliaryTargets may also provide the already
+        # selected foreground map as (B, 1, D, H, W).
+        if dtm.ndim == 6:
+            dtm = torch.transpose(dtm, dim0=5, dim1=1)
+            dtm = torch.squeeze(dtm, dim=5)
+        elif dtm.ndim != 5:
+            raise ValueError(f"DTM must have 5 or 6 dimensions, got {tuple(dtm.shape)}")
 
         y_pred = softmax(y_pred, dim=1)
-        # Return mean of dtm * pred for the foreground class
-        num = torch.mean(dtm[:, 1, ...] * y_pred[:, 1, ...])
+        # Return mean of dtm * pred for the foreground class. Keep the
+        # historical unmasked mean; valid_mask remains part of the signature
+        # for compatibility with existing DTM trainers.
+        dtm_foreground = dtm[:, 1, ...] if dtm.shape[1] > 1 else dtm[:, 0, ...]
+        num = torch.mean(dtm_foreground * y_pred[:, 1, ...])
         return torch.mean(num)
 
 
 class LinearSchedule:
     def __init__(self, num_epochs, init_pause=0):
         if num_epochs <= init_pause:
             raise ValueError("The number of epochs must be greater than the initial pause.")
         self.num_epochs = num_epochs - 1
         self.init_pause = init_pause
 

