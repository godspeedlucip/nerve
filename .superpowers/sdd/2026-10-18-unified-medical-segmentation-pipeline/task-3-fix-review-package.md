COMMITS
cbbb68b (HEAD -> master) fix: share normalized targets across auxiliary losses
STAT
 .../task-3-report.md                               |  7 +++++
 nnunetv2/tests/unit/test_composite_loss.py         | 34 ++++++++++++++++++++++
 nnunetv2/training/loss/composite_loss.py           | 20 ++++++++++---
 3 files changed, 57 insertions(+), 4 deletions(-)
DIFF
diff --git a/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-3-report.md b/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-3-report.md
index 020df80..2f4bcf4 100644
--- a/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-3-report.md
+++ b/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-3-report.md
@@ -25,10 +25,17 @@ The direct `pytest` executable was unavailable in this environment, so the equiv
 
 Commit: `refactor: compose segmentation auxiliary losses` (final hash is recorded in the task handoff)
 
 Planned message: `refactor: compose segmentation auxiliary losses`
 
 ## Concerns
 
 - The existing `AuxiliaryTargets` contract requires DTM and validity-mask tensors to have exactly matching shapes, while the legacy DTM loader naturally stores class channels in a 6D DTM tensor and validity masks as 5D tensors. This task keeps the committed Task 1 contract unchanged; a later trainer adapter may need to normalize that legacy shape before constructing `AuxiliaryTargets`.
 - The environment does not expose a standalone `pytest` executable; verification used `python -m pytest`.
 - User edits in `.gitignore` and `TRAINING_INFERENCE_GUIDE.md` were left untouched.
+
+## Review fix: shared target normalization
+
+- RED: the regression test `test_dtm_receives_projection_normalized_final_target_while_segmentation_keeps_target_tree` failed because DTM received the raw three-channel regions target while projection received the two-channel ignore-masked target.
+- GREEN: after introducing one `_NormalizedTargets` result for the final output/target pair and reusing it for both projection and DTM, the focused suite passed with `8 passed`.
+- The segmentation loss continues to receive the original deep-supervision target tree unchanged.
+- Reviewer minor issues were intentionally left unchanged in this round.
diff --git a/nnunetv2/tests/unit/test_composite_loss.py b/nnunetv2/tests/unit/test_composite_loss.py
index b427774..9a202ce 100644
--- a/nnunetv2/tests/unit/test_composite_loss.py
+++ b/nnunetv2/tests/unit/test_composite_loss.py
@@ -166,20 +166,54 @@ def test_regions_and_ignore_label_are_normalized_for_projection_mask():
     target[0, 2, 0, 0, 0] = 1
 
     loss(base_output(), target)
 
     pred, target_onehot = proj.calls[0]
     assert target_onehot.shape[1] == 2
     assert target_onehot[0, :, 0, 0, 0].tolist() == [0.0, 0.0]
     assert pred[0, :, 0, 0, 0].tolist() == [0.0, 0.0]
 
 
+def test_dtm_receives_projection_normalized_final_target_while_segmentation_keeps_target_tree():
+    seg = RecordingSegmentationLoss()
+    proj = RecordingProjectionLoss()
+    dtm = RecordingDTMLoss()
+    loss = CompositeSegmentationLoss(
+        seg,
+        EnhancementConfig(
+            projection_angles_deg=(0.0,),
+            projection_consistency_weight=1.0,
+            dtm_mode="active",
+            dtm_weight=1.0,
+            dtm_active_weight=1.0,
+        ),
+        label_manager(regions=True, ignore=99, classes=2),
+        projection_loss=proj,
+        dtm_loss=dtm,
+    )
+    final_target = torch.zeros((1, 3, 2, 2, 2), dtype=torch.float32)
+    final_target[0, 0, 0] = torch.tensor([[1, 0], [0, 1]])
+    final_target[0, 1, 0] = torch.tensor([[0, 1], [1, 0]])
+    final_target[0, 2, 0, 0, 0] = 1
+    target_tree = [final_target, torch.zeros((1, 3, 1, 1, 1))]
+    auxiliary = AuxiliaryTargets(
+        torch.ones((1, 1, 2, 2, 2)), torch.ones((1, 1, 2, 2, 2))
+    )
+
+    loss([base_output(), base_output()], target_tree, auxiliary, epoch=0)
+
+    assert seg.targets[-1] is target_tree
+    assert torch.equal(dtm.calls[0][0], proj.calls[0][1])
+    assert dtm.calls[0][0].shape == (1, 2, 2, 2, 2)
+    assert dtm.calls[0][0][0, :, 0, 0, 0].tolist() == [0.0, 0.0]
+
+
 def test_deep_supervision_segmentation_loss_receives_original_tree_and_combined_compatibility_is_scalar():
     seg = RecordingSegmentationLoss()
     composite = CompositeSegmentationLoss(seg, EnhancementConfig(), label_manager())
     output = [base_output(), base_output()]
     target = [torch.zeros((1, 1, 2, 2, 2), dtype=torch.long), torch.zeros((1, 1, 1, 1, 1), dtype=torch.long)]
 
     result = composite(output, target)
 
     assert seg.targets[-1] is target
     assert result.total.ndim == 0
diff --git a/nnunetv2/training/loss/composite_loss.py b/nnunetv2/training/loss/composite_loss.py
index 91ca48b..7cb2c5b 100644
--- a/nnunetv2/training/loss/composite_loss.py
+++ b/nnunetv2/training/loss/composite_loss.py
@@ -19,20 +19,26 @@ class LossBreakdown:
 
     total: Tensor
     segmentation: Tensor
     projection: Tensor
     dtm: Tensor
 
     def as_scalar(self) -> Tensor:
         return self.total
 
 
+@dataclass(frozen=True)
+class _NormalizedTargets:
+    output: Tensor
+    target: Tensor
+
+
 def _first_resolution(value):
     return value[0] if isinstance(value, (list, tuple)) else value
 
 
 class CompositeSegmentationLoss(nn.Module):
     """Compose the existing segmentation, projection, and DTM loss modules."""
 
     def __init__(
         self,
         segmentation_loss: nn.Module,
@@ -53,41 +59,41 @@ class CompositeSegmentationLoss(nn.Module):
             raise ValueError(
                 "projection_loss is required when projection consistency is enabled"
             )
 
     @staticmethod
     def _as_scalar(value: Tensor | float | int, reference: Tensor) -> Tensor:
         if isinstance(value, Tensor):
             return value if value.ndim == 0 else value.mean()
         return reference.new_tensor(value)
 
-    def _projection_targets(self, output: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
+    def _normalize_final_targets(self, output: Tensor, target: Tensor) -> _NormalizedTargets:
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
-        return output, target_onehot
+        return _NormalizedTargets(output=output, target=target_onehot)
 
     def _dtm_component(
         self, output: Tensor, target: Tensor, auxiliary: Optional[AuxiliaryTargets], epoch: Optional[int]
     ) -> Tensor:
         zero = output.sum() * 0
         if self.config.dtm_mode == "disabled":
             return zero
         if auxiliary is None or self.dtm_loss is None:
             if not self._missing_auxiliary_warned:
                 warnings.warn(
@@ -108,20 +114,26 @@ class CompositeSegmentationLoss(nn.Module):
         output,
         target,
         auxiliary: Optional[AuxiliaryTargets] = None,
         epoch: Optional[int] = None,
     ) -> LossBreakdown:
         segmentation = self._as_scalar(self.segmentation_loss(output, target), output[0] if isinstance(output, (list, tuple)) else output)
         output_final = _first_resolution(output)
         target_final = _first_resolution(target)
 
         projection = output_final.sum() * 0
+        normalized = None
+        if self.config.projection_consistency_weight or self.config.dtm_mode != "disabled":
+            normalized = self._normalize_final_targets(output_final, target_final)
+
         if self.config.projection_consistency_weight and self.projection_loss is not None:
-            projection_input, target_onehot = self._projection_targets(output_final, target_final)
+            projection_input, target_onehot = normalized.output, normalized.target
             projection = self._as_scalar(self.projection_loss(projection_input, target_onehot), output_final)
 
-        dtm = self._dtm_component(output_final, target_final, auxiliary, epoch)
+        normalized_output = normalized.output if normalized is not None else output_final
+        normalized_target = normalized.target if normalized is not None else target_final
+        dtm = self._dtm_component(normalized_output, normalized_target, auxiliary, epoch)
         total = segmentation + self.config.projection_consistency_weight * projection
         if self.config.dtm_mode == "active" and auxiliary is not None and self.dtm_loss is not None:
             if epoch is not None and epoch >= self.config.dtm_activate_after_epoch:
                 total = total + self.config.dtm_weight * self.config.dtm_active_weight * dtm
         return LossBreakdown(total=total, segmentation=segmentation, projection=projection, dtm=dtm)

