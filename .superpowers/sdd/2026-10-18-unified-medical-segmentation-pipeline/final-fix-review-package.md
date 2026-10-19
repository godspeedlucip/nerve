COMMITS
42e60f3 (HEAD -> master) docs: record composite loss review fixes
191233a fix: harden composite loss review findings
STAT
 .../task-5-report.md                               |  26 +++++
 nnunetv2/tests/unit/test_composite_loss.py         | 110 +++++++++++++++++++++
 nnunetv2/training/loss/composite_loss.py           |  50 +++++++++-
 3 files changed, 185 insertions(+), 1 deletion(-)
DIFF
diff --git a/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-5-report.md b/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-5-report.md
index 65c9a81..24da329 100644
--- a/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-5-report.md
+++ b/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-5-report.md
@@ -33,10 +33,36 @@ The pytest warnings are dependency deprecations from `pytest_asyncio`, SciPy nam
 The pre-existing `.gitignore` edit and other unrelated workspace artifacts were not changed or staged.
 
 ## Environment-only limitations
 
 - The bare `pytest` executable is unavailable in this shell, so equivalent commands were run through `python -m pytest`.
 - No dependency prevented the requested final verification: the full unit suite and `verify_modules.py` both completed successfully on CPU.
 
 ## Commit
 
 Implementation commit: `706c0cf` — `test: verify unified enhancement pipeline`.
+
+## Final review fix wave
+
+Scope was limited to the two Important findings from `final-review-package.md`.
+
+### Important 1: missing or incomplete DTM auxiliary targets
+
+- RED: `test_empty_auxiliary_targets_warn_once_and_keep_dtm_zero` and `test_partially_populated_auxiliary_targets_warn_once_and_keep_dtm_zero` failed because the existing implementation emitted no warning for `AuxiliaryTargets.empty()` or partially populated targets.
+- GREEN: `CompositeSegmentationLoss` now treats `None`, empty, and partially populated auxiliary targets uniformly, emits the existing RuntimeWarning text with an explicit incomplete-target qualifier exactly once per instance, and returns a zero DTM component without calling the DTM loss.
+
+### Important 2: component-specific shape validation
+
+- RED: projection and DTM batch/spatial mismatch regressions failed because no validation ran before component calls.
+- GREEN: final output/target batch and spatial dimensions are validated for each enabled component; DTM 5D and legacy 6D auxiliary layouts validate batch/spatial dimensions before the DTM loss call. Errors identify the component and include actual output, target, DTM, and validity-mask shapes where applicable.
+
+### Fix-wave verification
+
+- `python -m pytest nnunetv2/tests/unit/test_composite_loss.py -q` → `15 passed, 1 warning`.
+- `python -m pytest nnunetv2/tests/unit -q` → `72 passed, 8 warnings`.
+- `python -m py_compile nnunetv2/training/loss/composite_loss.py nnunetv2/tests/unit/test_composite_loss.py` → exit 0.
+- Public loss import check → `loss imports ok`.
+- `git diff --check` → no whitespace errors; only existing LF/CRLF conversion warnings.
+
+Deferred review minors were not changed.
+
+Fix commit: `191233a` — `fix: harden composite loss review findings`.
diff --git a/nnunetv2/tests/unit/test_composite_loss.py b/nnunetv2/tests/unit/test_composite_loss.py
index 9a202ce..203ef6e 100644
--- a/nnunetv2/tests/unit/test_composite_loss.py
+++ b/nnunetv2/tests/unit/test_composite_loss.py
@@ -146,20 +146,130 @@ def test_missing_auxiliary_data_keeps_dtm_zero_without_calling_loss():
         label_manager(), dtm_loss=dtm,
     )
 
     with pytest.warns(RuntimeWarning, match="AuxiliaryTargets are missing"):
         breakdown = loss(base_output(), torch.zeros((1, 1, 2, 2, 2), dtype=torch.long), epoch=10)
 
     assert breakdown.dtm.item() == 0.0
     assert len(dtm.calls) == 0
 
 
+def test_empty_auxiliary_targets_warn_once_and_keep_dtm_zero():
+    dtm = RecordingDTMLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(), EnhancementConfig(dtm_mode="active"),
+        label_manager(), dtm_loss=dtm,
+    )
+    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)
+
+    with pytest.warns(RuntimeWarning, match="AuxiliaryTargets are missing") as captured:
+        first = loss(base_output(), target, AuxiliaryTargets.empty(), epoch=10)
+        second = loss(base_output(), target, AuxiliaryTargets.empty(), epoch=10)
+
+    assert len(captured) == 1
+    assert first.dtm.item() == 0.0
+    assert second.dtm.item() == 0.0
+    assert len(dtm.calls) == 0
+
+
+def test_partially_populated_auxiliary_targets_warn_once_and_keep_dtm_zero():
+    dtm = RecordingDTMLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(), EnhancementConfig(dtm_mode="active"),
+        label_manager(), dtm_loss=dtm,
+    )
+    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)
+    partial = AuxiliaryTargets(dtm=torch.ones((1, 1, 2, 2, 2)))
+
+    with pytest.warns(RuntimeWarning, match="AuxiliaryTargets are missing") as captured:
+        first = loss(base_output(), target, partial, epoch=10)
+        second = loss(base_output(), target, partial, epoch=10)
+
+    assert len(captured) == 1
+    assert first.dtm.item() == 0.0
+    assert second.dtm.item() == 0.0
+    assert len(dtm.calls) == 0
+
+
+@pytest.mark.parametrize(
+    "target_shape",
+    [(2, 1, 2, 2, 2), (1, 1, 3, 2, 2)],
+)
+def test_projection_rejects_final_batch_or_spatial_shape_mismatch(target_shape):
+    projection = RecordingProjectionLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(),
+        EnhancementConfig(projection_angles_deg=(0.0,), projection_consistency_weight=1.0),
+        label_manager(), projection_loss=projection,
+    )
+    output = base_output()
+    target = torch.zeros(target_shape, dtype=torch.long)
+
+    with pytest.raises(
+        ValueError,
+        match=r"Projection loss shape mismatch: output=\(1, 2, 2, 2, 2\), target=.*",
+    ) as error:
+        loss(output, target)
+
+    assert str(target_shape) in str(error.value)
+    assert len(projection.calls) == 0
+
+
+def test_dtm_rejects_final_output_target_shape_mismatch():
+    dtm = RecordingDTMLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(), EnhancementConfig(dtm_mode="active"),
+        label_manager(), dtm_loss=dtm,
+    )
+    output = base_output()
+    target = torch.zeros((2, 1, 2, 2, 2), dtype=torch.long)
+    auxiliary = AuxiliaryTargets(
+        dtm=torch.ones((2, 1, 2, 2, 2)),
+        valid_mask=torch.ones((2, 1, 2, 2, 2)),
+    )
+
+    with pytest.raises(
+        ValueError,
+        match=r"DTM loss shape mismatch: output=.*target=.*",
+    ) as error:
+        loss(output, target, auxiliary, epoch=10)
+
+    assert "output=(1, 2, 2, 2, 2)" in str(error.value)
+    assert "target=(2, 1, 2, 2, 2)" in str(error.value)
+    assert len(dtm.calls) == 0
+
+
+@pytest.mark.parametrize(
+    "dtm_shape",
+    [(2, 1, 2, 2, 2, 2), (1, 1, 3, 2, 2, 2)],
+)
+def test_dtm_rejects_batch_or_spatial_shape_mismatch(dtm_shape):
+    dtm = RecordingDTMLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(), EnhancementConfig(dtm_mode="active"),
+        label_manager(), dtm_loss=dtm,
+    )
+    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)
+    auxiliary = AuxiliaryTargets(
+        dtm=torch.ones(dtm_shape), valid_mask=torch.ones(dtm_shape)
+    )
+
+    with pytest.raises(
+        ValueError,
+        match=r"DTM loss shape mismatch: output=.*dtm=.*",
+    ) as error:
+        loss(base_output(), target, auxiliary, epoch=10)
+
+    assert str(dtm_shape) in str(error.value)
+    assert len(dtm.calls) == 0
+
+
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
diff --git a/nnunetv2/training/loss/composite_loss.py b/nnunetv2/training/loss/composite_loss.py
index 7cb2c5b..927d897 100644
--- a/nnunetv2/training/loss/composite_loss.py
+++ b/nnunetv2/training/loss/composite_loss.py
@@ -81,52 +81,100 @@ class CompositeSegmentationLoss(nn.Module):
                 valid = (target_indices != self.label_manager.ignore_label).unsqueeze(1).float()
                 target_indices = target_indices.masked_fill(valid[:, 0] == 0, 0)
             target_onehot = F.one_hot(target_indices, num_classes=num_classes)
             target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()
 
         if valid is not None:
             output = output * valid
             target_onehot = target_onehot * valid
         return _NormalizedTargets(output=output, target=target_onehot)
 
+    @staticmethod
+    def _validate_final_shapes(component: str, output: Tensor, target: Tensor) -> None:
+        if output.ndim != 5 or target.ndim != 5:
+            raise ValueError(
+                f"{component} loss shape mismatch: output={tuple(output.shape)}, "
+                f"target={tuple(target.shape)}; expected 5D tensors"
+            )
+        if output.shape[0] != target.shape[0] or output.shape[2:] != target.shape[2:]:
+            raise ValueError(
+                f"{component} loss shape mismatch: output={tuple(output.shape)}, "
+                f"target={tuple(target.shape)}"
+            )
+
+    @staticmethod
+    def _validate_dtm_shapes(output: Tensor, dtm: Tensor, valid_mask: Tensor) -> None:
+        if dtm.ndim == 5:
+            dtm_batch = dtm.shape[0]
+            dtm_spatial = dtm.shape[2:]
+        elif dtm.ndim == 6:
+            dtm_batch = dtm.shape[0]
+            dtm_spatial = dtm.shape[2:5]
+        else:
+            raise ValueError(
+                f"DTM loss shape mismatch: output={tuple(output.shape)}, "
+                f"dtm={tuple(dtm.shape)}, valid_mask={tuple(valid_mask.shape)}; "
+                "expected DTM to be 5D or 6D"
+            )
+        if output.shape[0] != dtm_batch or output.shape[2:] != dtm_spatial:
+            raise ValueError(
+                f"DTM loss shape mismatch: output={tuple(output.shape)}, "
+                f"dtm={tuple(dtm.shape)}, valid_mask={tuple(valid_mask.shape)}"
+            )
+
     def _dtm_component(
         self, output: Tensor, target: Tensor, auxiliary: Optional[AuxiliaryTargets], epoch: Optional[int]
     ) -> Tensor:
         zero = output.sum() * 0
         if self.config.dtm_mode == "disabled":
             return zero
         if auxiliary is None or self.dtm_loss is None:
             if not self._missing_auxiliary_warned:
                 warnings.warn(
-                    "DTM loss is enabled but AuxiliaryTargets are missing; DTM component is zero.",
+                    "DTM loss is enabled but AuxiliaryTargets are missing or incomplete; "
+                    "DTM component is zero.",
                     RuntimeWarning,
                     stacklevel=2,
                 )
                 self._missing_auxiliary_warned = True
             return zero
 
         dtm, valid_mask = auxiliary.final_resolution()
         if dtm is None or valid_mask is None:
+            if not self._missing_auxiliary_warned:
+                warnings.warn(
+                    "DTM loss is enabled but AuxiliaryTargets are missing or incomplete; "
+                    "DTM component is zero.",
+                    RuntimeWarning,
+                    stacklevel=2,
+                )
+                self._missing_auxiliary_warned = True
             return zero
+        self._validate_dtm_shapes(output, dtm, valid_mask)
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
 
+        if self.config.projection_consistency_weight:
+            self._validate_final_shapes("Projection", output_final, target_final)
+        if self.config.dtm_mode != "disabled":
+            self._validate_final_shapes("DTM", output_final, target_final)
+
         projection = output_final.sum() * 0
         normalized = None
         if self.config.projection_consistency_weight or self.config.dtm_mode != "disabled":
             normalized = self._normalize_final_targets(output_final, target_final)
 
         if self.config.projection_consistency_weight and self.projection_loss is not None:
             projection_input, target_onehot = normalized.output, normalized.target
             projection = self._as_scalar(self.projection_loss(projection_input, target_onehot), output_final)
 
         normalized_output = normalized.output if normalized is not None else output_final

