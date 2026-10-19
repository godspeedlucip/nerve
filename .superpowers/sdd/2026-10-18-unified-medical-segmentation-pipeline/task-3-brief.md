### Task 3: Composite segmentation, projection, and DTM loss

**Files:**
- Create: `nnunetv2/training/loss/composite_loss.py`
- Modify: `nnunetv2/network_modules/projection_consistency.py`
- Modify: `nnunetv2/training/loss/dtm_losses.py`
- Modify: `nnunetv2/training/loss/__init__.py`
- Test: `nnunetv2/tests/unit/test_composite_loss.py`

**Interfaces:**
- Consumes `EnhancementConfig` and `AuxiliaryTargets` from Task 1, existing segmentation loss, `ProjectionConsistencyLoss`, and `GenSurfLoss`.
- Produces `LossBreakdown(total, segmentation, projection, dtm)` and `CompositeSegmentationLoss.forward(output, target, auxiliary=None, epoch=None)`.
- Keeps `CombinedSegProjLoss.forward(output, target)` behavior by adapting it to the composite implementation.

- [ ] **Step 1: Write failing tests** for segmentation-only, segmentation+ProjCons, DTM monitor, DTM active scheduling, weighted total calculation, no auxiliary data, ordinary class-index target, regions target, ignore label masking, and deep-supervision list output.
- [ ] **Step 2: Run focused tests** with `pytest nnunetv2/tests/unit/test_composite_loss.py -q`; expected failure because the composite interfaces do not exist.
- [ ] **Step 3: Implement** target normalization and final-resolution selection once in the composite loss; calculate each enabled component once and return a named breakdown while exposing a scalar-compatible path.
- [ ] **Step 4: Add compatibility adaptation** so external callers using `trainer.loss(output, target)` still receive a scalar tensor and existing `CombinedSegProjLoss` imports remain valid.
- [ ] **Step 5: Run focused tests**; expected all component and masking tests pass on CPU with finite scalar gradients.
- [ ] **Step 6: Commit** with `refactor: compose segmentation auxiliary losses`.
