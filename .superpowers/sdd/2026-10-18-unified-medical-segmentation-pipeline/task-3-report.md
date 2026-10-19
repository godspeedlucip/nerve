# Task 3 report: composite segmentation, projection, and DTM loss

## Result

Implemented the Task 3 composite loss pipeline and preserved the scalar compatibility path for `CombinedSegProjLoss.forward(output, target)`.

## TDD evidence

- RED: `python -m pytest nnunetv2/tests/unit/test_composite_loss.py -q` failed during collection with `ModuleNotFoundError: nnunetv2.training.loss.composite_loss`, before the production module existed.
- GREEN: the focused suite passed with `7 passed`.
- Full unit suite: `python -m pytest nnunetv2/tests/unit -q` passed with `45 passed`.
- `git diff --check`: passed.

The direct `pytest` executable was unavailable in this environment, so the equivalent module invocation was used. The full suite reports one unrelated pytest-asyncio deprecation warning under Python 3.14.

## Changed files

- `nnunetv2/training/loss/composite_loss.py`: added `LossBreakdown` and `CompositeSegmentationLoss`; centralized final-resolution selection, target normalization, ignore masking, projection calculation, DTM monitor/active scheduling, and missing-auxiliary handling.
- `nnunetv2/network_modules/projection_consistency.py`: adapted `CombinedSegProjLoss` to the composite implementation while retaining its scalar forward signature and import path.
- `nnunetv2/training/loss/dtm_losses.py`: preserved the existing foreground DTM mean and added support for 5D foreground maps alongside the existing 6D stored-DTM layout.
- `nnunetv2/training/loss/__init__.py`: exported the composite loss and breakdown.
- `nnunetv2/tests/unit/test_composite_loss.py`: added focused coverage for all Task 3 behaviors.

## Commit

Commit: `refactor: compose segmentation auxiliary losses` (final hash is recorded in the task handoff)

Planned message: `refactor: compose segmentation auxiliary losses`

## Concerns

- The existing `AuxiliaryTargets` contract requires DTM and validity-mask tensors to have exactly matching shapes, while the legacy DTM loader naturally stores class channels in a 6D DTM tensor and validity masks as 5D tensors. This task keeps the committed Task 1 contract unchanged; a later trainer adapter may need to normalize that legacy shape before constructing `AuxiliaryTargets`.
- The environment does not expose a standalone `pytest` executable; verification used `python -m pytest`.
- User edits in `.gitignore` and `TRAINING_INFERENCE_GUIDE.md` were left untouched.

## Review fix: shared target normalization

- RED: the regression test `test_dtm_receives_projection_normalized_final_target_while_segmentation_keeps_target_tree` failed because DTM received the raw three-channel regions target while projection received the two-channel ignore-masked target.
- GREEN: after introducing one `_NormalizedTargets` result for the final output/target pair and reusing it for both projection and DTM, the focused suite passed with `8 passed`.
- The segmentation loss continues to receive the original deep-supervision target tree unchanged.
- Reviewer minor issues were intentionally left unchanged in this round.
