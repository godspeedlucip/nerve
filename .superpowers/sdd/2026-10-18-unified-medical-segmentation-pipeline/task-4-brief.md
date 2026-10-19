### Task 4: Trainer assembly and legacy presets

**Files:**
- Create: `nnunetv2/training/pipeline_trainer.py`
- Modify: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_AnisoStent.py`
- Modify: `nnunetv2/training/nnUNetTrainer/nnUNetTrainerDTM.py`
- Modify: `nnunetv2/training/nnUNetTrainer/dual_network_trainers.py`
- Test: `nnunetv2/tests/unit/test_legacy_presets.py`

**Interfaces:**
- Consumes `EnhancementConfig`, `AuxiliaryTargets`, `EnhancementPipeline`, and `CompositeSegmentationLoss` from Tasks 1–3.
- Produces shared trainer assembly helpers for network/loss construction, auxiliary batch movement, component logging, and train/validation loss calculation.
- Preserves class names `nnUNetTrainerAnisoStent`, `nnUNetTrainerAnisoStent_ProjCons`, `nnUNetTrainerDTM`, `nnUNetTrainerDTM_GSL`, their A/B variants, and their existing default settings.

- [ ] **Step 1: Write failing tests** that import every legacy trainer, assert each maps to the expected config preset, assert A/B pairs have equal configs, and verify `trainer.loss(output, target)` remains callable.
- [ ] **Step 2: Run focused tests** with `pytest nnunetv2/tests/unit/test_legacy_presets.py -q`; expected failure because preset assembly is not centralized.
- [ ] **Step 3: Implement** a shared trainer assembly layer that builds the configured network and composite loss, handles optional DTM fields, and logs `seg_loss`, `proj_loss`, and `gsl_loss` from the same breakdown used for backpropagation.
- [ ] **Step 4: Convert legacy trainers** to thin preset classes while retaining public constructor signatures, DTM dataloader/preprocessor entry points, A/B names, and checkpoint-compatible module names.
- [ ] **Step 5: Run focused tests** and a minimal CPU trainer smoke test; expected all legacy imports, preset mappings, and one forward/backward step pass.
- [ ] **Step 6: Commit** with `refactor: assemble legacy trainers from presets`.
