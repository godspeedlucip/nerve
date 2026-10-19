# Unified Medical Segmentation Enhancement Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 DTM/GSL、投影一致性、各向异性条纹抑制和管状结构增强统一为配置驱动、可组合且兼容旧 trainer/checkpoint 的 nnU-Net 训练管线。

**Architecture:** 新增配置对象、辅助目标对象、模型增强管线和 Composite loss；trainer 通过一个装配层组合它们。旧 trainer 类名保留为兼容预设，原有 DTM 文件格式、ProjCons 接口、deep supervision 和 A/B trainer 行为保持不变。

**Tech Stack:** Python 3.10+, PyTorch, nnU-Net v2, pytest, CPU-only unit/smoke tests.

**Spec:** `docs/superpowers/specs/2026-10-18-unified-medical-segmentation-pipeline-design.md`

## Global Constraints

- 不修改 nnU-Net backbone、投影几何定义、DTM 文件格式或已有损失的数学含义。
- 不删除旧 trainer 类名、旧模块导出或旧 `CombinedSegProjLoss.forward(output, target)` 调用路径。
- 保持已有模块 state-dict 前缀 `aniso_attn`、`backbone`、`tub_enh`。
- 现有用户改动 `.gitignore` 和 `TRAINING_INFERENCE_GUIDE.md` 不得被覆盖。
- 测试不得依赖 GPU、真实数据集或 ANTsPy；DTM 算法本身继续由现有实现提供。

## Review Focus

- 关闭全部增强模块时 wrapper 必须等价于 backbone，且 deep supervision 输出结构不变；测试归 Task 2。
- 启用 DTM 但 batch 缺失辅助数据时必须返回零 DTM 项并给出一次明确 warning，不得误把 active 模式当成已生效；测试归 Task 3。
- regions、ignore label、tensor/list deep supervision 三种 target 形态必须统一处理；测试归 Task 3。
- 旧 checkpoint 的模块命名和旧 trainer loss 调用必须保持兼容；测试归 Task 4。
- A/B dual trainer 只能改变结果目录身份，不得改变模型或 loss 配置；测试归 Task 4。

---

### Task 1: Configuration and auxiliary target contracts

**Files:**
- Create: `nnunetv2/training/pipeline_config.py`
- Create: `nnunetv2/training/auxiliary_targets.py`
- Modify: `nnunetv2/training/__init__.py`
- Test: `nnunetv2/tests/unit/test_pipeline_contracts.py`

**Interfaces:**
- Produces `EnhancementConfig`, `AuxiliaryTargets`, and `AuxiliaryTargets.empty()` for Tasks 2–4.
- `EnhancementConfig` exposes validated fields named in the spec and a `from_legacy_preset(name: str)` factory for compatibility presets.
- `AuxiliaryTargets.to(device)` and `AuxiliaryTargets.final_resolution()` preserve tensor/list shape semantics.

- [ ] **Step 1: Write failing tests** for default/legacy preset values, invalid negative weights, invalid DTM modes, recursive device movement, empty auxiliary targets, and mismatched batch/shape validation.
- [ ] **Step 2: Run the focused tests** with `pytest nnunetv2/tests/unit/test_pipeline_contracts.py -q`; expected failure because the new contracts do not exist.
- [ ] **Step 3: Implement** the frozen-style configuration dataclass and auxiliary target dataclass with explicit validation and shape errors containing actual shapes.
- [ ] **Step 4: Run the focused tests** again; expected all contract tests pass on CPU.
- [ ] **Step 5: Commit** with `feat: add unified pipeline contracts`.

### Task 2: Unified model enhancement pipeline

**Files:**
- Create: `nnunetv2/network_modules/enhancement_pipeline.py`
- Modify: `nnunetv2/network_modules/aniso_stent_unet.py`
- Modify: `nnunetv2/network_modules/__init__.py`
- Test: `nnunetv2/tests/unit/test_enhancement_pipeline.py`

**Interfaces:**
- Consumes `EnhancementConfig` from Task 1.
- Produces `EnhancementPipeline(backbone, in_channels, num_classes, config)` with delegated `decoder` and stable module attributes `aniso_attn`, `backbone`, `tub_enh`.
- `forward(x)` returns a tensor or list matching the backbone output type; TubularEnhancement only transforms index 0 of a deep-supervision list.

- [ ] **Step 1: Write failing tests** for all-off identity behavior, Aniso-only input transformation, Tubular-only final-output transformation, simultaneous Aniso+Tubular composition, deep supervision auxiliary-output preservation, decoder delegation, and stable state-dict keys.
- [ ] **Step 2: Run focused tests** with `pytest nnunetv2/tests/unit/test_enhancement_pipeline.py -q`; expected failure because `EnhancementPipeline` does not exist.
- [ ] **Step 3: Implement** the pipeline using existing `AnisotropicAttention` and `TubularEnhancement`; instantiate disabled modules as identity-safe paths while retaining stable compatibility attribute names.
- [ ] **Step 4: Adapt `AnisoStentUNet`** to delegate to or share the pipeline implementation without changing its constructor or state-dict names.
- [ ] **Step 5: Run focused tests** and the existing `verify_modules.py` CPU-compatible portion; expected all pipeline tests pass and existing module imports remain valid.
- [ ] **Step 6: Commit** with `refactor: unify enhancement model pipeline`.

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

### Task 5: Documentation, verification, and regression cleanup

**Files:**
- Modify: `TRAINING_INFERENCE_GUIDE.md` only where the new unified configuration/preset model needs explanation; preserve existing user edits.
- Modify: `verify_modules.py` to call the unified composition path where practical without removing existing proofs.
- Test: `nnunetv2/tests/unit/test_unified_pipeline_smoke.py`

**Interfaces:**
- Consumes all public interfaces from Tasks 1–4.
- Produces a documented migration path and a single CPU smoke test covering Aniso + Tubular + ProjCons + DTM active together.

- [ ] **Step 1: Write failing smoke test** that builds a tiny backbone and all-enabled pipeline, computes the composite loss with DTM auxiliary targets, performs backward, and asserts all enabled component losses are finite.
- [ ] **Step 2: Run the smoke test** with `pytest nnunetv2/tests/unit/test_unified_pipeline_smoke.py -q`; expected failure until all integration interfaces are wired.
- [ ] **Step 3: Implement** only the minimal documentation and verification-script changes required by the final public interfaces.
- [ ] **Step 4: Run the complete unit suite** with `pytest nnunetv2/tests/unit -q`; expected zero failures.
- [ ] **Step 5: Run static/import checks** for the touched modules and the existing module verification entrypoint where dependencies permit; record any environment-only limitations separately.
- [ ] **Step 6: Commit** with `test: verify unified enhancement pipeline`.

## Final verification

- Run `pytest nnunetv2/tests/unit -q` and read the complete result.
- Run `git diff --check`.
- Run `git status --short` and confirm only intended new/modified files are present; `.gitignore` and `TRAINING_INFERENCE_GUIDE.md` user changes must remain intact.
- Review the final diff against the spec and verify every legacy trainer mapping listed in the spec.
