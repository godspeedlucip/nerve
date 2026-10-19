COMMITS
fed457a (HEAD -> master) docs: record Task 5 verification report
706c0cf test: verify unified enhancement pipeline
STAT
 .../task-5-report.md                               | 42 +++++++++++
 TRAINING_INFERENCE_GUIDE.md                        | 46 ++++++++++++
 nnunetv2/tests/unit/test_unified_pipeline_smoke.py | 73 +++++++++++++++++++
 verify_modules.py                                  | 84 +++++++++++++++++++++-
 4 files changed, 243 insertions(+), 2 deletions(-)
DIFF
diff --git a/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-5-report.md b/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-5-report.md
new file mode 100644
index 0000000..65c9a81
--- /dev/null
+++ b/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-5-report.md
@@ -0,0 +1,42 @@
+# Task 5 report: Documentation, verification, and regression cleanup
+
+## Result
+
+Task 5 is implemented directly on the existing `master` checkout. The new CPU smoke test exercises Aniso + Tubular + ProjCons + active DTM together, computes all loss components, performs backward, and checks finite values and gradients. The verifier now includes the same unified composition path and all existing proofs remain present.
+
+## TDD evidence
+
+- Initial command `pytest nnunetv2/tests/unit/test_unified_pipeline_smoke.py -q`: environment limitation; the `pytest` executable is not on PATH.
+- First `python -m pytest ...`: RED with an `AttributeError` because the new test called the scalar-compatible adapter `forward` while expecting a `LossBreakdown`. This was corrected in the test to call the documented `breakdown(...)` interface.
+- After that test-side correction, the smoke test passed immediately because Tasks 1–4 had already wired the required integration interfaces. No production API change was needed for the smoke path.
+- Existing verifier RED: `python verify_modules.py` reached Proof 5 and failed because it referenced the removed `CombinedSegProjLoss._target_to_onehot` helper. The verifier was minimally updated to build the equivalent one-hot target directly.
+
+## GREEN and verification evidence
+
+- `python -m pytest nnunetv2/tests/unit/test_unified_pipeline_smoke.py -q` → `1 passed, 1 warning`.
+- `python -m pytest nnunetv2/tests/unit -q` → `65 passed, 8 warnings`.
+- `python -m py_compile verify_modules.py nnunetv2/tests/unit/test_unified_pipeline_smoke.py nnunetv2/network_modules/enhancement_pipeline.py nnunetv2/training/auxiliary_targets.py nnunetv2/training/pipeline_config.py nnunetv2/training/pipeline_trainer.py nnunetv2/training/loss/composite_loss.py` → exit 0.
+- Public import check for the unified pipeline modules → `public imports ok`.
+- AST parse check for `verify_modules.py` and the new test → `AST parse ok`.
+- `git diff --check` → no whitespace errors; Git emitted only existing LF/CRLF conversion warnings.
+- `python verify_modules.py` → exit 0; all 8 proofs passed, including `Unified all-enabled composition`.
+
+The pytest warnings are dependency deprecations from `pytest_asyncio`, SciPy namespaces used by `batchgenerators`, and SWIG bindings; they do not fail the suite.
+
+## Changed files
+
+- `nnunetv2/tests/unit/test_unified_pipeline_smoke.py`: all-enabled CPU integration smoke test.
+- `verify_modules.py`: current target normalization compatibility fix plus unified all-enabled composition proof.
+- `TRAINING_INFERENCE_GUIDE.md`: unified `EnhancementConfig` model, legacy preset mapping, and migration example. Existing user edits in the guide were preserved.
+- `.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-5-report.md`: this report.
+
+The pre-existing `.gitignore` edit and other unrelated workspace artifacts were not changed or staged.
+
+## Environment-only limitations
+
+- The bare `pytest` executable is unavailable in this shell, so equivalent commands were run through `python -m pytest`.
+- No dependency prevented the requested final verification: the full unit suite and `verify_modules.py` both completed successfully on CPU.
+
+## Commit
+
+Implementation commit: `706c0cf` — `test: verify unified enhancement pipeline`.
diff --git a/TRAINING_INFERENCE_GUIDE.md b/TRAINING_INFERENCE_GUIDE.md
index fcb45f2..d519d4a 100644
--- a/TRAINING_INFERENCE_GUIDE.md
+++ b/TRAINING_INFERENCE_GUIDE.md
@@ -4,20 +4,21 @@
 
 1. [环境准备](#1-环境准备)
 2. [数据准备](#2-数据准备)
 3. [实验规划与预处理](#3-实验规划与预处理)
 4. [训练](#4-训练)
    - 4.1 [标准训练](#41-标准训练)
    - 4.2 [切换 Trainer](#42-训练时切换-trainer损失函数--策略)
    - 4.3 [DTM 训练](#43-dtm-训练)
    - 4.4 [Dual Containment 训练](#44-dual-containment-训练两网络联合训练)
    - 4.5 [AnisoStent 训练](#45-anisostent-训练网络增强模块)
+   - 4.6 [统一增强配置与迁移](#46-统一增强配置与迁移)
 5. [推理](#5-推理)
 6. [后处理](#6-后处理)
 7. [评估](#7-评估)
 8. [集成与寻找最佳配置](#8-集成与寻找最佳配置)
 9. [模型导出与共享](#9-模型导出与共享)
 10. [模式切换速查](#10-模式切换速查)
 11. [完整工作流示例](#11-完整工作流示例)
 
 ---
 
@@ -191,20 +192,21 @@ nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0
 nnUNetv2_train Dataset001_BrainTumor 3d_fullres all
 
 # 使用自定义 plans
 nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -p nnUNetPlans
 
 # 从断点继续训练
 nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 --c
 
 # 仅验证（无需重新训练）
 nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 --val
+
 # 使用最优 checkpoint 验证
 nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 --val_best
 
 # 加载预训练权重
 nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 \
     -pretrained_weights /path/to/checkpoint.pth
 
 # 多 GPU 训练（DDP）
 nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -num_gpus 4
 
@@ -437,20 +439,64 @@ nnUNetv2_predict \
 
 # 基于模型文件夹
 nnUNetv2_predict_from_modelfolder \
     -i input/ -o output/ \
     -m nnUNet_results/Dataset001_BrainTumor/\
 nnUNetTrainerAnisoStent_ProjCons__nnUNetPlans__3d_fullres
 ```
 
 AnisoStent 网络模块在推理时同样生效——各向异性注意力模块作用在输入上，管状结构增强模块作用在输出上。
 
+#### 4.6 统一增强配置与迁移
+
+Tasks 1–4 将网络增强和辅助损失收敛到一个 `EnhancementConfig`：
+
+| 配置项 | 作用 |
+|------|------|
+| `enable_anisotropic_attention` | 在 backbone 输入前启用 AnisotropicAttention |
+| `enable_tubular_enhancement` | 在最终分辨率输出上启用 TubularEnhancement |
+| `projection_angles_deg` / `projection_consistency_weight` | 启用 ProjectionConsistencyLoss 及其投影角度/权重 |
+| `dtm_mode` / `dtm_weight` / `dtm_active_weight` | 选择 DTM 的 `disabled`、`monitor` 或 `active` 模式及权重 |
+
+现有 trainer 名称仍然兼容，并映射到等价预设：
+
+| 旧 trainer | 统一配置等价项 |
+|------|------|
+| `nnUNetTrainerAnisoStent` | Aniso + Tubular |
+| `nnUNetTrainerAnisoStent_ProjCons` | Aniso + Tubular + ProjCons（`[0, 30, 60]`，权重 `0.1`） |
+| `nnUNetTrainerDTM` | DTM `monitor`（默认在 epoch 250 后可激活） |
+| `nnUNetTrainerDTM_GSL` | DTM `active`（权重 `1.0`，激活后系数 `0.5`） |
+
+新代码可直接构造统一网络和损失；训练循环仍可使用标量兼容的 `forward(output, target)`，需要日志明细时调用 `breakdown(...)`：
+
+```python
+from nnunetv2.training.pipeline_config import EnhancementConfig
+from nnunetv2.training.pipeline_trainer import build_composite_loss, build_network
+
+config = EnhancementConfig(
+    enable_anisotropic_attention=True,
+    enable_tubular_enhancement=True,
+    projection_angles_deg=(0.0, 30.0, 60.0),
+    projection_consistency_weight=0.1,
+    dtm_mode="active",
+    dtm_weight=1.0,
+)
+network = build_network(
+    architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
+    num_input_channels, num_output_channels, enable_deep_supervision, config,
+)
+loss = build_composite_loss(config, segmentation_loss, label_manager)
+breakdown = loss.breakdown(output, target, auxiliary_targets, epoch=epoch)
+```
+
+迁移时保留原 trainer 名称即可逐步切换：先用 `EnhancementConfig.from_legacy_preset("nnUNetTrainerAnisoStent_ProjCons")` 验证等价行为，再将配置显式传给 `build_network` 和 `build_composite_loss`。DTM 模式仍需要预先生成 DTM 文件；推理阶段不需要提供 DTM 输入。
+
 ---
 
 ## 5. 推理
 
 ### 5.1 标准推理（基于数据集）
 
 ```bash
 # 基本用法
 nnUNetv2_predict \
     -i INPUT_FOLDER \
diff --git a/nnunetv2/tests/unit/test_unified_pipeline_smoke.py b/nnunetv2/tests/unit/test_unified_pipeline_smoke.py
new file mode 100644
index 0000000..092338b
--- /dev/null
+++ b/nnunetv2/tests/unit/test_unified_pipeline_smoke.py
@@ -0,0 +1,73 @@
+from types import SimpleNamespace
+
+import torch
+from torch import nn
+
+from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
+from nnunetv2.training.pipeline_config import EnhancementConfig
+from nnunetv2.training.pipeline_trainer import build_composite_loss
+from nnunetv2.network_modules.enhancement_pipeline import EnhancementPipeline
+
+
+class TinyBackbone(nn.Module):
+    def __init__(self, in_channels=1, num_classes=2):
+        super().__init__()
+        self.decoder = nn.Module()
+        self.projection = nn.Conv3d(in_channels, num_classes, kernel_size=1)
+
+    def forward(self, x):
+        return self.projection(x)
+
+
+class TinySegmentationLoss(nn.Module):
+    def forward(self, output, target):
+        return nn.functional.cross_entropy(output, target[:, 0].long())
+
+
+def test_all_enabled_cpu_pipeline_loss_backward_is_finite():
+    torch.manual_seed(0)
+    config = EnhancementConfig(
+        enable_anisotropic_attention=True,
+        enable_tubular_enhancement=True,
+        projection_angles_deg=(0.0,),
+        projection_consistency_weight=0.25,
+        dtm_mode="active",
+        dtm_weight=0.5,
+        dtm_activate_after_epoch=0,
+        dtm_active_weight=1.0,
+    )
+    pipeline = EnhancementPipeline(TinyBackbone(), 1, 2, config).cpu()
+    label_manager = SimpleNamespace(
+        has_regions=False,
+        has_ignore_label=False,
+        ignore_label=None,
+        num_segmentation_heads=2,
+    )
+    loss = build_composite_loss(config, TinySegmentationLoss(), label_manager).cpu()
+
+    image = torch.randn(1, 1, 8, 8, 8)
+    target = torch.randint(0, 2, (1, 1, 8, 8, 8))
+    dtm = torch.rand(1, 1, 8, 8, 8, 2)
+    valid_mask = torch.ones_like(dtm)
+
+    output = pipeline(image)
+    breakdown = loss.breakdown(
+        output,
+        target,
+        AuxiliaryTargets(dtm=dtm, valid_mask=valid_mask),
+        epoch=0,
+    )
+    breakdown.total.backward()
+
+    for component in (
+        breakdown.total,
+        breakdown.segmentation,
+        breakdown.projection,
+        breakdown.dtm,
+    ):
+        assert torch.isfinite(component)
+    assert any(
+        parameter.grad is not None and torch.isfinite(parameter.grad).all()
+        for parameter in pipeline.parameters()
+        if parameter.requires_grad
+    )
diff --git a/verify_modules.py b/verify_modules.py
index b8e2e72..e105ebc 100644
--- a/verify_modules.py
+++ b/verify_modules.py
@@ -23,20 +23,24 @@ print("=" * 60)
 # ── Proof 1: Gradient flow through all 3 modules ────────────────────
 print("\n[Proof 1] Gradient flow through AnisoAttn, TubEnh, ProjCons")
 print("-" * 40)
 
 from nnunetv2.network_modules.anisotropic_attention import AnisotropicAttention
 from nnunetv2.network_modules.tubular_enhancement import TubularEnhancement
 from nnunetv2.network_modules.projection_consistency import (
     ProjectionConsistencyLoss, CombinedSegProjLoss,
 )
 from nnunetv2.network_modules.aniso_stent_unet import AnisoStentUNet
+from nnunetv2.network_modules.enhancement_pipeline import EnhancementPipeline
+from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
+from nnunetv2.training.pipeline_config import EnhancementConfig
+from nnunetv2.training.pipeline_trainer import build_composite_loss
 
 # Build a minimal backbone for testing
 from dynamic_network_architectures.architectures.unet import PlainConvUNet
 backbone = PlainConvUNet(
     input_channels=C, n_stages=4,
     features_per_stage=(8, 16, 32, 64),
     n_conv_per_stage=2,
     n_conv_per_stage_decoder=(2, 2, 2),
     conv_op=nn.Conv3d,
     kernel_sizes=3,
@@ -213,21 +217,23 @@ class MockLabelManager:
 combined = CombinedSegProjLoss(seg_loss, proj_loss, proj_weight=0.1, label_manager=MockLabelManager()).to(DEVICE)
 
 output = torch.randn(B, NUM_CLASSES, 32, 64, 64, device=DEVICE, requires_grad=True)
 target = torch.randint(0, NUM_CLASSES, (B, 1, 32, 64, 64), device=DEVICE)
 
 l_total = combined(output, target)
 l_seg = combined.seg_loss(output, target)
 
 # Compute proj component separately
 with torch.no_grad():
-    target_oh = combined._target_to_onehot(target, NUM_CLASSES)
+    target_oh = torch.nn.functional.one_hot(
+        target[:, 0].long(), NUM_CLASSES
+    ).permute(0, 4, 1, 2, 3).float()
     l_proj = combined.proj_loss(output, target_oh)
     l_proj_weighted = combined.proj_weight * l_proj
 
 print(f"  Total loss:          {l_total.item():.6f}")
 print(f"  DC+CE loss:          {l_seg.item():.6f}")
 print(f"  ProjCons loss:       {l_proj.item():.6f}")
 print(f"  Weighted ProjCons:   {l_proj_weighted.item():.6f}")
 
 # Verify: total = seg + proj_weight * proj
 reconstructed = l_seg + combined.proj_weight * l_proj
@@ -315,39 +321,113 @@ for step in range(5):
     losses.append(l.item())
 
 # Loss should decrease (or at least change meaningfully)
 loss_change = losses[0] - losses[-1]
 print(f"  Loss sequence: {[f'{l:.4f}' for l in losses]}")
 print(f"  {'[PASS]' if abs(loss_change) > 1e-6 else '[FAIL]'} Loss changes during training "
       f"(delta={loss_change:.6f})")
 
 p7_ok = abs(loss_change) > 1e-6
 
+# ── Proof 8: Unified all-enabled composition path ─────────────────────
+print("\n[Proof 8] Unified all-enabled pipeline + composite loss")
+print("-" * 40)
+
+class TinyVerificationBackbone(nn.Module):
+    def __init__(self):
+        super().__init__()
+        self.decoder = nn.Module()
+        self.projection = nn.Conv3d(C, NUM_CLASSES, kernel_size=1)
+
+    def forward(self, inputs):
+        return self.projection(inputs)
+
+
+class TinyVerificationSegmentationLoss(nn.Module):
+    def forward(self, output, target):
+        return nn.functional.cross_entropy(output, target[:, 0].long())
+
+
+unified_config = EnhancementConfig(
+    enable_anisotropic_attention=True,
+    enable_tubular_enhancement=True,
+    projection_angles_deg=(0.0,),
+    projection_consistency_weight=0.25,
+    dtm_mode="active",
+    dtm_weight=0.5,
+    dtm_activate_after_epoch=0,
+    dtm_active_weight=1.0,
+)
+unified_model = EnhancementPipeline(
+    TinyVerificationBackbone(), C, NUM_CLASSES, unified_config
+).to(DEVICE)
+
+class _UnifiedLabelManager:
+    has_regions = False
+    has_ignore_label = False
+    num_segmentation_heads = NUM_CLASSES
+
+
+unified_loss = build_composite_loss(
+    unified_config,
+    TinyVerificationSegmentationLoss(),
+    _UnifiedLabelManager(),
+).to(DEVICE)
+unified_input = torch.randn(1, C, 8, 8, 8, device=DEVICE)
+unified_target = torch.randint(0, NUM_CLASSES, (1, 1, 8, 8, 8), device=DEVICE)
+unified_dtm = torch.rand(1, 1, 8, 8, 8, NUM_CLASSES, device=DEVICE)
+unified_auxiliary = AuxiliaryTargets(
+    dtm=unified_dtm,
+    valid_mask=torch.ones_like(unified_dtm),
+)
+unified_output = unified_model(unified_input)
+unified_breakdown = unified_loss.breakdown(
+    unified_output, unified_target, unified_auxiliary, epoch=0
+)
+unified_breakdown.total.backward()
+unified_finite = all(
+    torch.isfinite(component).item()
+    for component in (
+        unified_breakdown.total,
+        unified_breakdown.segmentation,
+        unified_breakdown.projection,
+        unified_breakdown.dtm,
+    )
+)
+unified_grad = any(
+    parameter.grad is not None and torch.isfinite(parameter.grad).all()
+    for parameter in unified_model.parameters()
+    if parameter.requires_grad
+)
+p8_ok = unified_finite and unified_grad
+print(f"  {'[PASS]' if p8_ok else '[FAIL]'} all components finite and gradients present")
+
 # ── Summary ──────────────────────────────────────────────────────────
 print("\n" + "=" * 60)
 print("SUMMARY")
 print("=" * 60)
 results = {
     "1. Gradient flow (AnisoAttn)": p1_aniso,
     "   Gradient flow (TubEnh)": p1_tuben,
     "   Gradient flow (Backbone)": p1_bb,
     "2. AnisoAttn non-identity": p2_ok,
     "3. TubEnh non-identity": p3_ok,
     "4. ProjCons effectiveness": p4_ok,
     "5. Combined loss balance": p5_ok,
     "6. Projection physics": p6_ok,
     "7. End-to-end training": p7_ok,
+    "8. Unified all-enabled composition": p8_ok,
 }
 
 all_pass = True
 for name, result in results.items():
     status = "[PASS]" if result else "[FAIL]"
     if not result:
         all_pass = False
     print(f"  {status}  Proof {name}")
 
 print("=" * 60)
 if all_pass:
-    print("ALL PROOFS PASSED - All 3 modules are empirically active.")
+    print("ALL PROOFS PASSED - Unified enhancement pipeline is empirically active.")
 else:
     print("SOME PROOFS FAILED - see above for details.")
 print("=" * 60)

